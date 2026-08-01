"""A lightweight trajectory (sequence) replay buffer for the SimDist world model.

ibrl's main `rl/replay.py::ReplayBuffer` wraps `rela.SingleStepTransitionReplay`,
whose `sample` returns a SINGLE-STEP transition (`obs`, `next_obs` is n-step
later, `reward` is the n-step discounted sum, `bootstrap = gamma**n`). The
intermediate steps are not recoverable, so it cannot supply the length-(H+1)
contiguous trajectory segments the TD-MPC2-style world model needs for its
multi-step latent rollout loss.

This module adds a minimal PyTorch sequence buffer (TD-MPC2's own design):
episodes are stored whole and `sample(batch_size, horizon)` returns
`(states[B,H+1,obs], actions[B,H,act], rewards[B,H])`. Both the SOURCE online
rollouts (stage 1) and the TARGET demos (stage 2) are stored here. State-only
(use_state=1); the world model consumes `obs["state"]`.
"""

import json
import os
from collections import defaultdict

import h5py
import numpy as np
import torch

from env.robosuite_wrapper import STATE_KEYS, PROP_KEYS


class SequenceReplayBuffer:
    """Stores whole episodes; samples contiguous (H+1)-step segments.

    Episodes shorter than `horizon` are skipped at sample time. Sampling picks
    an episode with probability proportional to its number of valid start
    indices (= max(T - horizon + 1, 0)), then a uniform start within it — i.e.
    uniform over all valid (episode, start) pairs.
    """

    def __init__(self, max_episodes: int = 5000, device: str = "cpu"):
        self.max_episodes = max_episodes
        self.device = device
        self.episodes: list[dict] = []  # each: state[T+1,obs], action[T,act], reward[T]
        self._cur: dict | None = None
        self.num_episode = 0
        self.num_success = 0

    # ---------------------------------------------------------------- online
    def new_episode(self, state: torch.Tensor):
        """Begin a new episode with the initial state (obs['state'])."""
        self._cur = {"state": [state.detach().cpu().float()], "action": [], "reward": []}

    def add(self, next_state: torch.Tensor, action: torch.Tensor, reward: float,
            terminal: bool, success: bool = False):
        assert self._cur is not None, "call new_episode() before add()"
        self._cur["action"].append(action.detach().cpu().float())
        self._cur["reward"].append(float(reward))
        self._cur["state"].append(next_state.detach().cpu().float())
        if terminal:
            self._flush(success)

    def _flush(self, success: bool):
        cur = self._cur
        self._cur = None
        if cur is None or len(cur["action"]) == 0:
            return
        ep = {
            "state": torch.stack(cur["state"]),    # [T+1, obs]
            "action": torch.stack(cur["action"]),  # [T, act]
            "reward": torch.tensor(cur["reward"], dtype=torch.float32),  # [T]
            "success": bool(success),
        }
        self.episodes.append(ep)
        self.num_episode += 1
        if success:
            self.num_success += 1
        if len(self.episodes) > self.max_episodes:
            # num_success tracks the buffer's CURRENT contents (it's logged next
            # to size()/num_transitions()), so decrement when a success is evicted.
            # num_episode stays a lifetime counter (drives the warm-up gate).
            if self.episodes.pop(0)["success"]:
                self.num_success -= 1

    # --------------------------------------------------------------- offline
    def add_episode(self, states: torch.Tensor, actions: torch.Tensor,
                    rewards: torch.Tensor, success: bool = False,
                    expert_flag: torch.Tensor | None = None):
        """Add a complete episode. states[T+1,obs], actions[T,act], rewards[T].

        expert_flag[T] (optional): the paper's per-step 1_e(a_t) indicator — 1 iff
        a_t came from the uncorrupted expert policy (expert episode AND no noise
        this step), 0 otherwise. Used to mask the policy-BC loss. None => the
        episode's per-step flag is taken as its `success` value (a coarse fallback
        for data that doesn't carry expert tags, e.g. a plain RL replay dump)."""
        assert states.shape[0] == actions.shape[0] + 1
        assert actions.shape[0] == rewards.shape[0]
        ep = {
            "state": states.detach().cpu().float(),
            "action": actions.detach().cpu().float(),
            "reward": rewards.detach().cpu().float(),
            "success": bool(success),
        }
        if expert_flag is not None:
            ef = expert_flag.detach().cpu().float()
            assert ef.shape[0] == actions.shape[0]
            ep["expert_flag"] = ef
        self.episodes.append(ep)
        self.num_episode += 1
        if success:
            self.num_success += 1
        if len(self.episodes) > self.max_episodes:
            # see _flush: num_success mirrors the buffer's current contents.
            if self.episodes.pop(0)["success"]:
                self.num_success -= 1

    # ---------------------------------------------------------------- sample
    def size(self) -> int:
        return len(self.episodes)

    def num_transitions(self) -> int:
        return int(sum(e["action"].shape[0] for e in self.episodes))

    def can_sample(self, horizon: int) -> bool:
        return any(e["action"].shape[0] >= horizon for e in self.episodes)

    def state_stats(self):
        """Per-dim (mean, std) over ALL stored states — for the raw-state world
        model's z-score normalization. Returns (mean[obs], std[obs]) on CPU."""
        assert self.episodes, "empty buffer; cannot compute state stats"
        all_states = torch.cat([e["state"] for e in self.episodes], dim=0)  # [N, obs]
        return all_states.mean(0), all_states.std(0)

    def num_success_sampleable(self, horizon: int) -> int:
        """# of SUCCESS episodes long enough to sample a horizon-segment from."""
        return sum(1 for e in self.episodes
                   if e.get("success", False) and e["action"].shape[0] >= horizon)

    def _value_to_go(self, ep: dict, discount: float) -> torch.Tensor:
        """MC discounted return-to-go per STATE, cached per episode.
        rewards r[0..T-1] over states s[0..T]: v2g[T]=0, v2g[t]=r[t]+disc*v2g[t+1].
        Sparse 0/1 + end_on_success => disc^(steps-to-success) on success, 0 on fail."""
        if ep.get("_v2g_disc") == discount and "_v2g" in ep:
            return ep["_v2g"]
        r = ep["reward"]
        T = r.shape[0]
        v2g = torch.zeros(T + 1, dtype=torch.float32)
        for t in range(T - 1, -1, -1):
            v2g[t] = r[t] + discount * v2g[t + 1]
        ep["_v2g"] = v2g
        ep["_v2g_disc"] = discount
        return v2g

    def sample(self, batch_size: int, horizon: int, device: str = "cuda",
               return_mask: bool = False, success_only: bool = False,
               return_value: bool = False, discount: float = 0.99,
               recency_alpha: float = 0.0):
        """Return contiguous segments:
        states[B, H+1, obs], actions[B, H, act], rewards[B, H].

        If `return_mask`, also returns `policy_mask[B, H]` (float) — the paper's
        per-step 1_e(a_t) indicator aligned with `actions`, used to restrict the
        policy-BC loss to uncorrupted-expert actions while the dynamics/value/
        reward heads still train on ALL data. When an episode carries no
        `expert_flag` (e.g. a plain RL replay), its mask falls back to a constant
        equal to its `success` flag.

        If `success_only`, sample ONLY from episodes flagged success=True (for
        self-distilling the policy on the planner's SUCCESSFUL rollout actions).

        If `return_value`, also returns `value_targets[B, H+1]` — the Monte-Carlo
        discounted return-to-go per state (for value-head finetuning). Appended
        LAST (after policy_mask if both are requested).

        If `recency_alpha > 0`, bias episode selection toward RECENTLY-collected
        episodes ("balance sampling"): the base length-proportional weight of the
        k-th valid episode (in chronological buffer order, 0=oldest .. N-1=newest)
        is multiplied by `exp(recency_alpha * k/(N-1))`, so the newest episode is
        `exp(recency_alpha)`x more likely per start-index than the oldest, with a
        smooth exponential ramp in between. `recency_alpha=0` (default) recovers
        the plain uniform-over-(episode,start) sampling byte-for-byte.
        """
        valid = [(i, e["action"].shape[0]) for i, e in enumerate(self.episodes)
                 if e["action"].shape[0] >= horizon
                 and (not success_only or e.get("success", False))]
        assert valid, f"no {'success ' if success_only else ''}episode of length >= horizon={horizon}"
        idxs = np.array([i for i, _ in valid])
        weights = np.array([T - horizon + 1 for _, T in valid], dtype=np.float64)
        if recency_alpha > 0 and len(valid) > 1:
            # `valid` is built by enumerate() over self.episodes, i.e. oldest->newest;
            # rank in [0,1] over the valid list, exponential recency ramp.
            rank = np.arange(len(valid), dtype=np.float64) / (len(valid) - 1)
            weights *= np.exp(recency_alpha * rank)
        weights /= weights.sum()

        states, actions, rewards, mask, values = [], [], [], [], []
        chosen = np.random.choice(len(idxs), size=batch_size, p=weights)
        for c in chosen:
            ep = self.episodes[idxs[c]]
            T = ep["action"].shape[0]
            start = np.random.randint(0, T - horizon + 1)
            states.append(ep["state"][start:start + horizon + 1])
            actions.append(ep["action"][start:start + horizon])
            rewards.append(ep["reward"][start:start + horizon])
            if return_mask:
                if "expert_flag" in ep:
                    mask.append(ep["expert_flag"][start:start + horizon])
                else:
                    fill = 1.0 if ep.get("success", False) else 0.0
                    mask.append(torch.full((horizon,), fill, dtype=torch.float32))
            if return_value:
                values.append(self._value_to_go(ep, discount)[start:start + horizon + 1])

        states = torch.stack(states).to(device)    # [B, H+1, obs]
        actions = torch.stack(actions).to(device)  # [B, H, act]
        rewards = torch.stack(rewards).to(device)  # [B, H]
        out = [states, actions, rewards]
        if return_mask:
            out.append(torch.stack(mask).to(device))     # [B, H]
        if return_value:
            out.append(torch.stack(values).to(device))   # [B, H+1]
        return tuple(out) if (return_mask or return_value) else (states, actions, rewards)


def load_demos_to_sequence_buffer(
    buffer: SequenceReplayBuffer,
    data_path: str,
    num_data: int,
    reward_scale: float = 1.0,
    clamp_action: bool = True,
):
    """Load TARGET demos (collect_target_demos.py format) into a sequence buffer.

    Mirrors `rl/replay.py::add_demos_to_replay`'s parsing so the state layout is
    byte-for-byte the same as the env's: state is re-concatenated from
    STATE_KEYS[task_name] (NOT the precomputed obs/state), guaranteeing the
    world-model input dim matches `train_env.observation_shape`. Each demo is
    truncated at the first success step (reward==1), matching is_demo=True.
    """
    f = h5py.File(data_path, "r")
    cfg_path = os.path.join(os.path.dirname(data_path), "env_cfg.json")
    task_name = json.load(open(cfg_path, "r"))["env_name"]
    state_keys = STATE_KEYS[task_name]

    num_episode = len(list(f["data"].keys()))
    print(f"[seq-buffer] loading first {num_data} of {num_episode} demos from {data_path}")
    loaded = 0
    for episode_id in range(num_episode):
        if num_data > 0 and episode_id >= num_data:
            break
        episode = f[f"data/demo_{episode_id}"]
        actions = np.array(episode["actions"]).astype(np.float32)
        rewards = np.array(episode["rewards"]).astype(np.float32)
        if not (rewards == 1).any():
            print(f"  skipping demo_{episode_id}: no success step")
            continue

        # state: concat STATE_KEYS (same as env), prop: concat PROP_KEYS
        all_states = [np.array(episode["obs"][k]) for k in state_keys]
        states = np.concatenate(all_states, axis=1).astype(np.float32)
        assert states.shape[0] == actions.shape[0]

        # truncate at first success
        success_idx = int(np.argmax(rewards == 1))
        T = success_idx + 1
        states_seg = states[: T + 1] if states.shape[0] > T else states[: T]
        # obs has the same length as actions (T_full); state[t] is BEFORE action[t].
        # We need T+1 states (s_0..s_T) but obs only stores s_0..s_{T_full-1}.
        # Build next-state from obs[t+1]; for the terminal step reuse the last obs.
        seg_states = states[: T]  # s_0 .. s_{T-1}
        # append s_T: obs at index T if available else duplicate last
        s_last = states[T] if states.shape[0] > T else states[T - 1]
        seq = np.concatenate([seg_states, s_last[None]], axis=0)  # [T+1, obs]
        seg_actions = actions[: T]
        seg_rewards = rewards[: T] * reward_scale
        if clamp_action:
            seg_actions = np.clip(seg_actions, -1.0, 1.0)

        buffer.add_episode(
            torch.from_numpy(seq),
            torch.from_numpy(seg_actions),
            torch.from_numpy(seg_rewards),
            success=True,
        )
        loaded += 1
    f.close()
    print(f"[seq-buffer] loaded {loaded} demos; #episodes={buffer.size()}, "
          f"#transitions={buffer.num_transitions()}")


def load_source_replay_to_sequence_buffer(
    buffer: SequenceReplayBuffer,
    data_path: str,
    num_data: int = -1,
    reward_scale: float = 1.0,
    clamp_action: bool = True,
):
    """Load a MIXED source replay (rl/replay.py dump format:
    `data/demo_N/{obs/state[T,obs], actions[T,act], rewards[T]}`) into a sequence
    buffer for world-model distillation — the SimDist方案B data path.

    Differs from `load_demos_to_sequence_buffer` (TARGET demos) in two ways that
    are the whole point of this loader:
      * NO first-success truncation — the ENTIRE trajectory (incl. suboptimal /
        failed exploration) is kept. That broader state-action coverage is exactly
        what strengthens the dynamics head's generalization (the original SimDist
        mixes expert + non-expert checkpoints for the same reason).
      * `obs/state` is already the 37-dim env state, used directly (no STATE_KEYS
        re-concat needed — this dump was produced by the same env).
    The terminal next-state s_T is not stored (single-step replay drops it), so it
    is duplicated from s_{T-1} (an absorbing-state approximation under
    end_on_success), matching `load_demos_to_sequence_buffer`. This keeps the
    reward==1 success step (dropping the last step instead would lose it).
    """
    f = h5py.File(data_path, "r")
    g = f["data"]
    ids = sorted((k for k in g.keys() if k.startswith("demo_")),
                 key=lambda k: int(k.split("_")[1]))
    if num_data is not None and num_data > 0:
        ids = ids[:num_data]
    print(f"[seq-buffer] loading {len(ids)} source-replay trajectories from {data_path}")
    loaded = 0
    for ep_id in ids:
        ep = g[ep_id]
        states = np.array(ep["obs"]["state"]).astype(np.float32)  # [T, obs] = s_0..s_{T-1}
        actions = np.array(ep["actions"]).astype(np.float32)      # [T, act]
        raw_rewards = np.array(ep["rewards"]).astype(np.float32)   # [T]
        T = actions.shape[0]
        assert states.shape[0] == T, f"{ep_id}: state len {states.shape[0]} != action len {T}"
        if T < 1:
            continue
        # s_T := s_{T-1} (terminal next-state not stored; absorbing approx)
        seq = np.concatenate([states, states[-1:]], axis=0)        # [T+1, obs]
        if clamp_action:
            actions = np.clip(actions, -1.0, 1.0)
        rewards = raw_rewards * reward_scale
        # paper's 1_e(a_t): present in generate_simdist_data.py output, absent in
        # a plain RL replay dump (falls back to success-broadcast in sample()).
        eflag = (torch.from_numpy(np.array(ep["expert_flags"]).astype(np.float32))
                 if "expert_flags" in ep else None)
        buffer.add_episode(
            torch.from_numpy(seq),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            success=bool((raw_rewards == 1).any()),
            expert_flag=eflag,
        )
        loaded += 1
    f.close()
    print(f"[seq-buffer] loaded {loaded} source trajectories; #episodes={buffer.size()}, "
          f"#transitions={buffer.num_transitions()}, #success={buffer.num_success}")
