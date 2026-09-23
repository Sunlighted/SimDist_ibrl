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

from env.robosuite_wrapper import STATE_KEYS, PROP_KEYS, PROP_DIM


# Reserved obs-dict key carrying the proprio+action history (see rl/history.py).
HISTORY_KEY = "__hist"


class JpegFrames:
    """A camera's frames held as per-frame JPEG bytes instead of a raw uint8 array.

    Why: the pixel buffer is CPU-RAM resident, and at 224x224x3 with two cameras
    the raw store is 301,056 B/transition -- 6.1 TB for the 100k-trajectory
    reference recipe. JPEG (measured 15.9x on our renders) keeps it near 340 GB.
    Frames stay ENCODED in RAM and are decoded per batch in `sample()`, so the
    saving is a RAM saving, not merely a disk one.

    Layout: ONE concatenated `blob` per (episode, camera) plus `starts`/`ends`
    index arrays -- NOT a Python list of per-frame arrays. At 8M transitions x 2
    cameras that would be 16M live numpy objects (~150 B of interpreter overhead
    each, plus allocator fragmentation and GC pressure); this is 2 objects per
    episode-camera. `starts`/`ends` are explicit rather than a monotonic offset
    array so a frame can be REPEATED (terminal-step padding) or dropped
    (truncation) without copying the blob.

    Slicing returns a list of 1-D uint8 VIEWS into the blob; `decode_jpeg_batch`
    turns a flat list back into a [N,C,H,W] uint8 tensor in ONE call (nvJPEG when
    the target is cuda).
    """

    __slots__ = ("blob", "starts", "ends", "frame_shape")

    def __init__(self, encoded, frame_shape, _blob=None, _starts=None, _ends=None):
        self.frame_shape = tuple(int(x) for x in frame_shape)   # (C,H,W)
        if _blob is not None:
            self.blob, self.starts, self.ends = _blob, _starts, _ends
            return
        enc = [np.ascontiguousarray(b, dtype=np.uint8).ravel() for b in encoded]
        sizes = np.array([b.size for b in enc], dtype=np.int64)
        self.ends = np.cumsum(sizes)
        self.starts = self.ends - sizes
        self.blob = (np.concatenate(enc) if enc
                     else np.zeros(0, dtype=np.uint8))

    def __len__(self):
        return int(self.starts.shape[0])

    def _view(self, i: int):
        return self.blob[self.starts[i]:self.ends[i]]

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            return [self._view(i) for i in range(*sl.indices(len(self)))]
        return self._view(int(sl))

    def _reindex(self, idxs):
        idxs = np.asarray(idxs, dtype=np.int64)
        return JpegFrames(None, self.frame_shape, _blob=self.blob,
                          _starts=self.starts[idxs], _ends=self.ends[idxs])

    @property
    def shape(self):
        return (len(self),) + self.frame_shape

    def nbytes_encoded(self) -> int:
        return int((self.ends - self.starts).sum())

    def truncate_pad(self, T: int):
        """frames[:T] + frames[T] (or a duplicate of frames[T-1]) -- the exact
        truncate-and-pad the demo loader applies to `state`. Diverging here
        desynchronises frames from states at the terminal step, which is where
        the only reward lives."""
        n = len(self)
        last = T if n > T else T - 1
        return self._reindex(list(range(T)) + [last])

    def pad_last(self, n: int = 1):
        """s_T := s_{T-1}, matching how `state` is padded at the terminal step."""
        if len(self) == 0:
            return self
        return self._reindex(list(range(len(self))) + [len(self) - 1] * n)


def decode_jpeg_batch(flat: list, device: str):
    """list of 1-D uint8 arrays -> uint8 tensor [N,C,H,W] on `device`."""
    import torchvision.io as tvio
    bufs = [torch.from_numpy(np.ascontiguousarray(b)) for b in flat]
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    try:
        imgs = tvio.decode_jpeg(bufs, mode=tvio.ImageReadMode.RGB, device=dev)
    except (RuntimeError, TypeError):
        # older torchvision (no batched/list form) or no nvJPEG -> per-frame CPU
        imgs = [tvio.decode_jpeg(b, mode=tvio.ImageReadMode.RGB) for b in bufs]
    return torch.stack(imgs) if isinstance(imgs, list) else imgs


def read_frames_array(ds, idx=None) -> np.ndarray:
    """hdf5 image dataset -> DECODED np.uint8 [T,C,H,W], whatever the codec.

    For call sites that need a materialised array rather than the buffer's
    lazily-decoded JpegFrames -- e.g. the stage-2 anchor pools, which encode a
    bounded subset of episodes into latents ONCE and then drop the frames. A bare
    `np.asarray(ds)` on a vlen JPEG dataset yields an OBJECT array, which fails
    far downstream (or worse, silently), so every image read must come through
    here or through read_frames_dataset.

    `idx`: read/decode ONLY these frame indices. The anchor pool keeps a random
    40k-state subsample, so materialising every frame of every sampled episode
    first is pure waste -- and at 224 it is 108 GB of host RAM, which OOMs the
    node before the encoder is ever reached.
    """
    rows = range(ds.shape[0]) if idx is None else [int(i) for i in idx]
    if ds.attrs.get("codec", "raw") == "jpeg":
        enc = [np.asarray(ds[i], dtype=np.uint8) for i in rows]
        return decode_jpeg_batch(enc, "cpu").numpy().astype(np.uint8)
    arr = np.array(ds) if idx is None else np.stack([ds[i] for i in rows])
    assert arr.dtype == np.uint8, f"expected uint8 frames, got {arr.dtype}"
    return arr


def _stack_frames(per_sample: list, device: str):
    """per_sample: list (len B) of either a [H+1,C,H,W] uint8 tensor (legacy raw)
    or a list of H+1 encoded JPEG buffers. Returns uint8 [B,H+1,C,H,W]."""
    if per_sample and isinstance(per_sample[0], list):
        b = len(per_sample)
        h1 = len(per_sample[0])
        flat = [buf for seg in per_sample for buf in seg]
        imgs = decode_jpeg_batch(flat, device)                 # [B*(H+1),C,H,W]
        return imgs.view(b, h1, *imgs.shape[1:]).to(device)
    return torch.stack(per_sample).to(device)


def read_frames_dataset(ds):
    """hdf5 image dataset -> np.uint8 [T,C,H,W] (raw) or JpegFrames (compressed).

    The codec flag lives in the DATASET attrs, which is why the generator's shard
    merge must use h5py's native copy -- a manual create_dataset(data=...) walk
    silently drops them and the frames come back as an unreadable object array."""
    if ds.attrs.get("codec", "raw") == "jpeg":
        return JpegFrames([np.asarray(ds[i], dtype=np.uint8) for i in range(ds.shape[0])],
                          ds.attrs["frame_shape"])
    arr = np.array(ds)
    assert arr.dtype == np.uint8, f"expected uint8 frames, got {arr.dtype}"
    return arr


class SequenceReplayBuffer:
    """Stores whole episodes; samples contiguous (H+1)-step segments.

    Episodes shorter than `horizon` are skipped at sample time. Sampling picks
    an episode with probability proportional to its number of valid start
    indices (= max(T - horizon + 1, 0)), then a uniform start within it — i.e.
    uniform over all valid (episode, start) pairs.
    """

    def __init__(self, max_episodes: int = 5000, device: str = "cpu",
                 image_keys: tuple = (), max_transitions: int = 0):
        """image_keys: camera names to store alongside `state` for the PIXEL arm.
        () (default) = legacy behaviour, byte-for-byte -- important because
        train_tdmpc2.py, train_flow_qc_ug.py and train_flow_ug_dynadapt.py all
        import this class and must not change.

        Frames are kept as UINT8 (55,296 B/transition for two 96x96 cameras;
        float32 would be 4x for zero benefit). The 37-dim `state` is ALSO kept on
        the pixel arm: it costs 148 B and the frozen state critic needs it to
        produce V_src labels (train_simdist.py:354-367 builds {"state": ...} by
        hand), so dropping it would break value distillation.

        max_transitions: 0 = unlimited (legacy). >0 evicts oldest episodes until
        under budget -- episodes are the wrong unit when a frame costs 27 KB.
        """
        self.max_episodes = max_episodes
        self.device = device
        self.image_keys = tuple(image_keys)
        self.max_transitions = int(max_transitions)
        self.episodes: list[dict] = []  # each: state[T+1,obs], action[T,act], reward[T]
        self._cur: dict | None = None
        self.num_episode = 0
        self.num_success = 0
        self._n_trans = 0               # running count, so eviction is not O(N)

    @staticmethod
    def _frame(v: torch.Tensor) -> torch.Tensor:
        """Store a camera frame verbatim as uint8 on CPU. The env renders on the
        GPU when rl_image_size != image_size, hence the explicit copy: it makes the
        buffer independent of the wrapper's internal tensors for 27 KB."""
        assert v.dtype == torch.uint8, f"expected uint8 frame, got {v.dtype}"
        return v.detach().to("cpu", copy=True)

    # ---------------------------------------------------------------- online
    def new_episode(self, state: torch.Tensor, obs: dict | None = None):
        """Begin a new episode with the initial state (obs['state']).

        `obs` is required iff image_keys is non-empty, and must contain a frame
        for every image key. It is the FULL obs dict for the same timestep as
        `state` -- the alignment between them is load-bearing (see sample()).
        """
        self._cur = {"state": [state.detach().cpu().float()], "action": [], "reward": []}
        if self.image_keys:
            assert obs is not None, "image_keys set -> new_episode(state, obs=...) required"
            self._cur["obs"] = {k: [self._frame(obs[k])] for k in self.image_keys}

    def add(self, next_state: torch.Tensor, action: torch.Tensor, reward: float,
            terminal: bool, success: bool = False, next_obs: dict | None = None):
        assert self._cur is not None, "call new_episode() before add()"
        self._cur["action"].append(action.detach().cpu().float())
        self._cur["reward"].append(float(reward))
        self._cur["state"].append(next_state.detach().cpu().float())
        if self.image_keys:
            assert next_obs is not None, "image_keys set -> add(..., next_obs=...) required"
            for k in self.image_keys:
                self._cur["obs"][k].append(self._frame(next_obs[k]))
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
        if self.image_keys:
            # [T+1, C, H, W] uint8 per camera, index-aligned with ep["state"].
            ep["obs"] = {k: torch.stack(v) for k, v in cur["obs"].items()}
            for k, v in ep["obs"].items():
                assert v.shape[0] == ep["state"].shape[0], (
                    f"image/state length mismatch for {k}: "
                    f"{v.shape[0]} vs {ep['state'].shape[0]}")
        self.episodes.append(ep)
        self.num_episode += 1
        self._n_trans += int(ep["action"].shape[0])
        if success:
            self.num_success += 1
        self._evict_for_transitions()
        if len(self.episodes) > self.max_episodes:
            # num_success tracks the buffer's CURRENT contents (it's logged next
            # to size()/num_transitions()), so decrement when a success is evicted.
            # num_episode stays a lifetime counter (drives the warm-up gate).
            ev = self.episodes.pop(0)
            self._n_trans -= int(ev["action"].shape[0])
            if ev["success"]:
                self.num_success -= 1

    def _evict_for_transitions(self):
        """Cap the buffer by TRANSITIONS, not episodes. With uint8 frames at
        ~55 KB/transition for two cameras, `max_episodes` is the wrong unit: 400k
        transitions is ~22 GB, which competes with the eval workers' own envs for
        node RAM. No-op when max_transitions == 0 (legacy)."""
        if self.max_transitions <= 0:
            return
        while len(self.episodes) > 1 and self._n_trans > self.max_transitions:
            ev = self.episodes.pop(0)
            self._n_trans -= int(ev["action"].shape[0])
            if ev["success"]:
                self.num_success -= 1

    # --------------------------------------------------------------- offline
    def add_episode(self, states: torch.Tensor, actions: torch.Tensor,
                    rewards: torch.Tensor, success: bool = False,
                    expert_flag: torch.Tensor | None = None,
                    obs: dict | None = None):
        """Add a complete episode. states[T+1,obs], actions[T,act], rewards[T].

        expert_flag[T] (optional): the paper's per-step 1_e(a_t) indicator — 1 iff
        a_t came from the uncorrupted expert policy (expert episode AND no noise
        this step), 0 otherwise. Used to mask the policy-BC loss. None => the
        episode's per-step flag is taken as its `success` value (a coarse fallback
        for data that doesn't carry expert tags, e.g. a plain RL replay dump)."""
        assert states.shape[0] == actions.shape[0] + 1
        assert actions.shape[0] == rewards.shape[0]
        if self.image_keys:
            assert obs is not None, "image_keys set -> add_episode(..., obs=...) required"
            for k in self.image_keys:
                assert obs[k].shape[0] == states.shape[0], (
                    f"image/state length mismatch for {k}: "
                    f"{obs[k].shape[0]} vs {states.shape[0]}")
                assert isinstance(obs[k], JpegFrames) or obs[k].dtype == torch.uint8, \
                    f"{k} must be uint8 (or JpegFrames)"
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
        if self.image_keys:
            # [T+1, C, H, W] uint8 per camera, index-aligned with ep["state"]
            # (length asserted above).
            ep["obs"] = {k: obs[k] for k in self.image_keys}
        self.episodes.append(ep)
        self.num_episode += 1
        # Keep the transition counter and the transition-based eviction in sync
        # with _flush. Without this, max_transitions was a silent no-op for
        # OFFLINE-loaded data -- i.e. exactly the pixel arm, where a 24 GB replay
        # would blow past the cap with no error.
        self._n_trans += int(ep["action"].shape[0])
        if success:
            self.num_success += 1
        self._evict_for_transitions()
        if len(self.episodes) > self.max_episodes:
            # see _flush: num_success mirrors the buffer's current contents.
            ev = self.episodes.pop(0)
            self._n_trans -= int(ev["action"].shape[0])
            if ev["success"]:
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
               recency_alpha: float = 0.0, return_obs: bool = False,
               history: int = 0):
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
        obs_out = {k: [] for k in self.image_keys} if return_obs else {}
        # HISTORY rides INSIDE the obs dict under a reserved key, so every existing
        # `_s[-1]` unpacking site keeps getting the obs dict and nothing shifts.
        hist_out = [] if (history > 0 and return_obs) else None
        chosen = np.random.choice(len(idxs), size=batch_size, p=weights)
        for c in chosen:
            ep = self.episodes[idxs[c]]
            T = ep["action"].shape[0]
            start = np.random.randint(0, T - horizon + 1)
            if return_obs:
                # SAME [start : start+horizon+1] window as `state`, so obs[:, t]
                # and states[:, t] are the same timestep. Getting this wrong by one
                # would desynchronise images from states precisely at the terminal
                # step, which is where the only reward lives -- see the alignment
                # assertion in scripts/test_replay_images.py.
                for k in self.image_keys:
                    # JpegFrames slices to a LIST of encoded buffers; a raw tensor
                    # slices to a tensor. _stack_frames handles both.
                    obs_out[k].append(ep["obs"][k][start:start + horizon + 1])
            if hist_out is not None:
                from rl.history import build_batch
                hist_out.append(build_batch(
                    ep["state"][:, :PROP_DIM], ep["action"], start, horizon, history))
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
        if return_obs:
            # {camera: uint8[B, H+1, C, H, W]} -- appended LAST, after policy_mask
            # and value_targets, so every existing unpacking site is unchanged.
            # Kept UINT8 on the wire: float32 here would be ~340 MB/batch instead
            # of ~85 MB. The encoder casts (and normalises) on device.
            _o = {k: _stack_frames(v, device) for k, v in obs_out.items()}
            if hist_out is not None:
                _o[HISTORY_KEY] = torch.stack(hist_out).to(device)
            out.append(_o)
        if not (return_mask or return_value or return_obs):
            return (states, actions, rewards)
        return tuple(out)


def load_demos_to_sequence_buffer(
    buffer: SequenceReplayBuffer,
    data_path: str,
    num_data: int,
    reward_scale: float = 1.0,
    clamp_action: bool = True,
    demo_offset: int = 0,
):
    """Load TARGET demos (collect_target_demos.py format) into a sequence buffer.

    ★ `demo_offset` (default 0 = byte-for-byte legacy) takes the slice
    [offset, offset+num_data) instead of the FIRST num_data. Without it every
    adapt seed trains on the IDENTICAL demos, so a multi-seed error bar EXCLUDES
    §38's single-draw sigma = 0.092 -- the largest noise source in this project --
    and reads as far tighter than it is. The 200-episode target pool gives 10
    disjoint 20-demo draws; the blocked design shares one draw across the rungs
    WITHIN a seed (so ladder deltas are paired against ~0.027) and varies it
    ACROSS seeds (so the error bar on the mean delta honestly carries 0.092).

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
    assert demo_offset >= 0, demo_offset
    assert demo_offset == 0 or num_data > 0, (
        "demo_offset needs an explicit num_data (an open-ended slice from an "
        "offset would silently take a DIFFERENT count per seed)")
    assert demo_offset + max(num_data, 0) <= num_episode, (
        f"demo_offset={demo_offset} + num_data={num_data} exceeds the "
        f"{num_episode}-episode pool in {data_path}")
    print(f"[seq-buffer] loading demos [{demo_offset}, "
          f"{demo_offset + num_data if num_data > 0 else num_episode}) of "
          f"{num_episode} from {data_path}   (demo_offset={demo_offset})")
    loaded = 0
    for episode_id in range(demo_offset, num_episode):
        if num_data > 0 and episode_id >= demo_offset + num_data:
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

        # PIXEL: the frames collect_target_demos.py wrote (uint8 CHW, one row per
        # timestep, index-aligned with the state channels because the collector
        # appends them in the same iteration). Padded below EXACTLY like `state`.
        frames = {}
        for cam in getattr(buffer, "image_keys", ()):
            k = f"{cam}_image"
            assert k in episode["obs"], (
                f"demo_{episode_id} has no obs/{k}. collect_target_demos.py only "
                f"records cameras listed in the env's rl_cameras, and a STATE ckpt "
                f"carries rl_cameras=[] -- re-collect with --rl_cameras.")
            frames[cam] = read_frames_dataset(episode["obs"][k])

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
        # identical truncate-and-pad as `state`: s_0..s_{T-1} then s_T (or a
        # duplicate of s_{T-1}). Diverging here desynchronises frames from states
        # at the terminal step, which is where the only reward is.
        frames_seq = {}
        for cam, a in frames.items():
            if isinstance(a, JpegFrames):
                frames_seq[cam] = a.truncate_pad(T)
                continue
            f_last = a[T] if a.shape[0] > T else a[T - 1]
            frames_seq[cam] = torch.from_numpy(
                np.concatenate([a[:T], f_last[None]], axis=0))
        seg_actions = actions[: T]
        seg_rewards = rewards[: T] * reward_scale
        if clamp_action:
            seg_actions = np.clip(seg_actions, -1.0, 1.0)

        buffer.add_episode(
            torch.from_numpy(seq),
            torch.from_numpy(seg_actions),
            torch.from_numpy(seg_rewards),
            success=True,
            obs=frames_seq or None,
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
    n_skipped = 0
    _skip_reason = None
    for ep_id in ids:
        ep = g[ep_id]
        # A shard writer killed at its walltime can leave a final group whose
        # metadata exists but whose datasets are incomplete (no SWMR). Probe the
        # required members first and SKIP a broken tail rather than aborting a
        # load of tens of thousands of good episodes. Counted and reported below,
        # never silent.
        try:
            _ = ep["actions"].shape
            _ = ep["rewards"].shape
            _ = ep["obs"]["state"].shape
            for _cam in getattr(buffer, "image_keys", ()):
                _ = ep["obs"][f"{_cam}_image"].shape
        except Exception as _e:  # noqa: BLE001
            n_skipped += 1
            if _skip_reason is None:
                _skip_reason = f"{type(_e).__name__}: {_e}"
            continue
        states = np.array(ep["obs"]["state"]).astype(np.float32)  # [T, obs] = s_0..s_{T-1}
        # PIXEL: read the camera frames written by generate_simdist_data.py.
        # They are index-aligned with obs/state by construction (appended in the
        # same loop iteration), and MUST be padded at the terminal step exactly
        # like `state` below (s_T := s_{T-1}) or frames and states desynchronise
        # precisely where the only reward lives.
        frames = {}
        for cam in getattr(buffer, "image_keys", ()):
            frames[cam] = read_frames_dataset(ep["obs"][f"{cam}_image"])
        actions = np.array(ep["actions"]).astype(np.float32)      # [T, act]
        raw_rewards = np.array(ep["rewards"]).astype(np.float32)   # [T]
        T = actions.shape[0]
        assert states.shape[0] == T, f"{ep_id}: state len {states.shape[0]} != action len {T}"
        if T < 1:
            continue
        # s_T := s_{T-1} (terminal next-state not stored; absorbing approx)
        seq = np.concatenate([states, states[-1:]], axis=0)        # [T+1, obs]
        frames_seq = {c: (a.pad_last(1) if isinstance(a, JpegFrames)
                          else torch.from_numpy(np.concatenate([a, a[-1:]], axis=0)))
                      for c, a in frames.items()}   # same s_T := s_{T-1} padding
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
            obs=frames_seq or None,
        )
        loaded += 1
    f.close()
    if n_skipped:
        # "truncated writer output" is only ONE way to land here, and stating it
        # as the cause cost real time: a shard that is perfectly intact but was
        # collected with FEWER CAMERAS than image_keys asks for fails the same
        # probe, and every episode gets skipped with a message blaming the
        # writer. Report the first actual exception and, when everything was
        # skipped, say what the file really contains.
        print(f"[seq-buffer] !! skipped {n_skipped} episode(s) in {data_path}; "
              f"first reason: {_skip_reason}")
        if loaded == 0:
            try:
                _probe = g[ids[0]]
                _have = sorted(_probe["obs"].keys())
            except Exception:  # noqa: BLE001
                _have = ["<unreadable>"]
            print(f"[seq-buffer] !! NOTHING loaded from this shard. It provides "
                  f"obs keys {_have}; this run asks for image_keys "
                  f"{tuple(getattr(buffer, 'image_keys', ()))}. If a camera is "
                  f"missing, the shard needs scripts/rerender_simdist_data.py "
                  f"(a render-only pass over the stored sim_state), not a re-collect.")
    print(f"[seq-buffer] loaded {loaded} source trajectories; #episodes={buffer.size()}, "
          f"#transitions={buffer.num_transitions()}, #success={buffer.num_success}")
