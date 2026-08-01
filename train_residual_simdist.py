"""Residual off-policy RL (ResFiT-style) on a frozen ADAPTED SimDist base.

The adapted SimDist world model (`threading_simdist_adapt_seed1`, §24) plateaued
at TARGET strict-success ≈ 0.75 (MPPI) — dynamics adaptation can't push it
further. Here we keep improving it with a residual RL policy (rl/residual_simdist_
agent.py): the frozen adapted base proposes a_base(s) via MPPI, a trainable
residual actor adds a bounded delta, the executed action is clip(a_base+delta,-1,1),
and off-policy TD3 RL learns the correction (critic on the full action, actor
maximises Q) while the base stays frozen.

Faithful ResFiT + RLPD recipe (see rl/residual_rl_agent.py):
  ① every RL minibatch = (1-offline_fraction) online + offline_fraction demo;
  ② LayerNorm critic ensemble + REDQ random-subset min; ③ base+noise warmup +
  critic-only warmup; ⑤ deterministic actor Q-max + TD3 target smoothing + grad clip.

Two things the SimDist base forces (both handled here):
  * the base's MPPI is too slow to re-query during the RL update, so — exactly as
    ResFiT stores observation.base_action — the base action is computed ONCE per
    collected env step and STORED in the buffer's obs (`base_action`); the update
    reads it. rela n-step gives next_obs["base_action"] = base(s_{t+n}) for free.
  * n-step returns via rela (nstep=5), so the RLPD mix is Batch.merge_batches.

Train + eval in the TARGET domain (the adapt deploy domain). The buffer is
initialised from the adapted-MPPI-base rollout; the offline RLPD anchor = the 20
TARGET demos PLUS the successful base rollouts harvested during warmup.

Run:
  python train_residual_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_residual_simdist.yaml \
    --save_dir exps/mimicgen/threading_residual_simdist_seed1
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import yaml
import h5py
import pyrallis
import torch
import numpy as np
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite, STATE_KEYS
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl import replay
from rl.residual_rl_agent import ResidualRLConfig
from rl.residual_simdist_agent import ResidualSimDistAgent

import train_simdist
from rl.simdist_agent import SimDistAgent
from rl.simdist_planner import apply_original_recipe


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env
    task_name: str = "Threading_D0"
    episode_length: int = 400
    end_on_success: int = 1
    image_size: int = 224
    rl_image_size: int = 96
    rl_camera: str = "robot0_eye_in_hand"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1  # MUST match the adapted SimDist base's obs space
    use_state: int = 1
    # agent (residual RL knobs; base_policy_path is UNUSED — the SimDist base is
    # loaded from base_adapt_ckpt below)
    residual: ResidualRLConfig = field(default_factory=ResidualRLConfig)
    stddev_max: float = 0.1   # residual exploration noise (residual space)
    stddev_min: float = 0.1
    stddev_step: int = 500000
    # RL
    nstep: int = 5            # ResFiT n-step returns
    discount: float = 0.99
    replay_buffer_size: int = 500000
    batch_size: int = 256
    offline_fraction: float = 0.5   # RLPD: fraction of each RL batch from the offline anchor
    num_critic_update: int = 4      # UTD=4 (with update_freq=1)
    update_freq: int = 1            # RL updates per env step
    # --- stage-1 latent Dyna (§ residual + imagined latent transitions) ---
    # dyna_ratio > 0 adds N extra CRITIC-ONLY updates per env step on IMAGINED
    # latent transitions: from a REAL replay start state, imagine ONE latent step
    # (K=1) with the current combined policy, base action at z' from EXACT
    # latent-native MPPI (plan_from_latent), r_hat=0 consistency bootstrap
    # y=gamma*min_Q_target(z',a'). Densifies value propagation in latent space
    # (the diagnosed bottleneck: q_mean crawls); real reward stays on the real
    # batch. 0 = off (byte-identical to the plain residual run). Needs
    # residual.latent_features=1 (critic already in latent) + a planner base.
    dyna_ratio: float = 0.0         # #synthetic critic updates per env step / num_critic_update
    dyna_batch_size: int = 256      # synthetic batch
    # base action at the imagined z': exact=1 runs a full latent-native MPPI per
    # sample (~200ms/batch, exact) — 50-100x a critic step, so wall-clock heavy.
    # exact=0 (default) uses base_a'≈base_a (one-step base drift ~0.14·‖a‖ « action
    # magnitude at K=1, measured OK) — ~free. Start with 0 to answer "does synthetic
    # value bootstrap accelerate at all"; only go exact if the approx works but needs
    # the last bit of precision.
    dyna_exact_base: int = 0
    # synthetic transition reward: <0 => r_hat=0 (pure consistency bootstrap, the
    # original stage-1). >=0 => threshold pseudo-reward r_hat=1[V_src(z')>thresh]
    # with end_on_success-style termination (y=1 at anchor, else gamma*Q). Gives
    # the imagined transitions the sparse-reward ANCHOR a pure bootstrap lacks;
    # c=0.99 validated (diag_vsrc_threshold: near-success precision ~0.71, FPR
    # ~0.35% on the frozen094 base) so it won't reward-hack.
    dyna_reward_thresh: float = -1.0
    # K-step imagined rollout length (value expansion). K=1 = single imagined
    # step (near-zero info gain). K>1 needs dyna_exact_base=1: ONE MPPI solve on
    # z0 returns the whole base-action sequence for the rollout (plan_seq_from_latent),
    # so K-step costs ~1 MPPI, not K. K>1 with exact_base=0 drifts (base_a_0 reused).
    dyna_horizon: int = 1
    env_reward_scale: float = 1.0
    # base = the ADAPTED SimDist ckpt (its dir cfg.yaml -> stage1_ckpt -> stage-1
    # cfg.yaml rebuilds the world model; MPPI forced on + original recipe)
    base_adapt_ckpt: str = "exps/mimicgen/threading_simdist_adapt_seed1/model0.pt"
    # offline RLPD anchor: the TARGET demos (same set SimDist stage-2 BC used)
    offline_demo_path: str = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
    offline_num_data: int = 20
    # ALSO harvest successful MPPI-base rollouts during warmup into the offline
    # anchor (enriches it with more successful TARGET trajectories). Cap the count.
    offline_harvest_success: int = 100
    # OPTIONAL (the §"IQL-pretrained-critic" pipeline): a big precollected offline
    # anchor + a warm-started critic.
    #  * offline_transition_path: .npz of frozen-base TARGET rollouts (episodes with
    #    states/actions/rewards/success, as written by scripts/verify_iql_critic.py).
    #    Loaded into the offline RLPD anchor; the stored action IS the base action
    #    (frozen-base rollout, no residual), so base_action == action.
    #  * init_critic_from: .pt with {critic, vf, s_mean, s_std} from the IQL pretrain.
    #    Loads the critic (+ its z-score, to guarantee the critic sees the SAME
    #    normalization it was trained with) => the critic is calibrated exactly at
    #    the base actions the zero-init residual starts from => no early dip.
    offline_transition_path: str = ""
    # Set to 1 when the rollout npz did NOT come from a frozen base -- e.g. the
    # SimDist stage-1 adapt buffer, whose actions were planned by a still-adapting
    # dynamics model. Then `action != base_action` and base_action must be
    # RE-PLANNED with this run's frozen base. 0 = trust action == base_action.
    offline_replan_base_action: int = 0
    init_critic_from: str = ""
    # OPTIONAL: also warm-start the residual ACTOR from an IQL-AWR pretrain
    # (iql_actor_critic.pt's 'actor' key). Pairs with init_critic_from=<same file>.
    # With a warm actor you usually also want residual.progressive_clipping_steps=0
    # so the pretrained residual is active from step 0 (not clipped to ~0 early).
    init_actor_from: str = ""
    # warmup
    learning_starts: int = 5000          # env steps of base+noise before training
    warmup_noise_scale: float = 0.2      # uniform residual noise during warmup
    critic_warmup_steps: int = 10000     # critic-only RL updates before the actor
    # schedule
    num_train_step: int = 100000
    log_per_step: int = 1000             # eval every 1000 transitions (per the plan)
    num_eval_episode: int = 100
    mp_eval: int = 1
    num_eval_proc: int = 10
    # log
    save_dir: str = "exps/residual_simdist/run1"
    use_wb: int = 0
    # TARGET domain (train + eval share it): the §12 dynamics-only shift.
    domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    hand_init_perturb_joint_sigma: Optional[float] = None
    # OPTIONAL SGFT potential shaping (§10): a frozen source QAgent whose value()
    # densifies the residual critic's TD target. Non-empty ⇒ load it + attach +
    # set residual.sgft_shaping=1. Its obs space (use_state/*_stack) must match
    # this run's (load_frozen_agent asserts). Default "" = plain residual.
    sgft_source_agent: str = ""

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        assert self.use_state, "state-only"
        if self.sgft_source_agent in ("", "None", "none"):
            self.sgft_source_agent = ""
        else:
            self.residual.sgft_shaping = 1
        assert self.base_adapt_ckpt, "need base_adapt_ckpt (an adapted SimDist ckpt)"
        self.stddev_min = min(self.stddev_max, self.stddev_min)
        self.robots: list[str] = ["Panda"]

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:
    def __init__(self, cfg: MainConfig, from_main=True):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")
        if from_main:
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)
            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))
        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self._setup_env()

        self.agent = ResidualSimDistAgent(
            self.train_env.observation_shape,
            self.train_env.action_dim,
            cfg.residual,
            device="cuda",
        )
        self._attach_base_policy()
        self._maybe_attach_value_source()
        self.agent.cfg_discount = float(cfg.discount)  # stage-1 dyna single-step bootstrap

        if not from_main:
            return
        self._setup_replay()
        self._load_offline_demos()          # loads demos + fits z-score on the demo states
        self._load_offline_transitions()    # optional: big frozen-base rollout anchor (re-fits z-score on the pool)
        self._maybe_init_critic()            # optional: warm-start critic from the IQL pretrain (overrides z-score)
        self._maybe_init_actor()             # optional: warm-start residual actor from IQL-AWR pretrain

    # ------------------------------------------------------------------- setup
    def _setup_env(self):
        self.rl_cameras: list[str] = [] if self.cfg.use_state else list(self.cfg.rl_cameras)
        common_env = dict(
            env_name=self.cfg.task_name,
            robots=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.rl_cameras,
            rl_cameras=self.rl_cameras,
            use_state=bool(self.cfg.use_state),
            obs_stack=self.cfg.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=self.cfg.prop_stack,
            strict_success=self.cfg.strict_success,
        )
        self.train_env = PixelRobosuite(
            **common_env,
            env_reward_scale=self.cfg.env_reward_scale,
            end_on_success=bool(self.cfg.end_on_success),
            domain_shift=self.cfg.domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.hand_init_perturb_joint_sigma,
        )
        self.eval_env_params = dict(
            **common_env,
            domain_shift=self.cfg.domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.hand_init_perturb_joint_sigma,
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _attach_base_policy(self):
        """Load the ADAPTED SimDist world model and attach it (frozen, MPPI)."""
        adapt_ckpt = self.cfg.base_adapt_ckpt
        adapt_dir = os.path.dirname(adapt_ckpt)
        adapt_cfg = yaml.safe_load(open(os.path.join(adapt_dir, "cfg.yaml"), "r"))
        stage1_ckpt = adapt_cfg["stage1_ckpt"]
        stage1_cfg_path = os.path.join(os.path.dirname(stage1_ckpt), "cfg.yaml")
        base_cfg: train_simdist.MainConfig = pyrallis.load(
            train_simdist.MainConfig, open(stage1_cfg_path, "r"))  # type: ignore
        # FORCE MPPI + the ORIGINAL SimDist recipe (mirrors train_simdist_adapt).
        base_cfg.simdist.use_planner = 1
        p = apply_original_recipe(base_cfg.simdist.planner)
        print(f"[base] adapted SimDist MPPI: samples={p.num_samples} "
              f"(base_override={p.num_base_override}), init_std={p.init_std}, "
              f"iters={p.iterations}, horizon={p.horizon}")
        base = SimDistAgent(
            True, self.train_env.observation_shape, self.train_env.prop_shape,
            self.train_env.action_dim, base_cfg.rl_camera, base_cfg.simdist, device="cuda",
        )
        print(f"loading adapted SimDist base ckpt: {adapt_ckpt}")
        base.load_state_dict(torch.load(adapt_ckpt))
        base.eval()
        self.agent.attach_base_policy(base)

    def _maybe_attach_value_source(self):
        """Optional SGFT shaping: load a frozen source QAgent and attach it so the
        residual critic's TD target gets the potential term r+coef*(g^n V(s')-V(s))."""
        if not self.cfg.sgft_source_agent:
            return
        import train_rl
        obs_cfg = train_rl.MainConfig()
        obs_cfg.use_state = int(self.cfg.use_state)
        obs_cfg.state_stack = self.cfg.state_stack
        obs_cfg.obs_stack = self.cfg.obs_stack
        obs_cfg.prop_stack = self.cfg.prop_stack
        src = train_rl.load_frozen_agent(
            self.cfg.sgft_source_agent, self.train_env, obs_cfg, "cuda")
        self.agent.attach_value_source(src)
        print(f"[sgft] attached frozen value source: {self.cfg.sgft_source_agent} "
              f"(coef={self.cfg.residual.sgft_coef})")

    def _setup_replay(self):
        def make():
            return replay.ReplayBuffer(
                self.cfg.nstep, self.cfg.discount, frame_stack=1,
                max_episode_length=self.cfg.episode_length,
                replay_size=self.cfg.replay_buffer_size, use_bc=False,
            )
        self.online = make()
        self.offline = make()

    # --------------------------------------------------------- base-action util
    @torch.no_grad()
    def _base_actions(self, states: torch.Tensor) -> torch.Tensor:
        """MPPI base action for a [B, obs] state batch -> [B, act] (on cuda)."""
        return self.agent._base_action(states.to("cuda").float())

    def _push_episode(self, buffer, states: np.ndarray, base_a: np.ndarray,
                      actions: np.ndarray, rewards: np.ndarray, success: bool):
        """Push one episode into a rela buffer with base_action stored in obs.
        states/base_a: [T, .] (s_0..s_{T-1}); actions/rewards: [T, .]/[T].
        rela drops the terminal next-obs, so only T states are needed."""
        T = actions.shape[0]

        def obs_at(t):
            return {"state": torch.as_tensor(states[t]).float(),
                    "base_action": torch.as_tensor(base_a[t]).float()}

        buffer.new_episode(obs_at(0))
        for t in range(T):
            terminal = (t == T - 1)
            succ = bool(success) if terminal else False
            nxt = obs_at(min(t + 1, T - 1))  # ignored by rela on the terminal step
            reply = {"action": torch.as_tensor(actions[t]).float()}
            buffer.add(nxt, reply, float(rewards[t]), terminal, succ, image_obs={})

    def _load_offline_demos(self):
        """Load the TARGET demos into the offline buffer with MPPI base actions,
        and fit the residual/critic state z-score on the demo states."""
        path = self.cfg.offline_demo_path
        cfg_path = os.path.join(os.path.dirname(path), "env_cfg.json")
        task_name = json.load(open(cfg_path, "r"))["env_name"]
        state_keys = STATE_KEYS[task_name]
        f = h5py.File(path, "r")
        num_episode = len(list(f["data"].keys()))
        num_data = self.cfg.offline_num_data
        print(f"[offline] loading {num_data} of {num_episode} TARGET demos from {path}")
        loaded = 0
        all_states = []
        for ep_id in range(num_episode):
            if num_data > 0 and ep_id >= num_data:
                break
            ep = f[f"data/demo_{ep_id}"]
            actions = np.clip(np.array(ep["actions"]).astype(np.float32), -1.0, 1.0)
            rewards = np.array(ep["rewards"]).astype(np.float32)
            if not (rewards == 1).any():
                print(f"  skipping demo_{ep_id}: no success step")
                continue
            states = np.concatenate(
                [np.array(ep["obs"][k]) for k in state_keys], axis=1).astype(np.float32)
            T = int(np.argmax(rewards == 1)) + 1     # truncate at first success
            seg_states = states[:T]                  # s_0..s_{T-1}
            seg_actions = actions[:T]
            seg_rewards = rewards[:T] * self.cfg.env_reward_scale
            base_a = self._base_actions(
                torch.from_numpy(seg_states)).cpu().numpy()  # [T, act]
            self._push_episode(self.offline, seg_states, base_a,
                               seg_actions, seg_rewards, success=True)
            all_states.append(seg_states)
            loaded += 1
        f.close()
        print(f"[offline] loaded {loaded} demos; offline #transitions={self.offline.size()}")

        stacked = np.concatenate(all_states, axis=0)  # [N, obs]
        self._offline_state_pool = [stacked]  # extended by _load_offline_transitions
        self.agent.set_normalization(
            torch.from_numpy(stacked.mean(0)), torch.from_numpy(stacked.std(0)))
        print(f"[offline] z-score fit on {stacked.shape[0]} demo states")

    def _load_offline_transitions(self):
        """Optional: load a big precollected frozen-base TARGET rollout anchor
        (scripts/verify_iql_critic.py's rollouts.npz) into the offline buffer and
        RE-FIT the z-score on the combined (demo + rollout) pool. base_action ==
        action (frozen-base rollout, no residual/noise)."""
        path = self.cfg.offline_transition_path
        if not path:
            return
        d = np.load(path, allow_pickle=True)
        episodes = list(d["episodes"])
        replan = bool(self.cfg.offline_replan_base_action)
        if replan:
            print("[offline] re-planning base_action with the frozen base "
                  "(rollouts are NOT from a frozen base)")
        loaded = 0
        chunk = 256
        for e in episodes:
            states = e["states"].astype(np.float32)       # [T+1, obs]
            actions = e["actions"].astype(np.float32)     # [T, act]
            rewards = (e["rewards"].astype(np.float32) * self.cfg.env_reward_scale)
            T = actions.shape[0]
            if T < 1:
                continue
            if replan:
                base_a = np.concatenate([
                    self._base_actions(torch.from_numpy(states[i:i + chunk]))
                    .cpu().numpy() for i in range(0, T, chunk)], axis=0)[:T]
            else:
                base_a = actions          # frozen-base rollout => identical
            self._push_episode(self.offline, states[:T], base_a, actions,
                               rewards, success=bool(e["success"]))
            self._offline_state_pool.append(states[:T])
            loaded += 1
        n_succ = sum(int(e["success"]) for e in episodes)
        print(f"[offline] loaded {loaded} rollout episodes from {path} "
              f"({n_succ} success); offline #transitions={self.offline.size()}")
        stacked = np.concatenate(self._offline_state_pool, axis=0)
        self.agent.set_normalization(
            torch.from_numpy(stacked.mean(0)), torch.from_numpy(stacked.std(0)))
        print(f"[offline] z-score RE-FIT on {stacked.shape[0]} (demo+rollout) states")

    def _maybe_init_critic(self):
        """Optional: warm-start the residual critic from the IQL pretrain. Also
        overrides the z-score with the EXACT stats the critic was trained under,
        so the pretrained critic sees the same normalized inputs (else it is
        silently miscalibrated)."""
        path = self.cfg.init_critic_from
        if not path:
            return
        ckpt = torch.load(path, map_location="cuda")
        self._assert_feature_match(ckpt, path)
        if "s_mean" in ckpt and self.cfg.residual.normalize_states:
            self.agent.set_normalization(ckpt["s_mean"], ckpt["s_std"])
            print("[init_critic] overrode z-score with the IQL-pretrain stats")
        missing, unexpected = self.agent.critic.load_state_dict(ckpt["critic"], strict=True)
        self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())
        print(f"[init_critic] loaded IQL critic from {path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)}) + synced target")

    def _assert_feature_match(self, ckpt, path):
        """An IQL pretrain is tied to its feature space (raw z-score vs the frozen
        SimDist latent). Loading the wrong one would otherwise die on a cryptic
        shape mismatch — or, if the widths happened to agree, silently miscalibrate."""
        want = "latent" if self.cfg.residual.latent_features else "raw"
        got = ckpt.get("features")
        if got is None:
            # pre-dates the flag => raw by construction
            got = "raw"
        assert got == want, (
            f"{path} was pretrained with features='{got}' but this run has "
            f"residual.latent_features={self.cfg.residual.latent_features} "
            f"(=> '{want}'). Re-run scripts/verify_iql_critic.py --features {want}.")
        if "feat_dim" in ckpt:
            assert int(ckpt["feat_dim"]) == self.agent.feat_dim, (
                f"{path} feat_dim={ckpt['feat_dim']} != agent feat_dim="
                f"{self.agent.feat_dim}")

    def _maybe_init_actor(self):
        """Optional: warm-start the residual ACTOR from an IQL-AWR pretrain
        (iql_actor_critic.pt's 'actor' key), so the residual starts pointing at
        high-advantage demo-style corrections instead of zero."""
        path = self.cfg.init_actor_from
        if not path:
            return
        ckpt = torch.load(path, map_location="cuda")
        assert "actor" in ckpt, f"{path} has no 'actor' key (needs --train_actor pretrain)"
        self._assert_feature_match(ckpt, path)
        self.agent.actor.load_state_dict(ckpt["actor"], strict=True)
        self.agent.actor_target.load_state_dict(self.agent.actor.state_dict())
        print(f"[init_actor] loaded IQL-AWR residual actor from {path} + synced target")

    def _sample_mixed(self):
        """RLPD symmetric mix: online (1-frac) + offline demo (frac)."""
        n_off = int(round(self.cfg.batch_size * self.cfg.offline_fraction))
        n_on = self.cfg.batch_size - n_off
        on = self.online.sample(n_on, "cuda")
        if n_off <= 0:
            return on
        off = self.offline.sample(n_off, "cuda")
        return replay.Batch.merge_batches(on, off)

    # -------------------------------------------------------------------- eval
    def eval(self, seed) -> float:
        random_state = np.random.get_state()
        if self.cfg.mp_eval:
            scores = run_eval_mp(env_params=self.eval_env_params, agent=self.agent,
                                 num_proc=self.cfg.num_eval_proc,
                                 num_game=self.cfg.num_eval_episode, seed=seed, verbose=False)
        else:
            scores = run_eval(env_params=self.eval_env_params, agent=self.agent,
                              num_game=self.cfg.num_eval_episode, seed=seed,
                              record_dir=None, verbose=False)
        np.random.set_state(random_state)
        return float(np.mean(scores))

    # ----------------------------------------------------------------- warm-up
    def warm_up(self):
        """Roll the adapted MPPI base + uniform residual noise in the TARGET env
        for `learning_starts` env steps (fills `online`), harvesting successful
        episodes into the offline RLPD anchor (up to offline_harvest_success)."""
        scale = self.cfg.warmup_noise_scale
        ascale = self.cfg.residual.action_scale
        obs, _ = self.train_env.reset()
        cur_base = self._base_actions(obs["state"].view(1, -1))[0]  # [act] on cuda
        self.online.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
        # accumulate the current episode for offline harvest
        ep_s, ep_ba, ep_a, ep_r = [obs["state"].cpu().numpy()], [cur_base.cpu().numpy()], [], []
        harvested = 0
        pbar = tqdm(total=self.cfg.learning_starts, desc="warm-up (MPPI base+noise)",
                    dynamic_ncols=True)
        n_steps = 0
        n_succ = 0
        # Collect >= learning_starts steps, but only STOP at a terminal so the rela
        # online episode ends flushed/clean (a mid-episode break would leave a
        # dangling partial episode and crash the next new_episode()).
        while True:
            noise = (torch.rand(self.agent.action_dim, device=cur_base.device) * 2 - 1) * scale
            res = noise.clamp(-ascale, ascale)
            action = (cur_base + res).clamp(-1.0, 1.0).cpu()
            next_obs, reward, terminal, success, image_obs = self.train_env.step(action)
            next_base = self._base_actions(next_obs["state"].view(1, -1))[0]
            self.online.add({"state": next_obs["state"], "base_action": next_base.cpu()},
                            {"action": action}, float(reward), terminal, bool(success),
                            image_obs={})
            ep_a.append(action.numpy()); ep_r.append(float(reward))
            ep_s.append(next_obs["state"].cpu().numpy()); ep_ba.append(next_base.cpu().numpy())
            obs = next_obs; cur_base = next_base
            n_steps += 1; pbar.update(1)
            if terminal:
                n_succ += int(success)
                if success and harvested < self.cfg.offline_harvest_success:
                    # push s_0..s_{T-1} (drop the terminal next state, like rela)
                    self._push_episode(
                        self.offline, np.stack(ep_s[:-1]), np.stack(ep_ba[:-1]),
                        np.stack(ep_a), np.array(ep_r, dtype=np.float32), success=True)
                    harvested += 1
                if n_steps >= self.cfg.learning_starts:
                    break  # stop on a clean episode boundary
                obs, _ = self.train_env.reset()
                cur_base = self._base_actions(obs["state"].view(1, -1))[0]
                self.online.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
                ep_s, ep_ba, ep_a, ep_r = [obs["state"].cpu().numpy()], [cur_base.cpu().numpy()], [], []
        pbar.close()
        print(f"warm-up done. online #episode={self.online.size()} ({n_steps} steps, "
              f"successes: {n_succ}, harvested into offline: {harvested}); "
              f"offline #episode={self.offline.size()}")

    # ------------------------------------------------------------------- train
    def train(self):
        stat = common_utils.MultiCounter(
            self.work_dir, bool(self.cfg.use_wb), wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run, wb_group_name=self.cfg.wb_group, config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)
        self.warm_up()

        # critic-only warmup (the actor starts after critic_warmup_steps RL updates)
        if self.cfg.critic_warmup_steps > 0:
            print(f"critic warmup: {self.cfg.critic_warmup_steps} critic-only updates")
            for _ in tqdm(range(self.cfg.critic_warmup_steps), desc="critic warmup",
                          dynamic_ncols=True, mininterval=1.0):
                self.agent.update(self._sample_mixed(), update_actor=False)

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        cur_base = self._base_actions(obs["state"].view(1, -1))[0]
        self.online.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
        pbar = tqdm(total=self.cfg.num_train_step, desc="residual-simdist train",
                    dynamic_ncols=True, mininterval=1.0)
        recent: list[float] = []
        while self.global_step < self.cfg.num_train_step:
            self._set_clip_scale()  # progressive residual bound (holds through the eval)
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                # reuse cur_base (already planned) -> no extra MPPI for the action
                res = self.agent.actor(
                    self.agent._norm(obs["state"].view(1, -1).cuda().float()),
                    cur_base.view(1, -1))[0]
                res = (res + torch.randn_like(res) * stddev).clamp(-self.cfg.residual.action_scale,
                                                                   self.cfg.residual.action_scale)
                action = (cur_base + res).clamp(-1.0, 1.0).cpu()
                stat["data/stddev"].append(stddev)
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action)
            with stopwatch.time("base plan"):
                next_base = self._base_actions(obs["state"].view(1, -1))[0]
            self.online.add({"state": obs["state"], "base_action": next_base.cpu()},
                            {"action": action}, float(reward), terminal, bool(success),
                            image_obs={})
            cur_base = next_base
            self.global_step += 1
            pbar.update(1)
            if terminal:
                self.global_episode += 1
                stat["score/train_score"].append(float(success))
                recent.append(float(success))
                if len(recent) > 20:
                    recent.pop(0)
                pbar.set_postfix(ep=self.global_episode, succ=f"{np.mean(recent):.2f}")
                obs, _ = self.train_env.reset()
                cur_base = self._base_actions(obs["state"].view(1, -1))[0]
                self.online.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)
        pbar.close()

    def _set_clip_scale(self):
        """Progressive clipping: ramp the residual bound 0 -> action_scale over
        `progressive_clipping_steps` env steps (keeps the residual ~0 early so eval
        starts at the base score). Only called during training; standalone eval
        keeps the full action_scale default set in the agent ctor."""
        n = self.cfg.residual.progressive_clipping_steps
        frac = min(1.0, self.global_step / n) if n > 0 else 1.0
        self.agent.clip_scale = float(self.cfg.residual.action_scale) * frac

    def rl_train(self, stat):
        # ResFiT: `num_critic_update` (UTD) critic updates per env step, the actor
        # updated once (on the last iteration, actor_updates_per_iteration=1).
        for i in range(self.cfg.num_critic_update):
            batch = self._sample_mixed()
            update_actor = (i == self.cfg.num_critic_update - 1)
            metrics = self.agent.update(batch, update_actor=update_actor)
            stat.append(metrics)
        # stage-1 latent Dyna: extra critic-only updates on imagined transitions.
        # start states = real online replay states (in-distribution branch points).
        n_dyna = int(round(self.cfg.dyna_ratio * self.cfg.num_critic_update))
        for _ in range(n_dyna):
            b = self.online.sample(self.cfg.dyna_batch_size, "cuda")
            m = self.agent.update_critic_synthetic(
                b.obs["state"], b.obs["base_action"],
                exact_base=bool(self.cfg.dyna_exact_base),
                reward_thresh=(self.cfg.dyna_reward_thresh
                               if self.cfg.dyna_reward_thresh >= 0 else None),
                horizon=self.cfg.dyna_horizon)
            stat.append(m)

    def log_and_save(self, stopwatch, stat, saver):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/online_transitions"].append(self.online.size())
        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            eval_score = self.eval(seed=eval_seed)  # TARGET (the deployment metric)
            stat["score/score"].append(eval_score)
        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        stat.summary(self.global_step, reset=True)
        print(f"[eval @ step {self.global_step}] target_score={eval_score:.3f}  saved?={saved}")
        stopwatch.summary(reset=True)
        print(common_utils.get_mem_usage())


def load_model(weight_file, device):
    """Rebuild a ResidualSimDistAgent (+ re-attach the adapted SimDist base) for
    standalone eval (evaluate/eval.py --mode residual_simdist)."""
    cfg_path = os.path.join(os.path.dirname(weight_file), "cfg.yaml")
    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    workspace = Workspace(cfg, from_main=False)
    agent = workspace.agent
    agent.load_state_dict(torch.load(weight_file))
    agent = agent.to(device)
    return agent, workspace.eval_env, workspace.eval_env_params


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    workspace = Workspace(cfg)
    workspace.train()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    from rich.traceback import install
    install()
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
