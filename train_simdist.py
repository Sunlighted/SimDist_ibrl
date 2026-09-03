"""SimDist baseline (state-based), two stages back-to-back (see CLAUDE.md §17).

SimDist (Levy et al., RSS 2026) distills a simulator into a latent world model
and plans with MPPI; the original adapts online with real rollouts. Here we make
it a state-based ibrl baseline on the Threading_D0 source->target dynamics gap:

  Stage 1 (SOURCE sim, online): roll out in the nominal env (mode=none) and
    DISTILL the world model (encoder/dynamics/reward/value/policy). Data is
    collected MIXED: the first `expert_collect_steps` steps use the frozen
    SOURCE expert (threading_ibrl_source_seed1) + exploration noise (coverage +
    learnable under sparse reward), then the world model's own MPPI planner
    (on-policy, TD-MPC2 style).
  Stage 2 (offline finetune): using the fixed TARGET demos, finetune ONLY the
    dynamics + policy heads (encoder/reward/value frozen) — consistency on
    (s,a,s') + BC on (s,a). Replaces the original online adaptation.

Execution/eval = MPPI on the world model. TARGET strict-success is the
deployment metric; SOURCE eval is a diagnostic.

Run:
  python train_simdist.py --config_path release/cfgs/mimicgen/threading_d0_state_simdist.yaml \
      --source_agent exps/mimicgen/threading_ibrl_source_seed1/model0.pt \
      --save_dir exps/mimicgen/threading_simdist_seed1
"""

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import yaml
import pyrallis
import torch
import numpy as np
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig, InitDistConfig
from env.strict_success import StrictSuccessConfig
from rl.simdist_agent import SimDistAgent, SimDistConfig
from rl.sequence_replay import (
    SequenceReplayBuffer,
    load_demos_to_sequence_buffer,
    load_source_replay_to_sequence_buffer,
)
from train_rl import load_frozen_agent, attach_ibrl_prior


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
    state_stack: int = 1  # MUST be 1 (matches the source ckpt obs space)
    use_state: int = 1
    # agent
    simdist: SimDistConfig = field(default_factory=SimDistConfig)
    stddev_max: float = 0.0  # CLEAN: executed action = BC target; noise degrades the clone on this precision task
    stddev_min: float = 0.0
    stddev_step: int = 200000
    discount: float = 0.99
    env_reward_scale: float = 1
    batch_size: int = 256
    update_freq: int = 2
    seq_replay_max_episode: int = 5000
    # stage-1 data collection: frozen SOURCE expert guides the first
    # `expert_collect_steps` env steps (warm-up included), then MPPI self-explores.
    source_agent: str = ""
    expert_collect_steps: int = 0  # <=0 = all-expert (stable); >0 = switch to MPPI after N
    # The value head is distilled from the frozen expert's V. That expert was trained
    # with act_method="ibrl", i.e. it EXECUTES the max-Q pick over {RL actor, prior} --
    # so `V(s)=min_k Q_k(s, pi_rl(s))` (what load_frozen_agent gives you) describes a
    # policy it never runs, and it under-rates exactly those states where the prior is
    # the better action. Since the prior lifts V only where it wins, the bare-actor V
    # and the hybrid V are NOT a monotone rescaling of each other: they induce
    # different RANKINGS, which is all MPPI consumes. 1 = distill the hybrid (max-Q) V
    # (correct: it matches the policy the data + the policy head actually follow).
    # 0 = legacy bare-RL-actor V. Falls back to 0 if the expert declares no prior.
    source_value_ibrl: int = 1
    # OFFLINE source (方案 B): if non-empty, distill from this pre-collected MIXED
    # source replay (rl/replay.py dump: expert+suboptimal+random) instead of online
    # rolling out the frozen expert. Broader state-action coverage -> a more
    # generalizable dynamics head (the original SimDist's expert+non-expert data-gen
    # idea). `distill_steps` then counts TRAINING updates (no env stepping).
    # source_agent is still recommended (it provides the dense V_src value labels).
    source_replay_path: str = ""
    source_replay_num_data: int = -1   # -1/0 = all trajectories
    # OPTIONAL extra clean-expert source demos (collect_target_demos.py sidecar
    # format: obs/state precomputed, first-success truncated, 100% success, NO
    # expert_flags) mixed INTO the same seq_replay alongside source_replay_path.
    # They carry no expert_flags so the policy mask treats every frame as expert
    # (success-broadcast) -> strengthens the policy-BC signal on the success
    # manifold (which MPPI plans over), while also giving the dynamics head clean
    # success coverage. Loaded LAST so the ring buffer never evicts them. Bump
    # seq_replay_max_episode so they don't displace all the diverse data.
    expert_demo_path: str = ""
    expert_demo_num_data: int = -1   # -1/0 = all trajectories
    # Apply the paper's per-step 1_e(a_t) indicator (Eq. 2): the policy-BC term
    # only trains on UNCORRUPTED-EXPERT actions (expert episode AND no noise this
    # step), read from the dataset's `expert_flags` (generate_simdist_data.py).
    # The dynamics/value/reward/decoder heads still distill from ALL data (broad
    # coverage). On precision tasks MSE-BC over a mixed buffer averages to a mushy
    # near-mean policy (MPPI's proposal); restricting BC to clean expert actions
    # sharpens it without losing dynamics coverage; enc_pi makes it free. For data
    # lacking expert_flags (plain RL replay) the mask falls back to success. OFF.
    bc_expert_only: int = 1
    # OPTIONAL separate PURE-EXPERT buffer that ONLY the policy-BC head trains on
    # (ablation): dynamics/value/reward/decoder keep learning from the
    # exploration data in `source_replay_path`, while the policy clones ONLY
    # these clean expert trajectories. Decoupled cleanly because the policy has
    # its own encoder (enc_pi) and doesn't touch the dynamics latent. Empty =>
    # policy trains on the main (exploration) buffer like before.
    policy_replay_path: str = ""
    policy_replay_num_data: int = -1
    # stage-2 target demos
    target_demo_path: str = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
    target_num_data: int = 20
    # schedule
    num_warm_up_episode: int = 20
    distill_steps: int = 200000   # stage-1 online env steps
    log_per_step: int = 10000
    num_eval_episode: int = 50
    eval_source: int = 1
    mp_eval: int = 1
    # log
    save_dir: str = "exps/simdist/run1"
    # Resume a killed run from <save_dir>/resume.pt ("auto") or an explicit path.
    # "" = start from scratch. NOT the same as loading latest.pt: see save_resume().
    resume: str = ""
    # Only for a WEIGHTS-ONLY ckpt (latest.pt from a run that predates resume.pt):
    # the step to continue counting from. Ignored for a real resume.pt.
    resume_step: int = 0
    use_wb: int = 0
    state_only_no_render: int = 0
    # SOURCE (train) and TARGET (eval) domain shifts
    train_domain_shift: DomainShiftConfig = field(default_factory=lambda: DomainShiftConfig(mode="none"))
    eval_domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    # Reset-distribution override (Square nut z_rot window / Threading needle box).
    # Default InitDistConfig() is disabled, so every existing run is unchanged.
    # Stage-1 distillation itself is offline on source_replay_path, but the env
    # drives the in-training eval -- leaving this out would score a rotw1-trained
    # model on the full 360-degree distribution.
    train_init_dist: InitDistConfig = field(default_factory=InitDistConfig)
    eval_init_dist: InitDistConfig = field(default_factory=InitDistConfig)
    train_hand_init_perturb_joint_sigma: Optional[float] = None
    eval_hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        if self.source_agent in ["none", "None"]:
            self.source_agent = ""
        assert self.use_state, "SimDist baseline is state-only"
        self.stddev_min = min(self.stddev_max, self.stddev_min)
        self.robots: list[str] = ["Panda"]
        # keep planner/world-model horizons in sync unless explicitly diverged
        if self.simdist.planner.horizon != self.simdist.world_model.horizon:
            print(f"[warn] planner.horizon={self.simdist.planner.horizon} != "
                  f"world_model.horizon={self.simdist.world_model.horizon}")
        self.simdist.planner.discount = self.discount

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
        self.finetune_step = 0
        self.H = cfg.simdist.world_model.horizon
        self._setup_env()

        self.agent = SimDistAgent(
            bool(self.cfg.use_state),
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            self.cfg.rl_camera,
            cfg.simdist,
            device="cuda",
        )

        if not from_main:
            return

        # frozen SOURCE expert for stage-1 guided collection (optional)
        self.source_agent = None
        self.source_value_ibrl = False
        if cfg.source_agent:
            print(f"loading frozen SOURCE expert for collection: {cfg.source_agent}")
            self.source_agent = load_frozen_agent(cfg.source_agent, self.train_env, cfg, "cuda")
            if cfg.source_value_ibrl:
                # Only the VALUE path becomes hybrid; act_method stays "rl", so the
                # expert's own act() (stage-1 collection) is byte-for-byte unchanged.
                desc = attach_ibrl_prior(self.source_agent, cfg.source_agent,
                                         self.train_env, "cuda")
                self.source_value_ibrl = bool(desc)
                print(f"[value distill] V targets = {'IBRL HYBRID max-Q' if desc else 'bare RL actor'}"
                      + (f" | prior = {desc}" if desc else
                         " (expert declares no prior -> falling back)"))

        self._setup_replay()

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
            # See train_rl.py's field of the same name: skips robosuite's
            # offscreen renderer for state-based runs. Required on MIG-partitioned
            # GPUs, where CUDA_VISIBLE_DEVICES is a "MIG-<uuid>" string that
            # robosuite's EGL device selection cannot parse. Default 0 keeps the
            # old rendering path byte-for-byte.
            state_only_no_render=bool(self.cfg.state_only_no_render),
        )
        # SOURCE (online training)
        self.train_env = PixelRobosuite(
            **common_env,
            env_reward_scale=self.cfg.env_reward_scale,
            end_on_success=bool(self.cfg.end_on_success),
            domain_shift=self.cfg.train_domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.train_hand_init_perturb_joint_sigma,
            init_distribution=self.cfg.train_init_dist,
        )
        # TARGET (eval)
        self.eval_env_params = dict(
            **common_env,
            domain_shift=self.cfg.eval_domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.eval_hand_init_perturb_joint_sigma,
            init_distribution=self.cfg.eval_init_dist,
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _setup_replay(self):
        # SOURCE trajectory buffer (world model distillation)
        self.offline_source = bool(self.cfg.source_replay_path)
        self.seq_replay = SequenceReplayBuffer(
            max_episodes=self.cfg.seq_replay_max_episode, device="cpu"
        )
        if self.offline_source:
            # 方案 B: fill from a pre-collected MIXED source replay; no online rollout.
            load_source_replay_to_sequence_buffer(
                self.seq_replay, self.cfg.source_replay_path,
                num_data=self.cfg.source_replay_num_data,
                reward_scale=self.cfg.env_reward_scale,
            )
            assert self.seq_replay.can_sample(self.H), (
                f"no source-replay episode of length >= horizon={self.H}")
            if self.source_agent is None:
                print("[warn] offline source replay without --source_agent: value "
                      "labels fall back to TD-bootstrap, which collapses to V==0 "
                      "under sparse reward (see CLAUDE.md §17). Provide --source_agent.")
        # OPTIONAL clean-expert demos mixed into the SAME buffer (loaded LAST so the
        # ring buffer evicts diverse data before these). Same loader: it reads
        # obs/state directly and, lacking expert_flags, marks every frame expert.
        if self.cfg.expert_demo_path:
            before = self.seq_replay.size()
            load_source_replay_to_sequence_buffer(
                self.seq_replay, self.cfg.expert_demo_path,
                num_data=self.cfg.expert_demo_num_data,
                reward_scale=self.cfg.env_reward_scale,
            )
            print(f"[seq-buffer] + expert demos: buffer {before} -> "
                  f"{self.seq_replay.size()} episodes "
                  f"({self.seq_replay.num_success} success)")
            assert self.seq_replay.can_sample(self.H), (
                f"no expert-demo episode of length >= horizon={self.H}")
        # OPTIONAL pure-expert buffer for the policy-BC head only (ablation).
        self.policy_replay = None
        if self.cfg.policy_replay_path:
            self.policy_replay = SequenceReplayBuffer(
                max_episodes=self.cfg.seq_replay_max_episode, device="cpu")
            load_source_replay_to_sequence_buffer(
                self.policy_replay, self.cfg.policy_replay_path,
                num_data=self.cfg.policy_replay_num_data,
                reward_scale=self.cfg.env_reward_scale,
            )
            print(f"[policy-buffer] loaded {self.policy_replay.size()} pure-expert "
                  f"trajectories for policy-BC ({self.policy_replay.num_success} success)")
            assert self.policy_replay.can_sample(self.H), (
                f"no policy-expert episode of length >= horizon={self.H}")

        # TARGET demos (stage-2 finetune ONLY -- stage 1 never touches this buffer).
        # `target_demo_path` defaults to a THREADING path, so on any other task an
        # unset cfg silently loads Threading demos: 37-dim states where TPA is 51,
        # with nothing checking the width. Empty path = skip; finetune() errors if
        # it is actually needed. (CLAUDE.md rule 13/17: non-empty code defaults
        # that quietly swap the data out from under a run.)
        self.demo_replay = SequenceReplayBuffer(max_episodes=10000, device="cpu")
        if self.cfg.target_demo_path:
            load_demos_to_sequence_buffer(
                self.demo_replay, self.cfg.target_demo_path,
                num_data=self.cfg.target_num_data, reward_scale=self.cfg.env_reward_scale,
            )
            assert self.demo_replay.can_sample(self.H), (
                f"no TARGET demo episode of length >= horizon={self.H}")
            if self.seq_replay.episodes:
                src_dim = self.seq_replay.episodes[0]["state"].shape[-1]
                demo_dim = self.demo_replay.episodes[0]["state"].shape[-1]
                assert demo_dim == src_dim, (
                    f"TARGET demo state dim {demo_dim} != source replay {src_dim}.\n"
                    f"  target_demo_path: {self.cfg.target_demo_path}\n"
                    f"  source_replay_path: {self.cfg.source_replay_path}\n"
                    "These are different tasks -- set target_demo_path for THIS task "
                    "(or '' if stage-2 runs elsewhere).")
        else:
            print("[demo-buffer] target_demo_path empty -- TARGET demos skipped "
                  "(stage-1 does not use them; finetune_steps must stay 0)")

    def _maybe_set_state_normalization(self):
        """Raw-state world model (encoder_type='identity'): fit per-dim z-score
        stats on the SOURCE replay states and inject them into the world model
        (persisted in the ckpt buffers, so eval self-normalizes). No-op for the
        default latent encoder (SimNorm bounds the latent; no z-score needed)."""
        wm = self.cfg.simdist.world_model
        if wm.encoder_type != "identity" and getattr(wm, "input_fourier_l", 0) > 0:
            # latent arm + Fourier (PE) encoder input: fit the PE z-score stats.
            mean, std = self.seq_replay.state_stats()
            self.agent.set_input_fourier_stats(mean, std)
            print(f"[fourier-PE] z-score fit on "
                  f"{self.seq_replay.num_transitions()} source states "
                  f"(|mean| avg={mean.abs().mean():.3f}, std avg={std.mean():.3f})")
            return
        if wm.encoder_type != "identity":
            return
        mean, std = self.seq_replay.state_stats()
        self.agent.set_normalization(mean, std)
        print(f"[raw-state] z-score normalization fit on "
              f"{self.seq_replay.num_transitions()} source states "
              f"(|mean| avg={mean.abs().mean():.3f}, std avg={std.mean():.3f})")

    # ----------------------------------------------------------- collect
    def _collect_action(self, obs, stddev: float) -> torch.Tensor:
        """Collection policy. `expert_collect_steps<=0` => all-expert (stable,
        faithful to SimDist's offline expert data-gen); >0 => expert for the
        first N steps then MPPI self-explore (on-policy — can destabilize the
        world model on its own OOD rollouts, watch distill/consistency)."""
        ecs = self.cfg.expert_collect_steps
        use_expert = self.source_agent is not None and (ecs <= 0 or self.global_step < ecs)
        with torch.no_grad(), utils.eval_mode(self.agent):
            if use_expert:
                return self.source_agent.act(obs, eval_mode=False, stddev=stddev)
            return self.agent.act(obs, eval_mode=False, stddev=stddev)

    @torch.no_grad()
    def _source_value_targets(self, states):
        """Dense per-state V_src(s) from the frozen source critic for value
        distillation. states[B,H+1,obs] -> [B,H+1]. None if no source agent.

        With `source_value_ibrl` (default) V is the expert's IBRL HYBRID value
        (max-Q over {RL actor, prior}) -- the policy it actually executes, and the
        one whose actions the policy head is cloning. See MainConfig."""
        if self.source_agent is None:
            return None
        B, Hp1, od = states.shape
        v = self.source_agent.value(
            {"state": states.reshape(B * Hp1, od)}, use_ibrl=self.source_value_ibrl
        )
        return v.reshape(B, Hp1)

    # -------------------------------------------------------------------- eval
    def eval(self, seed, env_params=None) -> float:
        env_params = env_params if env_params is not None else self.eval_env_params
        random_state = np.random.get_state()
        if self.cfg.mp_eval:
            scores = run_eval_mp(env_params=env_params, agent=self.agent, num_proc=10,
                                 num_game=self.cfg.num_eval_episode, seed=seed, verbose=False)
        else:
            scores = run_eval(env_params=env_params, agent=self.agent,
                              num_game=self.cfg.num_eval_episode, seed=seed,
                              record_dir=None, verbose=False)
        np.random.set_state(random_state)
        return float(np.mean(scores))

    def _source_eval_params(self):
        params = dict(self.eval_env_params)
        params["domain_shift"] = DomainShiftConfig(mode="none")
        return params

    # ----------------------------------------------------------------- warm-up
    def warm_up(self):
        obs, _ = self.train_env.reset()
        self.seq_replay.new_episode(obs["state"])
        target = self.cfg.num_warm_up_episode
        pbar = tqdm(total=target, desc="warm-up (source)", dynamic_ncols=True)
        while True:
            stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
            action = self._collect_action(obs, stddev)
            obs, reward, terminal, success, _ = self.train_env.step(action)
            self.seq_replay.add(obs["state"], action, reward, terminal, success)
            if terminal:
                pbar.n = min(self.seq_replay.size(), target)
                pbar.set_postfix(succ=self.seq_replay.num_success)
                pbar.refresh()
                if self.seq_replay.size() < target:
                    obs, _ = self.train_env.reset()
                    self.seq_replay.new_episode(obs["state"])
                else:
                    break
        pbar.close()
        print(f"Warm up done. source replay #episode: {self.seq_replay.size()}")

    # --------------------------------------- stage 1: offline distill (方案 B)
    def _train_offline(self):
        """Distill the world model from the pre-collected MIXED source replay.
        No env interaction; `distill_steps` = number of training updates. Value
        labels come from the frozen source critic (V_src), same as online."""
        print(common_utils.wrap_ruler(
            f"stage 1: OFFLINE distill from {self.seq_replay.size()} replay "
            f"trajectories ({self.seq_replay.num_transitions()} transitions, "
            f"{self.seq_replay.num_success} success)"))
        self._maybe_set_state_normalization()
        stopwatch = common_utils.Stopwatch()
        pbar = tqdm(total=self.cfg.distill_steps, desc="simdist distill (offline)",
                    initial=self.global_step, dynamic_ncols=True, mininterval=1.0)
        while self.global_step < self.cfg.distill_steps:
            with stopwatch.time("train"):
                sampled = self.seq_replay.sample(
                    self.cfg.batch_size, self.H, "cuda",
                    return_mask=bool(self.cfg.bc_expert_only))
                states, actions, rewards = sampled[:3]
                policy_mask = sampled[3] if self.cfg.bc_expert_only else None
                value_targets = self._source_value_targets(states)
                p_states = p_actions = p_mask = None
                if self.policy_replay is not None:
                    p_sampled = self.policy_replay.sample(
                        self.cfg.batch_size, self.H, "cuda",
                        return_mask=bool(self.cfg.bc_expert_only))
                    p_states, p_actions, _ = p_sampled[:3]
                    p_mask = p_sampled[3] if self.cfg.bc_expert_only else None
                self.stat.append(self.agent.update_distill(
                    states, actions, rewards, value_targets, policy_mask=policy_mask,
                    policy_states=p_states, policy_actions=p_actions,
                    policy_states_mask=p_mask))
                self.train_step += 1
            self.global_step += 1
            pbar.update(1)
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, phase="distill")
        pbar.close()

    # ------------------------------------------------------- stage 1: distill
    def train(self):
        self.stat = common_utils.MultiCounter(
            self.work_dir, bool(self.cfg.use_wb), wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run, wb_group_name=self.cfg.wb_group, config=self.cfg_dict,
        )
        self.saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)
        self.load_resume()

        if self.offline_source:
            self._train_offline()
            return

        if self.seq_replay.num_episode < self.cfg.num_warm_up_episode:
            self.warm_up()
        self._maybe_set_state_normalization()

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        self.seq_replay.new_episode(obs["state"])
        pbar = tqdm(total=self.cfg.distill_steps, desc="simdist distill",
                    initial=self.global_step, dynamic_ncols=True, mininterval=1.0)
        recent: list[float] = []
        while self.global_step < self.cfg.distill_steps:
            with stopwatch.time("act"):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self._collect_action(obs, stddev)
                self.stat["data/stddev"].append(stddev)
            with stopwatch.time("env step"):
                obs, reward, terminal, success, _ = self.train_env.step(action)
            with stopwatch.time("add"):
                self.seq_replay.add(obs["state"], action, reward, terminal, success)
                self.global_step += 1
                pbar.update(1)
            if terminal:
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    self.stat["score/train_score"].append(float(success))
                    recent.append(float(success))
                    if len(recent) > 20:
                        recent.pop(0)
                    pbar.set_postfix(ep=self.global_episode,
                                     succ=f"{np.mean(recent):.2f}", replay=self.seq_replay.size())
                    obs, _ = self.train_env.reset()
                    self.seq_replay.new_episode(obs["state"])
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, phase="distill")
            if self.global_step % self.cfg.update_freq == 0 and self.seq_replay.can_sample(self.H):
                with stopwatch.time("train"):
                    sampled = self.seq_replay.sample(
                        self.cfg.batch_size, self.H, "cuda",
                        return_mask=bool(self.cfg.bc_expert_only))
                    states, actions, rewards = sampled[:3]
                    policy_mask = sampled[3] if self.cfg.bc_expert_only else None
                    value_targets = self._source_value_targets(states)
                    self.stat.append(self.agent.update_distill(
                        states, actions, rewards, value_targets, policy_mask=policy_mask))
                    self.train_step += 1
        pbar.close()

    # ----------------------------------------------------- stage 2: finetune
    def finetune(self):
        steps = self.cfg.simdist.finetune_steps
        if steps <= 0:
            print("finetune_steps<=0; skipping stage 2")
            return
        if self.demo_replay.size() == 0:
            raise RuntimeError(
                f"finetune_steps={steps} but target_demo_path is empty -- stage 2 "
                "has no TARGET demos to fit. Set target_demo_path for this task.")
        print(common_utils.wrap_ruler("stage 2: finetune dynamics+policy on TARGET demos"))
        stopwatch = common_utils.Stopwatch()
        pbar = tqdm(total=steps, desc="simdist finetune", initial=self.finetune_step,
                    dynamic_ncols=True, mininterval=1.0)
        if self.finetune_step:
            print(f"[resume] finetune continues at {self.finetune_step}/{steps}")
        for step in range(self.finetune_step, steps):
            states, actions, rewards = self.demo_replay.sample(self.cfg.batch_size, self.H, "cuda")
            self.stat.append(self.agent.update_finetune(states, actions, rewards))
            self.global_step += 1
            self.finetune_step = step + 1
            pbar.update(1)
            if (step + 1) % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, phase="finetune")
        pbar.close()
        # final eval + save
        self.log_and_save(stopwatch, phase="finetune")

    # --------------------------------------------------------------- resume
    def _resume_path(self) -> str:
        if self.cfg.resume in ("auto", "1", "AUTO"):
            return os.path.join(self.work_dir, "resume.pt")
        return self.cfg.resume

    def save_resume(self, phase: str = "distill"):
        """Write FULL training state so a killed run continues instead of restarting.

        TopkSaver's model0.pt/latest.pt hold WEIGHTS ONLY. Restarting from those is
        not a resume: Adam's moments restart at zero and global_step rewinds to 0,
        so the lr/stddev schedules replay and the run re-does work it already did.
        Written atomically (tmp + os.replace) so a kill mid-write cannot leave a
        truncated resume.pt -- the previous one stays valid.
        """
        state = {
            "agent": self.agent.state_dict(),
            "distill_opt": self.agent.distill_opt.state_dict(),
            "finetune_opt": (self.agent.finetune_opt.state_dict()
                             if self.agent.finetune_opt is not None else None),
            "global_step": self.global_step,
            "global_episode": self.global_episode,
            "train_step": self.train_step,
            "finetune_step": self.finetune_step,
            "phase": phase,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            # provenance / sanity-check fields
            "distill_steps": self.cfg.distill_steps,
            "offline_source": bool(self.offline_source),
            "source_replay_path": self.cfg.source_replay_path,
        }
        path = os.path.join(self.work_dir, "resume.pt")
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_resume(self) -> bool:
        """Restore state written by save_resume(). Returns True if anything loaded."""
        path = self._resume_path()
        if not path:
            return False
        if not os.path.exists(path):
            print(f"[resume] {path} not found -- starting from scratch")
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)

        if not (isinstance(state, dict) and "agent" in state):
            # A TopkSaver latest.pt/model0.pt: weights only. Runs that died before
            # save_resume() existed can still skip their completed steps, but this
            # is a DEGRADED resume -- say so loudly rather than let it look exact.
            self.agent.load_state_dict(state)
            self.global_step = max(int(self.cfg.resume_step), 0)
            self.train_step = self.global_step // max(self.cfg.update_freq, 1) \
                if not self.offline_source else self.global_step
            print(f"[resume] DEGRADED: {path} is a weights-only ckpt.")
            print(f"[resume]   - weights restored; Adam moments restart at ZERO")
            print(f"[resume]   - RNG stream not restored")
            print(f"[resume]   - global_step set from --resume_step to {self.global_step}"
                  + ("  <-- 0 means the schedules replay from the start; pass the "
                     "step the dead run reached" if self.global_step == 0 else ""))
            return True

        # Refuse silently-wrong resumes rather than producing a run whose numbers
        # cannot be compared to anything (CLAUDE.md rule 17: the env/data follow
        # the ckpt, so a mismatched replay path is a different experiment).
        if bool(state.get("offline_source")) != bool(self.offline_source):
            raise RuntimeError(
                f"[resume] offline_source mismatch: ckpt={state.get('offline_source')} "
                f"vs cfg={bool(self.offline_source)}")
        old_replay = state.get("source_replay_path", "")
        if old_replay and old_replay != self.cfg.source_replay_path:
            raise RuntimeError(
                f"[resume] source_replay_path changed:\n  ckpt: {old_replay}\n"
                f"  cfg : {self.cfg.source_replay_path}\n"
                "Resuming onto different data is a different experiment; "
                "point --source_replay_path back, or drop --resume.")

        self.agent.load_state_dict(state["agent"])
        self.agent.distill_opt.load_state_dict(state["distill_opt"])
        if state.get("finetune_opt") is not None and self.agent.finetune_opt is not None:
            self.agent.finetune_opt.load_state_dict(state["finetune_opt"])
        self.global_step = int(state["global_step"])
        self.global_episode = int(state["global_episode"])
        self.train_step = int(state["train_step"])
        self.finetune_step = int(state.get("finetune_step", 0))

        # RNG: keep the stream going instead of replaying the same batches/seeds.
        torch.set_rng_state(state["rng_torch"])
        if torch.cuda.is_available() and state.get("rng_cuda"):
            try:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            except Exception as e:   # device-count change -> not fatal
                print(f"[resume] could not restore CUDA RNG ({e}); continuing")
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])

        old_total = state.get("distill_steps")
        if old_total != self.cfg.distill_steps:
            print(f"[resume] NOTE distill_steps changed {old_total} -> "
                  f"{self.cfg.distill_steps}")
        print(f"[resume] loaded {path}: phase={state.get('phase')} "
              f"global_step={self.global_step} train_step={self.train_step} "
              f"finetune_step={self.finetune_step} "
              f"(distill_steps={self.cfg.distill_steps})")
        if not self.offline_source:
            print("[resume] WARNING online stage-1: seq_replay is NOT part of the "
                  "resume state, so the on-policy replay restarts empty and "
                  "warm_up() runs again. Only the offline arm "
                  "(--source_replay_path) resumes exactly.")
        return True

    def log_and_save(self, stopwatch, phase: str):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        self.stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        self.stat["other/episode"].append(self.global_episode)
        self.stat["other/step"].append(self.global_step)
        self.stat["other/replay"].append(self.seq_replay.size())
        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            eval_score = self.eval(seed=eval_seed)  # TARGET (deployment metric)
            self.stat["score/score"].append(eval_score)
            policy_score = None
            if self.cfg.simdist.use_planner:
                # also eval the bare policy head (no MPPI) so the planner's
                # contribution is visible throughout training
                self.agent.cfg.use_planner = 0
                try:
                    policy_score = self.eval(seed=eval_seed)
                finally:
                    self.agent.cfg.use_planner = 1
                self.stat["score/policy_score"].append(policy_score)
            source_score = None
            if self.cfg.eval_source:
                source_score = self.eval(seed=eval_seed, env_params=self._source_eval_params())
                self.stat["score/source_score"].append(source_score)
        saved = self.saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        self.save_resume(phase=phase)
        self.stat.summary(self.global_step, reset=True)
        pol_str = f"  policy_score={policy_score:.3f}" if policy_score is not None else ""
        src_str = f"  source_score={source_score:.3f}" if source_score is not None else ""
        print(f"[eval @ {phase} step {self.global_step}] target_score={eval_score:.3f}{pol_str}{src_str}")
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print(common_utils.get_mem_usage())


def load_model(weight_file, device):
    """Rebuild a SimDistAgent for standalone eval (evaluate/eval.py --mode simdist)."""
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
    workspace.train()      # stage 1
    workspace.finetune()   # stage 2
    if cfg.use_wb:
        import wandb
        wandb.finish()
    # MuJoCo EGL render contexts + replay threads block a clean interpreter
    # shutdown; all real work + saves are done, so flush and hard-exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    from rich.traceback import install
    install()
    os.environ.setdefault("MUJOCO_GL", "egl")   # setdefault, not "=": an outer
    # MUJOCO_GL=osmesa must survive. On MIG-partitioned GPUs (Compute Canada fir)
    # `import mujoco` with MUJOCO_GL=egl dies at IMPORT time -- "your EGL driver
    # does not support the PLATFORM_DEVICE extension" -- so those runs must pass
    # MUJOCO_GL=osmesa from outside, together with --state_only_no_render 1 so no
    # GL context is ever actually created.
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
