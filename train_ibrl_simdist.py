"""IBRL with the frozen ADAPTED SimDist base as the BC prior (vs the residual §25).

Same setup as train_residual_simdist.py — the adapted SimDist world model
(threading_simdist_adapt_seed1, §24; MPPI ≈ 0.75 on TARGET) is the frozen base —
but instead of learning a bounded RESIDUAL on top of it, this trains standard
IBRL (Hu et al.): an RL actor from scratch whose action competes with the base's
in a max-Q selection `a = argmax_{a in {pi_rl(s), pi_base(s)}} Q(s,a)`, plus the
RLPD ingredients (demo-mixed replay, RED-Q critic, BC-loss actor regularization).
The base is the IBRL BC prior. Train + eval in the TARGET domain — a head-to-head
"residual on the base" vs "IBRL with the base as prior" comparison.

The base's deployment behavior is MPPI, far too slow to re-query 256x per critic
update (IBRL's max-Q bootstrap calls pi_base(s') on the whole batch). So — exactly
as in the residual run — the base (MPPI) action is computed ONCE per collected env
step and STORED in the buffer's obs under `base_action`; a `StoredActionBCPolicy`
wrapper returns that stored action during training and falls back to LIVE MPPI at
eval (where the env obs carries no `base_action`). MPPI is thus queried only at
collection (1 plan/step) and eval (1 plan/step).

Run:
  python train_ibrl_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_ibrl_simdist.yaml \
    --save_dir exps/mimicgen/threading_ibrl_simdist_seed1
"""

import os
import sys
import json
import copy
from dataclasses import dataclass, field
from typing import Optional

import yaml
import h5py
import pyrallis
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite, STATE_KEYS
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.q_agent import QAgent, QAgentConfig
from rl import replay

import train_simdist
from rl.simdist_agent import SimDistAgent
from rl.simdist_planner import apply_original_recipe


class StoredActionBCPolicy(nn.Module):
    """Adapts the frozen adapted SimDist base (MPPI) to the IBRL BC-prior interface
    `act(obs, ...) -> [B, action_dim]`. During training the base action is read from
    `obs["base_action"]` (precomputed once per env step at collection time and
    stored in the replay), so the expensive MPPI planner is NOT re-run per update.
    At eval (env obs has no `base_action`) it falls back to LIVE MPPI on the base.
    The base is list-held so it is NOT part of this module's (empty) state_dict."""

    def __init__(self, base: SimDistAgent):
        super().__init__()
        self._base = [base]
        self.training = False

    def train(self, mode: bool = True):
        self.training = False  # always frozen
        return self

    @torch.no_grad()
    def act(self, obs, *, eval_mode: bool = True, stddev: float = 0.0, cpu: bool = True):
        if "base_action" in obs and obs["base_action"] is not None:
            a = obs["base_action"]
            if a.dim() == 1:
                a = a.unsqueeze(0)
            a = a.float().clamp(-1.0, 1.0)
        else:
            # eval path: run the real MPPI base (handles single + batched obs)
            a = self._base[0].act(obs, eval_mode=True, stddev=0.0, cpu=False)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            a = a.float().clamp(-1.0, 1.0)
        return a.cpu() if cpu else a


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
    # agent (IBRL)
    q_agent: QAgentConfig = field(default_factory=QAgentConfig)
    stddev_max: float = 0.1
    stddev_min: float = 0.1
    stddev_step: int = 500000
    # RL
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 100000
    batch_size: int = 256
    mix_rl_rate: float = 0.5     # RLPD: fraction of each RL batch that is online (rest = demos)
    num_critic_update: int = 5   # RED-Q
    update_freq: int = 1
    env_reward_scale: float = 1.0
    # base = the ADAPTED SimDist ckpt (its dir cfg.yaml -> stage1_ckpt -> stage-1
    # cfg.yaml rebuilds the world model; MPPI forced on + original recipe)
    base_adapt_ckpt: str = "exps/mimicgen/threading_simdist_adapt_seed1/model0.pt"
    # Planner overrides applied AFTER apply_original_recipe, same syntax as
    # train_simdist_adapt.py's --planner_set. Needed because the original recipe's
    # init_std=1.0 is NOT portable across tasks (CLAUDE.md rule 9): on TPA_D0,
    # init_std 1.0 -> 0.15 was worth +0.05..0.08 (0.35-0.40 -> 0.42-0.46, n=100).
    # Empty = untouched, so existing runs stay bit-identical.
    planner_set: str = ""
    # WARM-START the trainable actor/critic from a competent QAgent ckpt (arch must
    # match q_agent). "" = cold start (prior-only, stalls — see ibrl_simdist ~50%).
    load_pretrained_agent: str = ""
    # demos seeded into the replay (RLPD anchor + BC-loss target): the TARGET demos
    offline_demo_path: str = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
    offline_num_data: int = 20
    # ALSO seed the replay with a dumped TARGET rollout set (episode-format npz,
    # e.g. the stage-1 SimDist adapt buffer). Its `action` is the MPPI action of a
    # NON-STATIONARY dynamics model, so `base_action` is RE-PLANNED with the final
    # frozen base rather than reusing the stored action. "" = off.
    offline_transition_path: str = ""
    # Load an offline-IQL-pretrained QAgent critic / actor (see
    # scripts/pretrain_iql_qagent.py -> iql_qagent.pt). Targets are hard-synced.
    # Unlike `load_pretrained_agent` these are trained on TARGET-domain offline
    # data, not warm-started from another agent. "" = off.
    init_critic_from: str = ""
    init_actor_from: str = ""
    # ALSO harvest successful MPPI-base rollouts during warmup into the replay
    offline_harvest_success: int = 100
    # OFFLINE critic pre-training: # of pure-critic Bellman updates on the (real,
    # σ=0.1 TARGET) demos+warmup-rollouts before the online loop. 0 = off (=h5warm).
    critic_pretrain_steps: int = 0
    # IBRL prior-preference margin annealing: keep max-Q deferring to the (0.94)
    # MPPI prior while the critic calibrates, so eval starts at the prior's score
    # (no early dip) and the executed data is clean prior-success data (calibrates Q
    # FAST + correctly). margin(t) = prior_margin_init * max(0, 1 - t/prior_margin_steps),
    # subtracted from the RL actor's Q. init >= 1.0 (> any Q gap) => always prior early;
    # anneals to 0 (full IBRL) by prior_margin_steps. 0 init = off (=h5warm).
    prior_margin_init: float = 0.0
    prior_margin_steps: int = 15000
    # PERMANENT margin floor: margin anneals init -> prior_margin_final and STAYS
    # there. A small nonzero floor turns IBRL's max-Q into "safe policy
    # improvement": defer to the MPPI prior unless the RL actor is CONFIDENTLY
    # better by `final`. This counteracts the critic's OOD over-estimation of
    # actor actions (the mechanism that dragged the m->0 runs from 0.85 to 0.70),
    # giving a prior-level floor with unbounded upside. 0 = anneal fully to 0.
    prior_margin_final: float = 0.0
    # VALUE-INIT: # of supervised steps regressing Q(s,a_prior)->V_sim(s) (SimDist
    # value head) before online, to seed the critic with the world model's value.
    # 0 = off. No TD bootstrap => no divergence (unlike critic_pretrain).
    value_init_steps: int = 0
    value_init_lr: float = 1e-4
    # ACTOR-BC PRETRAIN: # of BC steps cloning the RL actor onto the successful
    # trajectories (demos + harvested MPPI successes) before online, to give the
    # actor a competent (~prior-level) legit start. 0 = off.
    actor_bc_pretrain_steps: int = 0
    # MARGIN-FIX: freeze the actor (critic-only updates) while the prior-margin is
    # above `freeze_actor_margin_thresh`, so it can't drift OOD during the
    # deferring phase (see rl_train). Only relevant with prior_margin_init > 0.
    freeze_actor_during_margin: int = 0
    freeze_actor_margin_thresh: float = 0.05
    # warmup: roll the MPPI base to fill the replay before training
    learning_starts: int = 5000
    # schedule
    num_train_step: int = 100000
    log_per_step: int = 1000     # eval every 1000 transitions (matches the residual run)
    num_eval_episode: int = 100
    mp_eval: int = 1
    num_eval_proc: int = 10
    # log
    save_dir: str = "exps/ibrl_simdist/run1"
    use_wb: int = 0
    # TARGET domain (train + eval share it): the §12 dynamics-only shift.
    domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        # pyrallis turns an empty CLI string into "None"; treat both as "unset" so
        # a YAML-provided path can be disabled from the command line.
        for f in ("load_pretrained_agent", "offline_transition_path",
                  "init_critic_from", "init_actor_from"):
            if getattr(self, f) in ("None", "none"):
                setattr(self, f, "")
        assert self.use_state, "state-only"
        assert self.base_adapt_ckpt, "need base_adapt_ckpt (an adapted SimDist ckpt)"
        # actor_bc_pretrain runs AFTER __init__ and would silently overwrite an
        # IQL-pretrained actor. Pick exactly one way to initialize the actor.
        assert not (self.init_actor_from and self.actor_bc_pretrain_steps > 0), (
            "init_actor_from (IQL/AWR actor) and actor_bc_pretrain_steps (MPPI-clone "
            "actor) both initialize the actor; the latter would overwrite the former. "
            "Set actor_bc_pretrain_steps=0 when using init_actor_from.")
        # IBRL: the base is the max-Q prior, so act/bootstrap must be ibrl.
        self.q_agent.act_method = "ibrl"
        if not self.q_agent.bootstrap_method:
            self.q_agent.bootstrap_method = "ibrl"
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

        self.agent = QAgent(
            bool(cfg.use_state),
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            cfg.rl_camera,
            cfg.q_agent,
        )
        self._attach_prior()
        # WARM-START the trainable actor/critic (+ encoder) from a competent QAgent
        # ckpt (e.g. rlprior). Without it, a cold random actor + cold critic + only a
        # prior stalls IBRL (observed: prior-only ibrl_simdist plateaued ~0/50%).
        # The prior (list-held) is NOT in the state_dict, so this only overwrites the
        # trainable nets. strict=False so a mismatched (e.g. unused vit encoder) key
        # doesn't abort. Also hard-sync the targets to the loaded weights.
        if cfg.load_pretrained_agent:
            print(f"[warm-start] loading trainable actor/critic from {cfg.load_pretrained_agent}")
            sd = torch.load(cfg.load_pretrained_agent)
            missing, unexpected = self.agent.load_state_dict(sd, strict=False)
            print(f"[warm-start] missing={len(missing)} unexpected={len(unexpected)} keys")
            # keep target nets consistent with the warm-started online nets
            self.agent.actor_target.load_state_dict(self.agent.actor.state_dict())
            self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())

        # Offline-IQL pretrained nets (TARGET domain). Loaded AFTER load_pretrained_agent
        # so they win if both are set. Each hard-syncs its own target net.
        if cfg.init_critic_from:
            print(f"[iql-init] loading critic from {cfg.init_critic_from}")
            self.agent.critic.load_state_dict(torch.load(cfg.init_critic_from)["critic"])
            self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())
        if cfg.init_actor_from:
            ck = torch.load(cfg.init_actor_from)
            assert "actor" in ck, f"{cfg.init_actor_from} has no 'actor' key (actor_mode=none?)"
            print(f"[iql-init] loading actor from {cfg.init_actor_from} "
                  f"(mode={ck.get('actor_mode', '?')})")
            self.agent.actor.load_state_dict(ck["actor"])
            self.agent.actor_target.load_state_dict(self.agent.actor.state_dict())
        # ref agent for the (static) BC-loss RFT path; act_method="rl" so it never
        # queries the MPPI prior. Only referenced by update_actor_rft's dynamic
        # branch (off here), but update() asserts it is not None when bc_batch given.
        self.ref_agent = self.agent

        if not from_main:
            return
        self._setup_replay()
        self._load_offline_demos()
        if cfg.offline_transition_path:
            self._load_offline_transitions()

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

    def _attach_prior(self):
        """Load the ADAPTED SimDist world model (MPPI) and add it as the IBRL prior."""
        adapt_ckpt = self.cfg.base_adapt_ckpt
        adapt_dir = os.path.dirname(adapt_ckpt)
        adapt_cfg = yaml.safe_load(open(os.path.join(adapt_dir, "cfg.yaml"), "r"))
        stage1_ckpt = adapt_cfg["stage1_ckpt"]
        stage1_cfg_path = os.path.join(os.path.dirname(stage1_ckpt), "cfg.yaml")
        base_cfg: train_simdist.MainConfig = pyrallis.load(
            train_simdist.MainConfig, open(stage1_cfg_path, "r"))  # type: ignore
        base_cfg.simdist.use_planner = 1
        p = apply_original_recipe(base_cfg.simdist.planner)
        for kv in (x for x in self.cfg.planner_set.split(",") if x.strip()):
            k, v = kv.split("=")
            k = k.strip()
            old_v = getattr(p, k)
            setattr(p, k, type(old_v)(v))
            print(f"[prior] planner.{k}: {old_v} -> {getattr(p, k)}")
        print(f"[prior] adapted SimDist MPPI: samples={p.num_samples} "
              f"(base_override={p.num_base_override}), init_std={p.init_std}, "
              f"iters={p.iterations}, horizon={p.horizon}")
        base = SimDistAgent(
            True, self.train_env.observation_shape, self.train_env.prop_shape,
            self.train_env.action_dim, base_cfg.rl_camera, base_cfg.simdist, device="cuda",
        )
        print(f"loading adapted SimDist base ckpt: {adapt_ckpt}")
        base.load_state_dict(torch.load(adapt_ckpt))
        base.eval()
        self.base = base
        self.agent.add_bc_policy(StoredActionBCPolicy(base))

    def _setup_replay(self):
        self.replay = replay.ReplayBuffer(
            self.cfg.nstep, self.cfg.discount, frame_stack=1,
            max_episode_length=self.cfg.episode_length,
            replay_size=self.cfg.replay_buffer_size,
            use_bc=(self.cfg.mix_rl_rate < 1 or self.cfg.q_agent.bc_loss_coef > 0),
        )

    # --------------------------------------------------------- base-action util
    @torch.no_grad()
    def _base_actions(self, states: torch.Tensor) -> torch.Tensor:
        """MPPI base action for a [B, obs] state batch -> [B, act] (on cuda)."""
        a = self.base.act({"state": states.to("cuda").float()},
                          eval_mode=True, stddev=0.0, cpu=False)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return a.float().clamp(-1.0, 1.0)

    def _push_episode(self, states: np.ndarray, base_a: np.ndarray,
                      actions: np.ndarray, rewards: np.ndarray, success: bool):
        """Push one episode into the replay with base_action stored in obs.
        states/base_a: [T, .] (s_0..s_{T-1}); actions/rewards: [T, .]/[T]."""
        T = actions.shape[0]

        def obs_at(t):
            return {"state": torch.as_tensor(states[t]).float(),
                    "base_action": torch.as_tensor(base_a[t]).float()}

        self.replay.new_episode(obs_at(0))
        for t in range(T):
            terminal = (t == T - 1)
            succ = bool(success) if terminal else False
            nxt = obs_at(min(t + 1, T - 1))  # ignored by rela on the terminal step
            reply = {"action": torch.as_tensor(actions[t]).float()}
            self.replay.add(nxt, reply, float(rewards[t]), terminal, succ, image_obs={})

    def _load_offline_demos(self):
        """Seed the replay with the TARGET demos (base_action = MPPI(demo state))."""
        path = self.cfg.offline_demo_path
        cfg_path = os.path.join(os.path.dirname(path), "env_cfg.json")
        task_name = json.load(open(cfg_path, "r"))["env_name"]
        state_keys = STATE_KEYS[task_name]
        f = h5py.File(path, "r")
        num_episode = len(list(f["data"].keys()))
        num_data = self.cfg.offline_num_data
        print(f"[demos] loading {num_data} of {num_episode} TARGET demos from {path}")
        loaded = 0
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
            T = int(np.argmax(rewards == 1)) + 1
            seg_states = states[:T]
            seg_actions = actions[:T]
            seg_rewards = rewards[:T] * self.cfg.env_reward_scale
            base_a = self._base_actions(torch.from_numpy(seg_states)).cpu().numpy()
            self._push_episode(seg_states, base_a, seg_actions, seg_rewards, success=True)
            loaded += 1
        f.close()
        print(f"[demos] seeded {loaded} demos; replay #episode={self.replay.size()}, "
              f"bc_replay #success={self.replay.num_success}")

    def _load_offline_transitions(self):
        """Seed the replay with a dumped TARGET rollout set (episode-format npz).

        The stored actions came from a NON-STATIONARY MPPI (the dynamics model was
        still adapting), so they are NOT this run's frozen base's actions. We
        therefore RE-PLAN base_action for every state with the final base. The
        stored action stays the executed action (that is what the critic must
        evaluate); only the `base_action` obs field is recomputed.
        """
        path = self.cfg.offline_transition_path
        eps = list(np.load(path, allow_pickle=True)["episodes"])
        n_tr = sum(e["actions"].shape[0] for e in eps)
        n_succ = sum(int(e["success"]) for e in eps)
        print(f"[offline-tr] {len(eps)} episodes, {n_tr} transitions, "
              f"{n_succ} success ({n_succ / max(1, len(eps)):.2%}) <- {path}")
        print("[offline-tr] re-planning base_action with the frozen base ...")
        chunk = 256
        for e in tqdm(eps, desc="offline-tr", dynamic_ncols=True):
            T = e["actions"].shape[0]
            seg_states = e["states"][:T]
            base_a = np.concatenate([
                self._base_actions(torch.from_numpy(seg_states[i:i + chunk])).cpu().numpy()
                for i in range(0, T, chunk)], axis=0)
            self._push_episode(seg_states, base_a, e["actions"],
                               e["rewards"] * self.cfg.env_reward_scale,
                               success=bool(e["success"]))
        print(f"[offline-tr] replay #episode={self.replay.size()}, "
              f"bc_replay #success={self.replay.num_success}")

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
        """Roll the adapted MPPI base in the TARGET env for `learning_starts` env
        steps (fills the replay, base_action stored). Successful episodes also
        become BC-replay anchors (via success=True)."""
        obs, _ = self.train_env.reset()
        cur_base = self._base_actions(obs["state"].view(1, -1))[0]  # [act] on cuda
        self.replay.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
        pbar = tqdm(total=self.cfg.learning_starts, desc="warm-up (MPPI base)", dynamic_ncols=True)
        n_steps = 0
        n_succ = 0
        while True:
            action = cur_base.clamp(-1.0, 1.0).cpu()  # pure base (prior) rollout
            next_obs, reward, terminal, success, image_obs = self.train_env.step(action)
            next_base = self._base_actions(next_obs["state"].view(1, -1))[0]
            self.replay.add({"state": next_obs["state"], "base_action": next_base.cpu()},
                            {"action": action}, float(reward), terminal, bool(success),
                            image_obs={})
            obs = next_obs; cur_base = next_base
            n_steps += 1; pbar.update(1)
            if terminal:
                n_succ += int(success)
                if n_steps >= self.cfg.learning_starts:
                    break  # stop on a clean episode boundary
                obs, _ = self.train_env.reset()
                cur_base = self._base_actions(obs["state"].view(1, -1))[0]
                self.replay.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
        pbar.close()
        print(f"warm-up done. replay #episode={self.replay.size()} ({n_steps} steps, "
              f"successes: {n_succ}); bc #success={self.replay.num_success}")

    # ------------------------------------------------------------------- train
    def train(self):
        stat = common_utils.MultiCounter(
            self.work_dir, bool(self.cfg.use_wb), wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run, wb_group_name=self.cfg.wb_group, config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)
        self.warm_up()

        # ACTOR-BC PRETRAIN: clone the RL actor onto the SUCCESSFUL trajectories in
        # the replay (20 target demos + the harvested SimDist MPPI-base successes).
        # This gives the actor a COMPETENT (~0.85, matching the prior) legit starting
        # point — unlike the weak source actor (0.61, drags max-Q down) or a scratch
        # actor (garbage, collapses when the margin releases). With a competent actor,
        # IBRL max-Q between the ~0.85 actor and the 0.94 prior is safe, and the actor
        # can genuinely exceed the prior. All-legit (20 demos + SimDist rollouts).
        if self.cfg.actor_bc_pretrain_steps > 0:
            print(f"[actor-bc] {self.cfg.actor_bc_pretrain_steps} BC steps cloning "
                  "the actor onto successful trajectories (demos + MPPI successes)")
            for i in tqdm(range(self.cfg.actor_bc_pretrain_steps), desc="actor-bc",
                          dynamic_ncols=True, mininterval=1.0):
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")
                m = self.agent.pretrain_actor_with_bc(bc_batch)
                if (i + 1) % max(self.cfg.actor_bc_pretrain_steps // 5, 1) == 0:
                    print(f"[actor-bc] step {i+1}: bc_loss={m.get('pretrain/loss', float('nan')):.4f}")
            self.agent.actor_target.load_state_dict(self.agent.actor.state_dict())

        # VALUE-INIT: seed the critic with the SimDist world model's VALUE head.
        # V_sim(s)=predict_value(encode(s)) ranks target states extremely well
        # (Spearman 0.95 vs true return-to-go), so we SUPERVISE-regress
        # Q(s, a_prior(s)) -> V_sim(s) on the replay states. Because a_prior is the
        # good MPPI action, this teaches the critic that the prior's action is
        # high-value => max-Q defers to the prior from step 0. Pure regression to a
        # FIXED target (no TD self-bootstrap), so it does NOT diverge like offline
        # Bellman pretraining (idea ①). 0 = off.
        if self.cfg.value_init_steps > 0:
            print(f"[value-init] {self.cfg.value_init_steps} steps regressing "
                  "Q(s, a_prior) -> V_sim(s)")
            vopt = torch.optim.Adam(self.agent.critic.parameters(), lr=self.cfg.value_init_lr)
            for i in tqdm(range(self.cfg.value_init_steps), desc="value-init",
                          dynamic_ncols=True, mininterval=1.0):
                b = self.replay.sample(self.cfg.batch_size, "cuda:0")
                s = b.obs["state"]
                with torch.no_grad():
                    a_prior = b.obs["base_action"].to("cuda").float()  # stored MPPI action
                    v_sim = self.base.model.predict_value(self.base.model.encode(s)).detach()
                q = self.agent.critic.forward_k(s, a_prior)  # [B, num_q]
                loss = ((q - v_sim.unsqueeze(-1)) ** 2).mean()
                vopt.zero_grad(set_to_none=True); loss.backward(); vopt.step()
                if (i + 1) % max(self.cfg.value_init_steps // 5, 1) == 0:
                    print(f"[value-init] step {i+1}: mse={loss.item():.4f} "
                          f"q_mean={q.mean().item():.3f} vsim_mean={v_sim.mean().item():.3f}")
            self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())

        # OFFLINE CRITIC PRE-TRAINING: before any online interaction, calibrate the
        # critic on the (all-real, σ=0.1 TARGET) data already in the replay (seeded
        # demos + MPPI-base warmup rollouts) via pure-critic Bellman updates
        # (update_actor=False), bootstrapped along the IBRL max-Q prior. This makes
        # Q correct in the deployment domain from online step 0 — removing the early
        # dip h5warm shows (0.81->0.58) where a σ=0.02-warm-started critic recalibrates
        # online. No model bias (real transitions only).
        if self.cfg.critic_pretrain_steps > 0:
            print(f"[critic-pretrain] {self.cfg.critic_pretrain_steps} offline "
                  "critic-only Bellman updates on demos+rollouts")
            stddev0 = utils.schedule(self.cfg.stddev_schedule, 0)
            for i in tqdm(range(self.cfg.critic_pretrain_steps), desc="critic-pretrain",
                          dynamic_ncols=True, mininterval=1.0):
                if self.cfg.mix_rl_rate < 1 and self.replay.num_success > 0:
                    rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                    batch = self.replay.sample_rl_bc(rl_bsize, self.cfg.batch_size - rl_bsize, "cuda:0")
                else:
                    batch = self.replay.sample(self.cfg.batch_size, "cuda:0")
                m = self.agent.update(batch, stddev0, update_actor=False,
                                      bc_batch=None, ref_agent=self.ref_agent)
                if (i + 1) % max(self.cfg.critic_pretrain_steps // 5, 1) == 0:
                    print(f"[critic-pretrain] step {i+1}: critic_qt={m.get('train/critic_qt', float('nan')):.3f} "
                          f"critic_loss={m.get('train/critic_loss', float('nan')):.4f}")

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        cur_base = self._base_actions(obs["state"].view(1, -1))[0]  # [act] cuda
        self.replay.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
        pbar = tqdm(total=self.cfg.num_train_step, desc="ibrl-simdist train",
                    dynamic_ncols=True, mininterval=1.0)
        recent: list[float] = []
        while self.global_step < self.cfg.num_train_step:
            # annealed prior-preference margin (see cfg): high early -> defer to the
            # 0.94 MPPI prior while the critic calibrates; 0 after prior_margin_steps
            # -> full IBRL. Set on the agent so BOTH the executed action AND the eval
            # (same agent, pickled to eval workers) use the current margin.
            if self.cfg.prior_margin_init > 0:
                frac = max(0.0, 1.0 - self.global_step / max(self.cfg.prior_margin_steps, 1))
                m_final = self.cfg.prior_margin_final
                self.agent.prior_margin = m_final + (self.cfg.prior_margin_init - m_final) * frac
                stat["data/prior_margin"].append(self.agent.prior_margin)
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                # obs["state"] + cur_base are BOTH cuda -> IBRL max-Q reuses the
                # stored MPPI action (no extra plan for action selection).
                action = self.agent.act(
                    {"state": obs["state"], "base_action": cur_base},
                    eval_mode=False, stddev=stddev)
                stat["data/stddev"].append(stddev)
            with stopwatch.time("env step"):
                next_obs, reward, terminal, success, image_obs = self.train_env.step(action)
            with stopwatch.time("base plan"):
                next_base = self._base_actions(next_obs["state"].view(1, -1))[0]
            self.replay.add({"state": next_obs["state"], "base_action": next_base.cpu()},
                            {"action": action}, float(reward), terminal, bool(success),
                            image_obs={})
            obs = next_obs; cur_base = next_base
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
                self.replay.new_episode({"state": obs["state"], "base_action": cur_base.cpu()})
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)
        pbar.close()

    def rl_train(self, stat):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        use_bc_loss = self.cfg.q_agent.bc_loss_coef > 0 and self.replay.num_success > 0
        # MARGIN-FIX: while the prior-preference margin is active, FREEZE the actor
        # (critic-only). This severs the self-defeat loop that collapsed the earlier
        # margin runs — with margin on, execution defers to the prior so the critic
        # never sees actor actions => Q(s,a_actor) is OOD-overestimated, and an
        # actor being updated to maximize that overestimated Q drifts OOD; when the
        # margin releases, max-Q picks the drifted actor and collapses. Freezing the
        # actor until the margin has (nearly) annealed keeps it a sane policy.
        actor_frozen = (self.cfg.freeze_actor_during_margin
                        and self.agent.prior_margin > self.cfg.freeze_actor_margin_thresh)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1 and self.replay.num_success > 0:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")
            update_actor = (i == self.cfg.num_critic_update - 1) and not actor_frozen
            bc_batch = None
            if update_actor and use_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")
            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)
            stat.append(metrics)

    def log_and_save(self, stopwatch, stat, saver):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/replay"].append(self.replay.size())
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
    """Rebuild the IBRL QAgent (+ re-attach the adapted SimDist prior) for
    standalone eval (evaluate/eval.py --mode ibrl_simdist)."""
    cfg_path = os.path.join(os.path.dirname(weight_file), "cfg.yaml")
    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    workspace = Workspace(cfg, from_main=False)
    agent = workspace.agent
    agent.load_state_dict(torch.load(weight_file))
    agent = agent.to(device)
    # CRITICAL: the prior-preference margin IS part of the policy (it biases the
    # IBRL max-Q toward the MPPI prior). In-training eval runs with the annealed
    # floor value; a standalone eval must reproduce it or it evaluates a DIFFERENT
    # (hard max-Q) policy that collapses to ~0.74. Restore the floor.
    if cfg.prior_margin_init > 0:
        agent.prior_margin = cfg.prior_margin_final
        print(f"[load_model] restored prior_margin = {agent.prior_margin}")
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
