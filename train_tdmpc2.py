"""Online TD-MPC2 with an IBRL prior, on SOURCE or TARGET (CLAUDE.md §28).

The head-to-head against SimDist (§17/§24): SAME latent world model family, SAME
MPPI execution — but the model+value are learned ONLINE from the agent's own
interaction (TD-MPC2) instead of DISTILLED from a frozen expert's data. The
domain is chosen entirely by the config's `train_domain_shift`:

  SOURCE arm : train_domain_shift.mode=none  (+ eval on source)
  TARGET arm : train_domain_shift = the §12 dynamics-only shift (+ eval on target)

Sparse reward (0/1 + end_on_success) makes from-scratch TD-MPC2 hopeless here, so
a FROZEN prior policy is attached and the LEARNED Q arbitrates between it and the
world model's action (IBRL; see rl/tdmpc2_agent.py for the three touch points).
The prior also drives warm-up collection, so the buffer contains successes from
step 0 and the reward/Q heads have something to fit.

Run:
  python train_tdmpc2.py --config_path release/cfgs/mimicgen/threading_d0_state_tdmpc2_source.yaml \
      --save_dir exps/mimicgen/threading_tdmpc2_source_seed1
  python train_tdmpc2.py --config_path release/cfgs/mimicgen/threading_d0_state_tdmpc2_target.yaml \
      --save_dir exps/mimicgen/threading_tdmpc2_target_seed1
"""

import os
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
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.tdmpc2_agent import TDMPC2Agent, TDMPC2Config
from rl.sequence_replay import SequenceReplayBuffer, load_demos_to_sequence_buffer


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
    state_stack: int = 1  # MUST be 1 (matches the prior ckpts' obs space)
    use_state: int = 1
    # agent
    tdmpc2: TDMPC2Config = field(default_factory=TDMPC2Config)
    discount: float = 0.99
    env_reward_scale: float = 1
    batch_size: int = 256
    # exploration noise added ON TOP of the planner's own std (TD-MPC2 explores via
    # the CEM std). 0 = pure TD-MPC2 exploration.
    stddev_max: float = 0.0
    stddev_min: float = 0.0
    stddev_step: int = 200000
    # ------------------------------------------------------------- IBRL prior
    # prior_kind: "bc"   -> a trained BC head        (train_bc.load_model)
    #             "rl"   -> a frozen RL agent        (train_rl.load_frozen_agent)
    #             "none" -> no prior (vanilla TD-MPC2; expect 0 under sparse reward)
    prior_path: str = ""
    prior_kind: str = "bc"
    # OPTIONAL prior-margin anneal — OFF by default (`-1`), because vanilla IBRL does
    # NOT use one: `threading_ibrl_*` runs pure greedy max-Q with prior_margin=0 from
    # step 0. What lets IBRL survive that cold start is the demo-mixed batch
    # (`mix_rl_rate`, below) — its actor has no BC anchor either (add_bc_loss=0).
    # Set >=0 to re-enable the schedule (init -> tdmpc2.prior_margin over
    # prior_margin_steps) as an ablation / for the §26 permanent-prior variant.
    prior_margin_init: float = -1.0
    prior_margin_steps: int = 50000
    # --------------------------------------------------------------- schedule
    num_train_step: int = 400000        # online env steps
    update_freq: int = 1                # 1 update per env step (TD-MPC2 default)
    num_warm_up_episode: int = 20       # prior-driven (or random) seed episodes
    pretrain_updates: int = 5000        # burst of updates on the seed data (TD-MPC2
                                        # does `seed_steps` updates before acting)
    warm_up_stddev: float = 0.1         # exploration noise for the warm-up policy
    # ------------------------------------------- IBRL's demo-mixed batch (mix_rl_rate)
    # THE mechanism that makes vanilla IBRL survive a sparse-reward cold start: a fixed
    # fraction of EVERY batch is drawn from the demo buffer, so the reward/Q heads have
    # a success signal that does NOT depend on the agent's own rollouts succeeding.
    # (Merely PRELOADING demos into the online buffer is not enough — 10-20 demo
    # episodes get diluted to <1% of a 5000-episode ring buffer and are ~never sampled.)
    # mix_rl_rate = fraction of the batch from the ONLINE replay; the rest from demos.
    # 1.0 = no demo mixing (pure TD-MPC2 ablation).
    mix_rl_rate: float = 0.5
    demo_path: str = ""
    demo_num_data: int = 0
    replay_max_episode: int = 5000
    # log / eval
    log_per_step: int = 10000
    num_eval_episode: int = 50
    eval_source: int = 0                # extra SOURCE eval as a diagnostic
    mp_eval: int = 1
    save_dir: str = "exps/tdmpc2/run1"
    use_wb: int = 0
    # domains
    train_domain_shift: DomainShiftConfig = field(
        default_factory=lambda: DomainShiftConfig(mode="none"))
    eval_domain_shift: DomainShiftConfig = field(
        default_factory=lambda: DomainShiftConfig(mode="none"))
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    train_hand_init_perturb_joint_sigma: Optional[float] = None
    eval_hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        if self.prior_path in ["none", "None"]:
            self.prior_path = ""
        if self.demo_path in ["none", "None"]:
            self.demo_path = ""
        assert self.use_state, "TD-MPC2 baseline here is state-only"
        assert self.state_stack == 1, "state_stack must be 1 (prior obs space)"
        self.stddev_min = min(self.stddev_max, self.stddev_min)
        self.robots: list[str] = ["Panda"]
        self.tdmpc2.discount = self.discount
        # Snapshot the FINAL margin. The training loop anneals by WRITING into
        # `agent.cfg.prior_margin`, and `agent.cfg` IS this `tdmpc2` object — so
        # reading the target off `tdmpc2.prior_margin` at schedule time would read
        # back the value we just wrote and pin the margin at prior_margin_init
        # forever (the world model would never take over).
        self.prior_margin_final: float = float(self.tdmpc2.prior_margin)

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"

    @property
    def prior_margin_schedule(self):
        return (f"linear({self.prior_margin_init},{self.prior_margin_final},"
                f"{self.prior_margin_steps})")


def load_prior(cfg: MainConfig, env: PixelRobosuite, device: str):
    """Build the frozen IBRL prior. Returns (policy_or_None, description)."""
    if not cfg.prior_path or cfg.prior_kind == "none":
        return None, ""
    if cfg.prior_kind == "bc":
        import train_bc
        policy, _, _ = train_bc.load_model(cfg.prior_path, device)
        return policy, f"BC head <- {cfg.prior_path}"
    if cfg.prior_kind == "rl":
        from train_rl import load_frozen_agent, FrozenAgentBCPolicy
        agent = load_frozen_agent(cfg.prior_path, env, cfg, device)  # type: ignore
        return FrozenAgentBCPolicy(agent), f"frozen RL agent <- {cfg.prior_path}"
    raise ValueError(f"unknown prior_kind: {cfg.prior_kind}")


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
        self.H = cfg.tdmpc2.horizon
        self._setup_env()

        self.agent = TDMPC2Agent(
            bool(cfg.use_state),
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            cfg.rl_camera,
            cfg.tdmpc2,
            device="cuda",
        )

        prior, desc = load_prior(cfg, self.train_env, "cuda")
        self.prior = prior
        if prior is not None:
            self.agent.attach_prior(prior)
            print(f"[IBRL] prior = {desc} | ibrl={cfg.tdmpc2.ibrl} "
                  f"margin={cfg.tdmpc2.prior_margin} "
                  f"num_prior_trajs={cfg.tdmpc2.num_prior_trajs}")
        else:
            print("[IBRL] NO prior attached -> vanilla TD-MPC2 (sparse reward: "
                  "expect the reward/Q heads to see no success)")

        if not from_main:
            return
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
        )
        # ONLINE training env: source or target, per train_domain_shift.
        self.train_env = PixelRobosuite(
            **common_env,
            env_reward_scale=self.cfg.env_reward_scale,
            end_on_success=bool(self.cfg.end_on_success),
            domain_shift=self.cfg.train_domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.train_hand_init_perturb_joint_sigma,
        )
        self.eval_env_params = dict(
            **common_env,
            domain_shift=self.cfg.eval_domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.eval_hand_init_perturb_joint_sigma,
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _setup_replay(self):
        # ONLINE replay (the agent's own rollouts).
        self.replay = SequenceReplayBuffer(
            max_episodes=self.cfg.replay_max_episode, device="cpu")
        # DEMO replay, kept SEPARATE (IBRL's bc_replay) so `mix_rl_rate` can force a
        # fixed demo fraction into every batch. Preloading demos into the online
        # buffer instead would dilute them to ~0 and defeat the whole mechanism.
        self.demo_replay = None
        if self.cfg.demo_path and self.cfg.demo_num_data > 0:
            self.demo_replay = SequenceReplayBuffer(max_episodes=10000, device="cpu")
            load_demos_to_sequence_buffer(
                self.demo_replay, self.cfg.demo_path, num_data=self.cfg.demo_num_data,
                reward_scale=self.cfg.env_reward_scale)
            assert self.demo_replay.can_sample(self.H), \
                f"no demo episode of length >= horizon={self.H}"
        if self.demo_replay is None and self.cfg.mix_rl_rate < 1:
            raise ValueError(
                "mix_rl_rate < 1 needs a demo buffer (set demo_path/demo_num_data), "
                "else there is nothing to mix in. Set mix_rl_rate=1 for pure TD-MPC2.")

    # ---------------------------------------------------------------- collect
    def _warm_up_action(self, obs, stddev: float) -> torch.Tensor:
        """Seed-phase policy: the frozen prior (+noise) if we have one, else
        uniform random. With a sparse reward a random seed phase yields ZERO
        successes, so the prior is what makes the reward/Q heads learnable."""
        if self.prior is None:
            return torch.empty(self.train_env.action_dim).uniform_(-1, 1)
        with torch.no_grad(), utils.eval_mode(self.agent):
            a = self.prior.act({"state": obs["state"].cuda()}, eval_mode=True, cpu=True)
        if a.dim() == 2:
            a = a[0]
        if stddev > 0:
            a = (a + torch.randn_like(a) * stddev).clamp(-1, 1)
        return a

    # -------------------------------------------------------------------- eval
    def eval(self, seed, env_params=None) -> float:
        env_params = env_params if env_params is not None else self.eval_env_params
        random_state = np.random.get_state()
        with utils.eval_mode(self.agent):
            if self.cfg.mp_eval:
                scores = run_eval_mp(env_params=env_params, agent=self.agent, num_proc=10,
                                     num_game=self.cfg.num_eval_episode, seed=seed,
                                     verbose=False)
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
        target = self.cfg.num_warm_up_episode
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs["state"])
        pbar = tqdm(total=target, desc="warm-up (prior)", dynamic_ncols=True)
        n_ep = 0
        while n_ep < target:
            action = self._warm_up_action(obs, self.cfg.warm_up_stddev)
            obs, reward, terminal, success, _ = self.train_env.step(action)
            self.replay.add(obs["state"], action, reward, terminal, success)
            if terminal:
                n_ep += 1
                pbar.n = n_ep
                pbar.set_postfix(succ=self.replay.num_success)
                pbar.refresh()
                if n_ep < target:
                    obs, _ = self.train_env.reset()
                    self.replay.new_episode(obs["state"])
        pbar.close()
        print(f"warm-up done: {self.replay.size()} episodes, "
              f"{self.replay.num_transitions()} transitions, "
              f"{self.replay.num_success} success")

    def _update_once(self):
        """IBRL's demo-mixed batch: `mix_rl_rate` of the batch from the ONLINE replay,
        the rest from the DEMO replay (= train_rl.py's `sample_rl_bc`). The world model
        heads therefore always see successful transitions, whether or not the agent's
        own rollouts are succeeding yet."""
        B = self.cfg.batch_size
        n_rl = int(B * self.cfg.mix_rl_rate)
        n_demo = B - n_rl
        parts = []
        if n_rl > 0:
            parts.append(self.replay.sample(n_rl, self.H, "cuda"))
        if n_demo > 0:
            parts.append(self.demo_replay.sample(n_demo, self.H, "cuda"))
        states, actions, rewards = (torch.cat([p[i] for p in parts], dim=0)
                                    for i in range(3))

        bc_states = bc_actions = None
        if self.demo_replay is not None and self.cfg.tdmpc2.bc_loss_coef > 0:
            d_s, d_a, _ = self.demo_replay.sample(B, self.H, "cuda")
            bc_states = d_s[:, : self.H].reshape(-1, d_s.shape[-1])
            bc_actions = d_a.reshape(-1, d_a.shape[-1])

        self.stat.append(self.agent.update(states, actions, rewards,
                                           bc_states=bc_states, bc_actions=bc_actions))
        self.stat["data/demo_frac"].append(n_demo / B)
        self.train_step += 1

    def pretrain(self):
        """TD-MPC2 does `seed_steps` gradient updates on the seed data before it
        starts acting with the model. Same idea: give the Q/reward/dynamics heads
        a head start on the prior's (successful) warm-up data."""
        n = self.cfg.pretrain_updates
        if n <= 0:
            return
        for _ in tqdm(range(n), desc="pretrain on seed data", dynamic_ncols=True,
                      mininterval=1.0):
            self._update_once()

    # ------------------------------------------------------------------- train
    def train(self):
        self.stat = common_utils.MultiCounter(
            self.work_dir, bool(self.cfg.use_wb), wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run, wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        if self.replay.num_episode < self.cfg.num_warm_up_episode:
            self.warm_up()
        assert self.replay.can_sample(self.H), \
            f"no warm-up episode of length >= horizon={self.H}"
        self.pretrain()

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs["state"])
        pbar = tqdm(total=self.cfg.num_train_step, desc="tdmpc2", dynamic_ncols=True,
                    mininterval=1.0)
        recent: list[float] = []
        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                if self.agent.has_prior and self.cfg.prior_margin_init >= 0:
                    # anneal the prior's preference away as the critic becomes
                    # informative (see MainConfig.prior_margin_init)
                    self.agent.cfg.prior_margin = utils.schedule(
                        self.cfg.prior_margin_schedule, self.global_step)
                    self.stat["ibrl/prior_margin"].append(self.agent.cfg.prior_margin)
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=False, stddev=stddev)
                if self.agent.has_prior:
                    # fraction of collection steps where the learned Q preferred the
                    # world model's action over the prior's (IBRL takeover rate)
                    self.stat["ibrl/act_take_rl"].append(self.agent._last_take_rl)
            with stopwatch.time("env step"):
                obs, reward, terminal, success, _ = self.train_env.step(action)
            with stopwatch.time("add"):
                self.replay.add(obs["state"], action, reward, terminal, success)
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
                                     succ=f"{np.mean(recent):.2f}",
                                     replay=self.replay.size())
                    obs, _ = self.train_env.reset()
                    self.replay.new_episode(obs["state"])
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self._update_once()
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch)
        pbar.close()

    def log_and_save(self, stopwatch):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        self.stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        self.stat["other/episode"].append(self.global_episode)
        self.stat["other/step"].append(self.global_step)
        self.stat["other/replay"].append(self.replay.size())
        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            eval_score = self.eval(seed=eval_seed)
            self.stat["score/score"].append(eval_score)
            source_score = None
            if self.cfg.eval_source:
                source_score = self.eval(seed=eval_seed,
                                         env_params=self._source_eval_params())
                self.stat["score/source_score"].append(source_score)
        saved = self.saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        self.stat.summary(self.global_step, reset=True)
        src = f"  source_score={source_score:.3f}" if source_score is not None else ""
        print(f"[eval @ step {self.global_step}] score={eval_score:.3f}{src}")
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print(common_utils.get_mem_usage())


def load_model(weight_file, device):
    """Rebuild a TDMPC2Agent for standalone eval (evaluate/eval.py --mode tdmpc2).
    The IBRL prior is not in the ckpt (held in a list), so re-attach it by path."""
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
    os._exit(0)  # MuJoCo EGL contexts block a clean shutdown; all saves are done


if __name__ == "__main__":
    from rich.traceback import install
    install()
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
