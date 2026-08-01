"""Faithful ResFiT residual RL — IN-DOMAIN validation run.

This is the "original" residual off-policy RL (a clean ResFiT port, see
rl/residual_rl_agent.py) with the RLPD ingredients the gap-weighted variant
(train_residual.py) dropped:
  ① every RL minibatch mixes ONLINE + OFFLINE demo transitions (50/50);
  ② LayerNorm critic ensemble + REDQ random-subset min;
  ③ base+noise exploration warmup + critic-only warmup;
  ⑤ deterministic actor Q-max + TD3 target smoothing + grad clip.

NO dynamics gap. Trains AND evals in the SAME (source) domain. The frozen base
is a SOURCE-demo BC policy (threading_source_mlp_bc_n10). Sanity check: in-domain
residual RL on a frozen BC base should clearly beat the base BC.

Run:
  python train_residual_rl.py --config_path release/cfgs/mimicgen/threading_d0_state_residual_rl.yaml \
      --residual.base_policy_path exps/mimicgen/threading_source_mlp_bc_n10_seed1/model0.pt \
      --offline_demo_path release/data/mimicgen/threading_d0_source_sigma0.1_sample10/dataset.hdf5 \
      --save_dir exps/mimicgen/threading_residual_rl_source_seed1
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
from env.robosuite_wrapper import PixelRobosuite, STATE_KEYS
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.residual_rl_agent import ResidualRLAgent, ResidualRLConfig

# train_rebrac's flat demo loader gives exactly the (s,a,r,s',done) layout we mix.
from train_rebrac import _load_demos_flat


class FlatOnlineBuffer:
    """A simple flat ring buffer of single-step transitions on GPU.

    rela's ReplayBuffer returns n-step transitions and ties next_obs to the
    episode's next push — fine for IBRL but awkward to CONCAT with an offline
    demo batch (the RLPD mix). A flat (s,a,r,s',done) store makes the symmetric
    online+offline sampling trivial."""

    def __init__(self, obs_dim, action_dim, capacity, device):
        self.device = device
        self.capacity = capacity
        self.s = torch.zeros(capacity, obs_dim, device=device)
        self.a = torch.zeros(capacity, action_dim, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.s2 = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, device=device)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = torch.as_tensor(s, device=self.device)
        self.a[i] = torch.as_tensor(a, device=self.device)
        self.r[i] = float(r)
        self.s2[i] = torch.as_tensor(s2, device=self.device)
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, n):
        hi = len(self)
        idx = torch.randint(0, hi, (n,), device=self.device)
        return {"states": self.s[idx], "actions": self.a[idx], "rewards": self.r[idx],
                "next_states": self.s2[idx], "dones": self.done[idx]}


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
    state_stack: int = 1  # MUST match the base BC ckpt
    use_state: int = 1
    # agent
    residual: ResidualRLConfig = field(default_factory=ResidualRLConfig)
    stddev_max: float = 0.1   # residual exploration noise (in residual space)
    stddev_min: float = 0.1
    stddev_step: int = 500000
    discount: float = 0.99    # mirrors residual.gamma (kept for schedule symmetry)
    replay_buffer_size: int = 500000
    batch_size: int = 256
    offline_fraction: float = 0.5   # RLPD: fraction of each RL batch from demos
    num_critic_update: int = 1
    update_freq: int = 2            # one RL update per this many env steps
    env_reward_scale: float = 1.0
    # warmup
    learning_starts: int = 5000          # env steps of base+noise before training
    warmup_noise_scale: float = 0.2      # uniform residual noise during warmup
    critic_warmup_steps: int = 10000     # critic-only RL updates before the actor
    # offline demo buffer (RLPD anchor) — the SAME demos the base BC was trained on
    offline_demo_path: str = ""
    offline_num_data: int = 10
    # schedule
    num_train_step: int = 300000
    log_per_step: int = 10000
    num_eval_episode: int = 50
    mp_eval: int = 1
    # log
    save_dir: str = "exps/residual_rl/run1"
    use_wb: int = 0
    # IN-DOMAIN: train + eval share this shift (default none = source).
    domain_shift: DomainShiftConfig = field(default_factory=lambda: DomainShiftConfig(mode="none"))
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        assert self.use_state, "state-only"
        assert self.residual.base_policy_path, "need residual.base_policy_path (a BC ckpt)"
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

        self.agent = ResidualRLAgent(
            self.train_env.observation_shape,
            self.train_env.action_dim,
            cfg.residual,
            device="cuda",
        )
        self._attach_base_policy()

        if not from_main:
            return
        self._setup_data()
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
        import train_bc
        path = self.cfg.residual.base_policy_path
        print(f"loading frozen base BC policy from {path}")
        policy, _, _ = train_bc.load_model(path, "cuda", verbose=False)
        self.agent.attach_base_policy(policy)

    def _setup_data(self):
        """Offline demo buffer (RLPD anchor) + state-normalization stats."""
        assert self.cfg.offline_demo_path, "need offline_demo_path (demos for the RLPD mix)"
        flat = _load_demos_flat(self.cfg.offline_demo_path, self.cfg.offline_num_data,
                                reward_scale=self.cfg.env_reward_scale)
        self.offline = {k: v.to("cuda") for k, v in flat.items()}
        self.n_offline = self.offline["states"].shape[0]
        # z-score stats from the demo states (cheap, stable; matches ReBRAC recipe)
        mean = self.offline["states"].mean(0)
        std = self.offline["states"].std(0)
        self.agent.set_normalization(mean, std)
        print(f"[offline] {self.n_offline} demo transitions for the RLPD mix")

    def _setup_replay(self):
        self.online = FlatOnlineBuffer(
            self.train_env.observation_shape[0], self.train_env.action_dim,
            self.cfg.replay_buffer_size, device="cuda",
        )

    def _sample_offline(self, n):
        keys = ("states", "actions", "rewards", "next_states", "dones")
        idx = torch.randint(0, self.n_offline, (n,), device="cuda")
        return {k: self.offline[k][idx] for k in keys}

    # -------------------------------------------------------------------- eval
    def eval(self, seed) -> float:
        random_state = np.random.get_state()
        if self.cfg.mp_eval:
            scores = run_eval_mp(env_params=self.eval_env_params, agent=self.agent, num_proc=10,
                                 num_game=self.cfg.num_eval_episode, seed=seed, verbose=False)
        else:
            scores = run_eval(env_params=self.eval_env_params, agent=self.agent,
                              num_game=self.cfg.num_eval_episode, seed=seed,
                              record_dir=None, verbose=False)
        np.random.set_state(random_state)
        return float(np.mean(scores))

    # ----------------------------------------------------------------- warm-up
    def warm_up(self):
        """Collect `learning_starts` env steps of base policy + uniform residual
        noise (ResFiT use_base_policy_for_warmup=True)."""
        scale = self.cfg.warmup_noise_scale
        ascale = self.cfg.residual.action_scale
        obs, _ = self.train_env.reset()
        pbar = tqdm(total=self.cfg.learning_starts, desc="warm-up (base+noise)", dynamic_ncols=True)
        n_succ = 0
        while len(self.online) < self.cfg.learning_starts:
            s = obs["state"]
            with torch.no_grad():
                base_a = self.agent._base_action(s.view(1, -1).cuda().float())[0]
            noise = (torch.rand(self.agent.action_dim, device=base_a.device) * 2 - 1) * scale
            res = noise.clamp(-ascale, ascale)
            action = (base_a + res).clamp(-1.0, 1.0).cpu()
            next_obs, reward, terminal, success, image_obs = self.train_env.step(action)
            # done: bootstrap stops only on a genuine success terminal (end_on_success);
            # episode-length truncation keeps done=0 (timeout, keep bootstrapping).
            done = float(success)
            self.online.add(s, action, float(reward), next_obs["state"], done)
            obs = next_obs
            pbar.update(1)
            if terminal:
                n_succ += int(success)
                obs, _ = self.train_env.reset()
        pbar.close()
        print(f"warm-up done. online size={len(self.online)} (successes during warmup: {n_succ})")

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
            for _ in tqdm(range(self.cfg.critic_warmup_steps), desc="critic warmup", dynamic_ncols=True):
                batch = self._mixed_batch()
                self.agent.update(batch, update_actor=False)

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        pbar = tqdm(total=self.cfg.num_train_step, desc="residual-rl train",
                    dynamic_ncols=True, mininterval=1.0)
        recent: list[float] = []
        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self.agent.act(obs, eval_mode=False, stddev=stddev)
                stat["data/stddev"].append(stddev)
            s = obs["state"]
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action)
            self.online.add(s, action, float(reward), obs["state"], float(success))
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
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)
        pbar.close()

    def _mixed_batch(self):
        """RLPD symmetric mix: online (1-frac) + offline demo (frac)."""
        n_off = int(round(self.cfg.batch_size * self.cfg.offline_fraction))
        n_on = self.cfg.batch_size - n_off
        on = self.online.sample(n_on)
        if n_off <= 0:
            return on
        off = self._sample_offline(n_off)
        return {k: torch.cat([on[k], off[k]], dim=0) for k in on}

    def rl_train(self, stat):
        # critic-only warmup already ran in train() before the env loop, so the
        # actor trains from the first main-loop update.
        for i in range(self.cfg.num_critic_update):
            batch = self._mixed_batch()
            metrics = self.agent.update(batch, update_actor=True)
            stat.append(metrics)

    def log_and_save(self, stopwatch, stat, saver):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            eval_score = self.eval(seed=eval_seed)
            stat["score/score"].append(eval_score)
        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        stat.summary(self.global_step, reset=True)
        print(f"[eval @ step {self.global_step}] score={eval_score:.3f}  saved?={saved}")
        stopwatch.summary(reset=True)
        print(common_utils.get_mem_usage())


def load_model(weight_file, device):
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
