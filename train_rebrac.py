"""ReBRAC (offline TD3+BC) on the Threading_D0 target demos — head-to-head vs MLP-BC.

ReBRAC (Tarasov et al., NeurIPS 2023; ref at /home/Chenyang/ReBRAC) is a minimalist
offline-RL recipe on top of TD3+BC. Here it trains PURELY OFFLINE on the same 20
TARGET demos the MLP-BC baseline uses (threading_d0_target_dynamics, dynamics-only
target §12), with the SAME actor network as MLP-BC (StateBcPolicy) — so the only
difference vs `threading_d0_state_mlp_bc_target` is the objective (TD3+BC vs pure MSE).
No source agent, no online interaction (offline RL); eval is on the TARGET env, with
a nominal-env SOURCE eval as a diagnostic.

Run:
  python train_rebrac.py --config_path release/cfgs/mimicgen/threading_d0_state_rebrac.yaml \
      --save_dir exps/mimicgen/threading_rebrac_seed1
  # apples-to-apples raw-state (no z-score) ablation matching MLP-BC's input:
  python train_rebrac.py ... --rebrac.normalize_states 0
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import h5py
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
from rl.rebrac_agent import ReBRACAgent, ReBRACConfig


def _load_demos_flat(data_path, num_data, reward_scale, weight_lut=None, default_weight=1.0,
                     include_failures=False):
    """Flatten demo episodes into ReBRAC's (s,a,r,s',a',done,weight) arrays.

    Mirrors `rl/replay.py::add_demos_to_replay` parsing: state re-concatenated
    from STATE_KEYS[task] (so the layout matches the env's obs), each demo
    truncated at its first success step, no-success demos skipped. `next_action[t]
    = a_{t+1}` (terminal = dummy, masked by done=1). The ORIGINAL demo index
    (demo_0,1,...) is the key into `weight_lut` (a {(ep_id, t): weight} map from
    the GT-gap cache, whose ep_id is the same demo order); demos with no entry
    fall back to `default_weight` (target demos pass weight_lut=None => all 1.0).

    `include_failures=True` (used for the source recovery datasets ds2/ds3, §19):
    keep no-success episodes too — as a FULL-length trajectory with done=0
    throughout (a timeout truncation, NOT an absorbing terminal, so the critic
    keeps bootstrapping). Without this, offline RL would silently drop exactly
    the failure data it is meant to learn from, and the transition budget would
    no longer match the weighted-BC runs.
    """
    f = h5py.File(data_path, "r")
    cfg_path = os.path.join(os.path.dirname(data_path), "env_cfg.json")
    task_name = json.load(open(cfg_path, "r"))["env_name"]
    state_keys = STATE_KEYS[task_name]
    num_episode = len(list(f["data"].keys()))
    print(f"[rebrac-flat] loading first {num_data} of {num_episode} demos from {data_path}")

    S, A, R, SN, AN, D, W = [], [], [], [], [], [], []
    loaded = 0
    for ep_id in range(num_episode):
        if num_data > 0 and ep_id >= num_data:
            break
        episode = f[f"data/demo_{ep_id}"]
        actions = np.clip(np.array(episode["actions"]).astype(np.float32), -1.0, 1.0)
        rewards = np.array(episode["rewards"]).astype(np.float32)
        has_success = bool((rewards == 1).any())
        if not has_success and not include_failures:
            continue
        states = np.concatenate([np.array(episode["obs"][k]) for k in state_keys],
                                axis=1).astype(np.float32)
        n_obs = states.shape[0]
        if has_success:
            T = int(np.argmax(rewards == 1)) + 1  # truncate at first success
        else:
            T = min(len(actions), n_obs)  # full failure episode (timeout, no terminal)
        for t in range(T):
            s_next = states[t + 1] if (t + 1) < n_obs else states[t]
            a_next = actions[t + 1] if (t + 1) < T else actions[t]
            S.append(states[t]); A.append(actions[t]); R.append(rewards[t] * reward_scale)
            SN.append(s_next); AN.append(a_next)
            # done=1 ONLY at a real success terminal; failure timeouts stay done=0.
            D.append(1.0 if (has_success and t == T - 1) else 0.0)
            W.append(weight_lut.get((ep_id, t), default_weight) if weight_lut else default_weight)
        loaded += 1
    f.close()
    out = {
        "states": torch.from_numpy(np.asarray(S, np.float32)),
        "actions": torch.from_numpy(np.asarray(A, np.float32)),
        "rewards": torch.tensor(np.asarray(R, np.float32)),
        "next_states": torch.from_numpy(np.asarray(SN, np.float32)),
        "next_actions": torch.from_numpy(np.asarray(AN, np.float32)),
        "dones": torch.tensor(np.asarray(D, np.float32)),
        "weights": torch.tensor(np.asarray(W, np.float32)),
    }
    print(f"[rebrac-flat] loaded {loaded} demos, {out['states'].shape[0]} transitions, "
          f"#success-terminal={int(out['dones'].sum())}, "
          f"mean weight={out['weights'].mean():.3f}")
    return out


def _gap_to_weight(gap, mode, tau, threshold, floor):
    if mode == "binary":
        w = (gap < threshold).astype(np.float32)
    else:
        w = np.exp(-gap / max(tau, 1e-8)).astype(np.float32)
    if floor > 0:
        w = np.maximum(w, floor)
    return w


class ReBRACOfflineDataset:
    """Flat offline buffer. Two modes:
      * target-only (default): the 20 TARGET demos, unweighted (weights omitted
        from the batch => plain ReBRAC, byte-for-byte the no-weighting path).
      * weighted SOURCE + TARGET mix (cfg.source_demo_path set): a LOT of
        gap-weighted source + a few target (weight 1), drawn at a fixed per-batch
        ratio — the offline-RL analog of train_weighted_bc.py (CLAUDE.md §19).
    """

    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.weighted = bool(cfg.source_demo_path)

        # TARGET demos (weight 1)
        self.tgt = {k: v.to(device) for k, v in _load_demos_flat(
            cfg.target_demo_path, cfg.target_num_data, cfg.env_reward_scale).items()}
        self.n_tgt = self.tgt["states"].shape[0]

        if not self.weighted:
            self.src = None
            self._all_states = self.tgt["states"]
            print(f"[rebrac-buffer] TARGET-only: {self.n_tgt} transitions, "
                  f"obs={self.tgt['states'].shape[1]}, act={self.tgt['actions'].shape[1]}")
            return

        # SOURCE demos (gap-weighted) — key the GT-gap cache by (ep_id, t).
        assert os.path.exists(cfg.gap_cache), (
            f"gap cache not found: {cfg.gap_cache} — run scripts/compute_source_gaps.py")
        d = np.load(cfg.gap_cache)
        gw = _gap_to_weight(d["gap"], cfg.gap_c_mode, cfg.gap_tau,
                            cfg.gap_threshold, cfg.weight_floor)
        lut = {(int(e), int(t)): float(w) for e, t, w in zip(d["ep_id"], d["t_in_ep"], gw)}
        self.src = {k: v.to(device) for k, v in _load_demos_flat(
            cfg.source_demo_path, cfg.source_num_data, cfg.env_reward_scale,
            weight_lut=lut, include_failures=True).items()}
        self.n_src = self.src["states"].shape[0]
        self.ratio = cfg.target_batch_ratio
        self._all_states = torch.cat([self.src["states"], self.tgt["states"]], dim=0)
        print(f"[rebrac-buffer] WEIGHTED mix: src {self.n_src} (gap-weighted, "
              f"mode={cfg.gap_c_mode} tau={cfg.gap_tau}) + tgt {self.n_tgt} (w=1); "
              f"per-batch ratio {self.ratio:.2f} tgt / {1 - self.ratio:.2f} src")

    def state_stats(self):
        return self._all_states.mean(0), self._all_states.std(0)

    @staticmethod
    def _draw(pool, n, keys):
        idx = torch.randint(0, pool["states"].shape[0], (n,), device=pool["states"].device)
        return {k: pool[k][idx] for k in keys}

    def sample(self, batch_size):
        keys = ("states", "actions", "rewards", "next_states", "next_actions", "dones")
        if not self.weighted:
            # plain ReBRAC: no `weights` key => agent uses the unweighted path.
            return self._draw(self.tgt, batch_size, keys)
        n_tgt = max(0, min(batch_size, int(round(batch_size * self.ratio))))
        n_src = batch_size - n_tgt
        keys_w = keys + ("weights",)
        parts = []
        if n_src > 0:
            parts.append(self._draw(self.src, n_src, keys_w))
        if n_tgt > 0:
            parts.append(self._draw(self.tgt, n_tgt, keys_w))
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in keys_w}


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env (eval only — offline RL never interacts during training)
    task_name: str = "Threading_D0"
    episode_length: int = 400
    end_on_success: int = 1
    image_size: int = 224
    rl_image_size: int = 96
    rl_camera: str = "robot0_eye_in_hand"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1
    use_state: int = 1
    # agent
    rebrac: ReBRACConfig = field(default_factory=ReBRACConfig)
    env_reward_scale: float = 1.0
    batch_size: int = 1024
    # offline target demos
    target_demo_path: str = "release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5"
    target_num_data: int = 20
    # weighted SOURCE+TARGET mix (offline-RL analog of train_weighted_bc; §19).
    # Leave source_demo_path empty for plain target-only ReBRAC. When set, source
    # transitions are gap-weighted (target=1) and the BC terms are weighted by it
    # (critic TD too if rebrac.weight_td=1). Mirrors threading_d0_state_weighted_bc.
    source_demo_path: str = ""
    source_num_data: int = 2000
    gap_cache: str = ""                # gt_gaps_n*.npz from scripts/compute_source_gaps.py
    gap_c_mode: str = "soft"           # "soft": c=exp(-gap/tau); "binary": c=1[gap<thr]
    gap_tau: float = 0.01
    gap_threshold: float = 0.013
    weight_floor: float = 0.0
    target_batch_ratio: float = 0.5    # per-batch fraction drawn from TARGET demos
    # schedule
    num_train_step: int = 200000
    log_per_step: int = 10000
    num_eval_episode: int = 50
    eval_source: int = 1
    mp_eval: int = 1
    # log
    save_dir: str = "exps/rebrac/run1"
    use_wb: int = 0
    # TARGET (eval) domain shift + SOURCE (diagnostic) is mode=none
    eval_domain_shift: DomainShiftConfig = field(default_factory=DomainShiftConfig)
    strict_success: StrictSuccessConfig = field(default_factory=StrictSuccessConfig)
    eval_hand_init_perturb_joint_sigma: Optional[float] = None

    def __post_init__(self):
        self.rl_cameras = self.rl_camera.split("+")
        assert self.use_state, "ReBRAC baseline is state-only"
        if self.source_demo_path in ["none", "None"]:
            self.source_demo_path = ""
        self.robots: list[str] = ["Panda"]


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
        self._setup_env()

        self.agent = ReBRACAgent(
            bool(self.cfg.use_state),
            self.eval_env.observation_shape,
            self.eval_env.prop_shape,
            self.eval_env.action_dim,
            self.cfg.rl_camera,
            cfg.rebrac,
            device="cuda",
        )

        if not from_main:
            return
        self._setup_data()

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
        # TARGET (eval / deployment)
        self.eval_env_params = dict(
            **common_env,
            domain_shift=self.cfg.eval_domain_shift,
            hand_init_perturb_joint_sigma=self.cfg.eval_hand_init_perturb_joint_sigma,
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _setup_data(self):
        self.dataset = ReBRACOfflineDataset(self.cfg, device="cuda")
        mean, std = self.dataset.state_stats()
        self.agent.set_normalization(mean, std)

    def _source_eval_params(self):
        params = dict(self.eval_env_params)
        params["domain_shift"] = DomainShiftConfig(mode="none")
        return params

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

    def train(self):
        self.stat = common_utils.MultiCounter(
            self.work_dir, bool(self.cfg.use_wb), wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run, wb_group_name=self.cfg.wb_group, config=self.cfg_dict,
        )
        self.saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)
        stopwatch = common_utils.Stopwatch()
        pbar = tqdm(total=self.cfg.num_train_step, desc="rebrac offline",
                    dynamic_ncols=True, mininterval=1.0)
        while self.global_step < self.cfg.num_train_step:
            with stopwatch.time("train"):
                batch = self.dataset.sample(self.cfg.batch_size)
                self.stat.append(self.agent.update(batch))
                self.global_step += 1
                pbar.update(1)
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch)
        pbar.close()
        self.log_and_save(stopwatch)

    def log_and_save(self, stopwatch):
        elapsed = max(stopwatch.elapsed_time_since_reset, 1e-6)
        self.stat["other/speed"].append(self.cfg.log_per_step / elapsed)
        self.stat["other/step"].append(self.global_step)
        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            eval_score = self.eval(seed=eval_seed)  # TARGET (deployment metric)
            self.stat["score/score"].append(eval_score)
            source_score = None
            if self.cfg.eval_source:
                source_score = self.eval(seed=eval_seed, env_params=self._source_eval_params())
                self.stat["score/source_score"].append(source_score)
        saved = self.saver.save(self.agent.state_dict(), eval_score, save_latest=True)
        self.stat.summary(self.global_step, reset=True)
        src_str = f"  source_score={source_score:.3f}" if source_score is not None else ""
        print(f"[eval @ step {self.global_step}] target_score={eval_score:.3f}{src_str}  saved?={saved}")
        stopwatch.summary(reset=True)
        print(common_utils.get_mem_usage())


def load_model(weight_file, device):
    """Rebuild a ReBRACAgent for standalone eval (evaluate/eval.py --mode rebrac)."""
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
