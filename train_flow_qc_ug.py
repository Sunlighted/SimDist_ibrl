"""Online model-free RL: flow-policy actor + Q-chunking critic + UG improvement.

SOURCE-only testbed (train_domain_shift=none, eval on source). The go/no-go gate:
RL reaches ~100% on source, so if this GPI-with-a-flow-actor can't climb toward
~100% here, the mechanism is broken (not the task). See docs/flow_qc_ug.md.

  python train_flow_qc_ug.py \
      --flow_ckpt exps/mimicgen/threading_flow_source_expert_seed1/best.pt \
      --save_dir exps/mimicgen/threading_flow_qc_ug_seed1
"""
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pyrallis
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

import common_utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig
from rl.sequence_replay import SequenceReplayBuffer
from rl.flow_qc_ug_agent import FlowQCUGAgent, FlowQCUGConfig


@dataclass
class MainConfig(common_utils.RunConfig):
    flow_ckpt: str = "exps/mimicgen/threading_flow_source_expert_seed1/best.pt"
    save_dir: str = "exps/mimicgen/threading_flow_qc_ug_seed1"
    seed: int = 1
    # env (SOURCE)
    task_name: str = "Threading_D0"
    episode_length: int = 400
    state_stack: int = 1
    hand_init_sigma: float = 0.1
    strict_dwell: int = 3
    strict_tripod_disp: float = 0.04
    end_on_success: int = 1
    # agent
    agent: FlowQCUGConfig = field(default_factory=FlowQCUGConfig)
    # training
    num_warm_up_episode: int = 30
    num_train_step: int = 300000
    update_freq: int = 1            # env steps between (num_critic_update crit + 1 actor)
    ug_collect_after: int = 5000    # collect flow-only until here, then flow+UG
    collect_stddev: float = 0.0     # small exploration noise on executed action
    seq_replay_max_episode: int = 6000
    # eval / save
    eval_every: int = 10000
    num_eval_episode: int = 100
    mp_eval: int = 1
    num_eval_proc: int = 10
    use_wb: int = 1

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)


def _load_flow(flow_ckpt, device):
    fcfg_json = json.load(open(os.path.join(os.path.dirname(flow_ckpt), "flow_cfg.json")))
    fcfg = StateFlowPolicyConfig(
        backbone=fcfg_json["backbone"], pred_horizon=fcfg_json["pred_horizon"],
        action_horizon=fcfg_json["action_horizon"],
        num_inference_steps=fcfg_json["num_inference_steps"])
    flow = StateFlowPolicy((fcfg_json["obs_dim"],), fcfg_json["action_dim"], fcfg).to(device)
    flow.load_state_dict(torch.load(flow_ckpt, map_location=device))
    # move EMA shadow to device then bake it into the online weights (best policy);
    # keep training the online weights + tracking EMA afterwards.
    for attr in ("_ema_state", "_ema_backup"):
        d = getattr(flow, attr, None)
        if isinstance(d, dict):
            for k in d:
                if torch.is_tensor(d[k]):
                    d[k] = d[k].to(device)
    if hasattr(flow, "apply_ema"):
        flow.apply_ema()
    print(f"[flow] loaded {flow_ckpt} | Tp={flow.Tp} Ta={flow.Ta} "
          f"obs={fcfg_json['obs_dim']} act={fcfg_json['action_dim']}")
    return flow, fcfg_json["obs_dim"], fcfg_json["action_dim"]


class Workspace:
    def __init__(self, cfg: MainConfig):
        self.cfg = cfg
        common_utils.set_all_seeds(cfg.seed)
        sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

        self._setup_env()
        flow, obs_dim, action_dim = _load_flow(cfg.flow_ckpt, "cuda")
        self.agent = FlowQCUGAgent(flow, obs_dim, action_dim, cfg.agent, "cuda").to("cuda")
        self.h = self.agent.h
        self.sample_h = self.agent.flow.Tp   # sample Tp-segments (critic uses first h)
        self.seq_replay = SequenceReplayBuffer(max_episodes=cfg.seq_replay_max_episode, device="cpu")
        self.global_step = 0

        self.stat = common_utils.MultiCounter(
            cfg.save_dir, bool(cfg.use_wb), wb_exp_name="mimicgen",
            wb_run_name=os.path.basename(cfg.save_dir), wb_group_name="flow_qc_ug",
            config={**cfg.__dict__, "agent": cfg.agent.__dict__})
        self.saver = common_utils.TopkSaver(save_dir=cfg.save_dir, topk=1)
        pyrallis.dump(cfg, open(os.path.join(cfg.save_dir, "cfg.yaml"), "w"))

    def _setup_env(self):
        cfg = self.cfg
        strict = StrictSuccessConfig(enabled=True, dwell_steps=cfg.strict_dwell,
                                     max_tripod_disp=cfg.strict_tripod_disp)
        common_env = dict(
            env_name=cfg.task_name, robots=["Panda"], episode_length=cfg.episode_length,
            reward_shaping=False, image_size=224, rl_image_size=96,
            camera_names=[], rl_cameras=[], use_state=True,
            obs_stack=1, state_stack=cfg.state_stack, prop_stack=1, strict_success=strict)
        self.train_env = PixelRobosuite(
            **common_env, env_reward_scale=1.0, end_on_success=bool(cfg.end_on_success),
            domain_shift=DomainShiftConfig(mode="none"),
            hand_init_perturb_joint_sigma=(cfg.hand_init_sigma if cfg.hand_init_sigma > 0 else None))
        self.eval_env_params = dict(
            **common_env, domain_shift=DomainShiftConfig(mode="none"),
            hand_init_perturb_joint_sigma=(cfg.hand_init_sigma if cfg.hand_init_sigma > 0 else None))
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    # ------------------------------------------------------------- collect
    def _collect_action(self, obs):
        use_ug = self.cfg.agent.use_ug and self.global_step >= self.cfg.ug_collect_after
        prev = self.agent.cfg.use_ug
        self.agent.cfg.use_ug = int(use_ug)
        a = self.agent.act(obs, eval_mode=False, stddev=self.cfg.collect_stddev, cpu=True)
        self.agent.cfg.use_ug = prev
        return a

    def warm_up(self):
        obs, _ = self.train_env.reset()
        self.seq_replay.new_episode(obs["state"])
        pbar = tqdm(total=self.cfg.num_warm_up_episode, desc="warm-up (source)")
        while self.seq_replay.size() < self.cfg.num_warm_up_episode:
            with torch.no_grad():
                a = self.agent.act(obs, eval_mode=False, stddev=self.cfg.collect_stddev, cpu=True)
            obs, reward, terminal, success, _ = self.train_env.step(a)
            self.seq_replay.add(obs["state"], a, reward, terminal, success)
            if terminal:
                pbar.n = min(self.seq_replay.size(), self.cfg.num_warm_up_episode)
                pbar.set_postfix(succ=self.seq_replay.num_success); pbar.refresh()
                obs, _ = self.train_env.reset()
                self.seq_replay.new_episode(obs["state"])
        pbar.close()

    # ------------------------------------------------------------- train
    def train(self):
        self.warm_up()
        obs, _ = self.train_env.reset()
        self.seq_replay.new_episode(obs["state"])
        pbar = tqdm(total=self.cfg.num_train_step, desc="train")
        while self.global_step < self.cfg.num_train_step:
            # collect one env step
            a = self._collect_action(obs)
            obs, reward, terminal, success, _ = self.train_env.step(a)
            self.seq_replay.add(obs["state"], a, reward, terminal, success)
            if terminal:
                obs, _ = self.train_env.reset()
                self.seq_replay.new_episode(obs["state"])
            self.global_step += 1
            pbar.update(1)

            # updates
            if self.global_step % self.cfg.update_freq == 0 and \
               self.seq_replay.can_sample(self.sample_h):
                self.agent.train(True)
                for _ in range(self.cfg.agent.num_critic_update):
                    s, act, rew = self.seq_replay.sample(256, self.sample_h, "cuda")
                    ci = self.agent.update_critic(s, act, rew)
                s, act, rew = self.seq_replay.sample(256, self.sample_h, "cuda")
                ai = self.agent.update_actor(s, act)
                self.agent.soft_update_target()
                self.agent.train(False)
                for k, v in {**ci, **ai}.items():
                    self.stat[f"train/{k}"].append(v)

            if self.global_step % self.cfg.eval_every == 0:
                self._eval_and_save()
        self._eval_and_save()

    def _eval_and_save(self):
        self.agent.train(False)
        random_state = np.random.get_state()
        if self.cfg.mp_eval:
            scores = run_eval_mp(env_params=self.eval_env_params, agent=self.agent,
                                 num_proc=self.cfg.num_eval_proc,
                                 num_game=self.cfg.num_eval_episode, seed=1, verbose=False)
        else:
            scores = run_eval(env_params=self.eval_env_params, agent=self.agent,
                              num_game=self.cfg.num_eval_episode, seed=1, verbose=False)
        np.random.set_state(random_state)
        score = float(np.mean(scores))
        self.stat["score/source"].append(score)
        self.stat["data/buffer_ep"].append(self.seq_replay.size())
        self.stat["data/buffer_succ"].append(self.seq_replay.num_success)
        print(f"[step {self.global_step}] source strict-success = {score:.3f} "
              f"(buffer {self.seq_replay.size()} ep / {self.seq_replay.num_success} succ)")
        self.stat.summary(self.global_step)
        self.saver.save(self.agent.state_dict(), score, save_latest=True)


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    ws = Workspace(cfg)
    ws.train()
    if cfg.use_wb:
        import wandb
        wandb.finish()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    from rich.traceback import install
    install()
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    main()
