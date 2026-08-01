"""Eval a SOURCE-trained flow BC policy, steered by a frozen TARGET expert Q via
Universal Guidance, on the TARGET (or source) env (2026-07-06 experiment).

Q = the critic of threading_ibrl_target_snapshots_rlprior_seed1 (a TARGET IBRL
agent). Tests whether ∇_a Q steering lifts a source flow to ~100% on target, and
which source flow (all-data vs expert-only) transfers better.

Run:
  python eval_flow_qug.py \
      --flow exps/mimicgen/threading_flow_source_all_seed1/latest.pt \
      --q_ckpt exps/mimicgen/threading_ibrl_target_snapshots_rlprior_seed1/latest.pt \
      --use_ug 1 --steering_coeff 0.3 --num_game 300 --shift target
"""

import argparse
import os

import numpy as np
import pyrallis
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.flow_qug_agent import FlowQUGAgent, FlowQUGConfig
from eval_flow_ug import load_flow, TARGET_SHIFT


def load_q_agent(q_ckpt, env, device):
    from rl.q_agent import QAgent
    from train_rl import MainConfig
    cfg_path = os.path.join(os.path.dirname(q_ckpt), "cfg.yaml")
    src_cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    agent = QAgent(bool(src_cfg.use_state), env.observation_shape, env.prop_shape,
                   env.action_dim, src_cfg.rl_camera, src_cfg.q_agent)
    agent.load_state_dict(torch.load(q_ckpt, map_location=device))
    agent.cfg.act_method = "rl"
    agent.train(False)
    for p in agent.parameters():
        p.requires_grad_(False)
    agent = agent.to(device)
    print(f"[q-agent] loaded {q_ckpt} | use_state={src_cfg.use_state} "
          f"num_q={src_cfg.q_agent.num_q if hasattr(src_cfg.q_agent,'num_q') else '?'}")
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", type=str, required=True)
    p.add_argument("--q_ckpt", type=str,
                   default="exps/mimicgen/threading_ibrl_target_snapshots_rlprior_seed1/latest.pt")
    p.add_argument("--use_ug", type=int, default=1)
    p.add_argument("--steering_coeff", type=float, default=0.3)
    p.add_argument("--ug_num_steps", type=int, default=0)
    p.add_argument("--ta_eval", type=int, default=1)
    p.add_argument("--shift", type=str, default="target", choices=["target", "source"])
    p.add_argument("--num_game", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--mp_eval", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shift = DomainShiftConfig(mode="none") if args.shift == "source" \
        else DomainShiftConfig(**TARGET_SHIFT)
    env_params = dict(
        env_name="Threading_D0", robots=["Panda"], episode_length=400,
        reward_shaping=False, image_size=224, rl_image_size=96,
        camera_names=[], rl_cameras=[], use_state=True,
        obs_stack=1, state_stack=1, prop_stack=1,
        strict_success=StrictSuccessConfig(enabled=bool(args.strict_success),
                                           dwell_steps=3, max_tripod_disp=0.04),
        domain_shift=shift,
        hand_init_perturb_joint_sigma=(args.hand_init_sigma if args.hand_init_sigma > 0 else None),
    )
    env = PixelRobosuite(**env_params)  # type: ignore

    # auto-detect: flow_ug format (flow_cfg.json + flow.pt) vs train_bc ckpt (cfg.yaml)
    if os.path.exists(os.path.join(os.path.dirname(args.flow), "flow_cfg.json")):
        flow = load_flow(args.flow, dev)
    else:
        import train_bc
        flow, _, _ = train_bc.load_model(args.flow, dev, verbose=False)
        applied = flow.apply_ema() if hasattr(flow, "apply_ema") else False
        flow.eval()
        print(f"[flow] loaded from train_bc ckpt {args.flow} | EMA applied={applied} | "
              f"Tp={flow.Tp} Ta={flow.Ta}")
    if args.ta_eval > 0:
        assert args.ta_eval <= flow.Tp
        flow.Ta = args.ta_eval
    q_agent = load_q_agent(args.q_ckpt, env, dev) if args.use_ug else None

    ug_cfg = FlowQUGConfig(use_ug=args.use_ug, steering_coeff=args.steering_coeff,
                           ug_num_steps=args.ug_num_steps)
    agent = FlowQUGAgent(flow, q_agent, ug_cfg, device=dev)
    agent.eval()

    arm = f"Q-UG(coeff={args.steering_coeff})" if args.use_ug else "flow-only"
    print(f"[setup] arm={arm} | flow={os.path.basename(os.path.dirname(args.flow))} "
          f"| shift={args.shift} | games={args.num_game} | Ta={flow.Ta}", flush=True)

    if args.mp_eval:
        scores = run_eval_mp(env_params=env_params, agent=agent, num_proc=args.num_proc,
                             num_game=args.num_game, seed=args.seed, verbose=False)
    else:
        scores = run_eval(env_params=env_params, agent=agent, num_game=args.num_game,
                          seed=args.seed, record_dir=None, verbose=False)
    sr = float(np.mean(scores))
    se = float(np.std([float(s) for s in scores]) / max(len(scores) ** 0.5, 1))
    print(f"[result] {arm} flow={os.path.basename(os.path.dirname(args.flow))} "
          f"shift={args.shift}: strict_success = {sr:.3f} ± {se:.3f}  (n={len(scores)})",
          flush=True)

    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
