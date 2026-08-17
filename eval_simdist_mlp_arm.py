"""Eval the ORIGINAL SimDist arm (MLP policy proposal + MPPI) through the SAME
harness / env params / protocol as eval_flow_ug.py.

Why this exists: the "SimDist MPPI = 0.70" number in the flow-vs-SimDist
comparison came from train_simdist_adapt.py's IN-TRAINING eval at a particular
transition count, not from a standalone eval of the saved best ckpt. To fill the
2x2 (proposal in {MLP, flow} x search in {MPPI, UG}) with one ruler, this runs
the MLP arm on the SAME ckpt (`model0.pt`) with the SAME env/strict/sigma/games.

  python eval_simdist_mlp_arm.py \
      --agent_ckpt exps/mimicgen/threading_simdist_adapt_seed1/model0.pt \
      --num_game 500 --shift target
"""

import argparse
import os
import sys

import numpy as np
import pyrallis
import torch

from env.domain_shift import DomainShiftConfig
from env.robosuite_wrapper import PixelRobosuite
from env.strict_success import StrictSuccessConfig
from evaluate import run_eval, run_eval_mp
from rl.simdist_agent import SimDistAgent
from rl.simdist_planner import apply_original_recipe
from train_simdist import MainConfig

# same target recipe eval_flow_ug.py uses (CLAUDE.md §12 dynamics-only target)
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=(0.0, 0.0, 0.0),
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_ckpt", type=str, required=True,
                   help="SimDist agent ckpt (e.g. a train_simdist_adapt model0.pt)")
    p.add_argument("--stage1_cfg", type=str, default="",
                   help="stage-1 cfg.yaml; default = follow the adapt run's cfg.yaml")
    p.add_argument("--original_mppi", type=int, default=1)
    p.add_argument("--use_planner", type=int, default=1,
                   help="0 = policy-head only (the policy-only reference bar)")
    p.add_argument("--shift", type=str, default="target", choices=["target", "source"])
    p.add_argument("--num_game", type=int, default=500)
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--mp_eval", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_stage1_cfg(agent_ckpt: str, override: str) -> str:
    if override:
        return override
    run_cfg = os.path.join(os.path.dirname(agent_ckpt), "cfg.yaml")
    import yaml
    d = yaml.safe_load(open(run_cfg))
    if "stage1_ckpt" in d:  # an AdaptConfig -> follow it to the stage-1 run
        return os.path.join(os.path.dirname(d["stage1_ckpt"]), "cfg.yaml")
    return run_cfg  # already a stage-1 MainConfig


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage1_cfg_path = _resolve_stage1_cfg(args.agent_ckpt, args.stage1_cfg)
    cfg: MainConfig = pyrallis.load(MainConfig, open(stage1_cfg_path, "r"))  # type: ignore
    cfg.simdist.use_planner = args.use_planner
    if args.use_planner and args.original_mppi:
        p = apply_original_recipe(cfg.simdist.planner)
        nb = p.num_base_override if p.num_base_override >= 0 else int(p.num_samples * p.mixture_coef)
        print(f"[mppi] ORIGINAL recipe: base={nb}+rand={p.num_samples - nb}, "
              f"iters={p.iterations}, init_std={p.init_std}, temp={p.temperature}, "
              f"elites={p.num_elites}, base_policy_std={p.base_policy_std}, H={p.horizon}")

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

    agent = SimDistAgent(True, env.observation_shape, env.prop_shape, env.action_dim,
                         "", cfg.simdist, device=dev).to(dev)
    sd = torch.load(args.agent_ckpt, map_location=dev)
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    print(f"[agent] {args.agent_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, (missing, unexpected)
    agent.eval()

    arm = ("MPPI(orig recipe)" if args.use_planner else "policy-head only") + " + MLP proposal"
    print(f"[setup] arm={arm} | shift={args.shift} | games={args.num_game} | "
          f"hand_init_sigma={args.hand_init_sigma} | strict={args.strict_success}")

    runner = run_eval_mp if args.mp_eval else run_eval
    kw = dict(env_params=env_params, agent=agent, num_game=args.num_game,
              seed=args.seed, verbose=False)
    scores = runner(num_proc=args.num_proc, **kw) if args.mp_eval else runner(record_dir=None, **kw)

    sr = float(np.mean(scores))
    se = float(np.sqrt(sr * (1 - sr) / max(len(scores), 1)))
    print(f"[result] {arm}: strict_success = {sr:.3f} ± {se:.3f}  (n={len(scores)})")
    sys.stdout.flush()
    os._exit(0)  # EGL teardown hangs


if __name__ == "__main__":
    main()
