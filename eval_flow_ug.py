"""Eval the flow BC policy (± Universal-Guidance) on TARGET, for the SimDist
policy-replacement experiment (CLAUDE.md §24 follow-up).

Arms (share ONE flow policy from train_flow_ug.py):
  (A)  --wm_ckpt ""                                    flow policy alone
  (B)  --wm_ckpt .../threading_simdist_diversev2_h5_seed1/model0.pt   flow + UG (source WM)
  (B1) --wm_ckpt .../threading_simdist_adapt_seed1/model0.pt          flow + UG (target-adapted WM)

Reference bar: the SimDist MLP policy-only (use_planner=0, target-FT'd) ≈ 42%.

Run:
  python eval_flow_ug.py --flow exps/mimicgen/threading_flow_ug_seed1/flow.pt \
      --wm_ckpt exps/mimicgen/threading_simdist_diversev2_h5_seed1/model0.pt \
      --use_ug 1 --steering_coeff 0.5 --num_game 500 --shift target
"""

import argparse
import json
import os

import numpy as np
import pyrallis
import torch

from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from env.domain_shift import DomainShiftConfig
from env.strict_success import StrictSuccessConfig
from rl.world_model import WorldModel
from rl.flow_ug_agent import FlowUGAgent, FlowUGConfig, OrigMppiFlowAgent
from rl.simdist_planner import PlannerConfig
from bc.flow_policy import StateFlowPolicy, StateFlowPolicyConfig

# §12 dynamics-only TARGET.
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=[0.0, 0.0, 0.0],
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", type=str, required=True, help="flow.pt from train_flow_ug.py")
    p.add_argument("--wm_ckpt", type=str, default="",
                   help="SimDist ckpt for the UG world model; '' = flow-only (arm A)")
    p.add_argument("--ta_eval", type=int, default=0,
                   help="override the flow's executed chunk length Ta at eval "
                        "(0 = keep trained Ta; 1 = re-query every step, = MLP baseline)")
    p.add_argument("--use_ug", type=int, default=0)
    p.add_argument("--steering_coeff", type=float, default=0.5)
    p.add_argument("--ug_num_steps", type=int, default=0, help="0 = flow's num_inference_steps")
    p.add_argument("--ug_rollout_H", type=int, default=0, help="0 = world model horizon")
    p.add_argument("--use_reward", type=int, default=1)
    p.add_argument("--stop_grad_velocity", type=int, default=1)
    # MPPI selection layer over flow(+UG) proposals
    p.add_argument("--mppi", type=int, default=0)
    p.add_argument("--mppi_candidates", type=int, default=16)
    p.add_argument("--mppi_random", type=int, default=0)
    p.add_argument("--mppi_random_std", type=float, default=0.1)
    p.add_argument("--mppi_select", type=str, default="best", choices=["best", "softmax"])
    p.add_argument("--mppi_elites", type=int, default=4)
    p.add_argument("--mppi_temperature", type=float, default=10.0)
    # ORIGINAL SimDist MppiPlanner (Gaussian random + CEM iterations) with flow base
    p.add_argument("--orig_mppi", type=int, default=0)
    p.add_argument("--omppi_num_base", type=int, default=100, help="# flow-chunk base candidates")
    p.add_argument("--omppi_num_rand", type=int, default=250, help="# Gaussian-random candidates (reduce for 少random)")
    p.add_argument("--omppi_iterations", type=int, default=3)
    p.add_argument("--omppi_init_std", type=float, default=1.0)
    p.add_argument("--omppi_min_std", type=float, default=0.05)
    p.add_argument("--omppi_temperature", type=float, default=0.4)
    p.add_argument("--omppi_base_policy_std", type=float, default=0.1)
    p.add_argument("--omppi_num_elites", type=int, default=64)
    p.add_argument("--omppi_use_reward", type=int, default=1)
    p.add_argument("--shift", type=str, default="target", choices=["target", "source"])
    p.add_argument("--num_game", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--mp_eval", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_flow(flow_ckpt, device):
    fcfg_json = json.load(open(os.path.join(os.path.dirname(flow_ckpt), "flow_cfg.json")))
    fcfg = StateFlowPolicyConfig(
        backbone=fcfg_json["backbone"],
        pred_horizon=fcfg_json["pred_horizon"],
        action_horizon=fcfg_json["action_horizon"],
        num_inference_steps=fcfg_json["num_inference_steps"],
    )
    flow = StateFlowPolicy((fcfg_json["obs_dim"],), fcfg_json["action_dim"], fcfg).to(device)
    flow.load_state_dict(torch.load(flow_ckpt, map_location=device))
    flow.eval()
    applied = flow.apply_ema()   # use EMA weights for eval (matches training)
    print(f"[flow] loaded {flow_ckpt} | EMA applied={applied} | "
          f"Tp={flow.Tp} Ta={flow.Ta} inf_steps={fcfg.num_inference_steps}")
    return flow


def _find_world_model_cfg(cfg_path):
    """Return the simdist.world_model dict, following stage1_ckpt when the
    sibling cfg is an AdaptConfig (which has no simdist block of its own)."""
    import yaml as _yaml
    d = _yaml.safe_load(open(cfg_path, "r"))
    if isinstance(d, dict) and "simdist" in d:
        return d["simdist"]["world_model"]
    if isinstance(d, dict) and d.get("stage1_ckpt"):
        stage1_cfg = os.path.join(os.path.dirname(d["stage1_ckpt"]), "cfg.yaml")
        return _find_world_model_cfg(stage1_cfg)
    raise ValueError(f"could not locate simdist.world_model in {cfg_path}")


def load_world_model(wm_ckpt, obs_dim, action_dim, device):
    from dataclasses import fields
    from rl.world_model import WorldModelConfig
    cfg_path = os.path.join(os.path.dirname(wm_ckpt), "cfg.yaml")
    wm_dict = _find_world_model_cfg(cfg_path)
    valid = {f.name for f in fields(WorldModelConfig)}
    wmcfg = WorldModelConfig(**{k: v for k, v in wm_dict.items() if k in valid})
    model = WorldModel(obs_dim, action_dim, wmcfg).to(device)
    full_sd = torch.load(wm_ckpt, map_location=device)
    model_sd = {k[len("model."):]: v for k, v in full_sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"[world-model] loaded {wm_ckpt} | horizon={wmcfg.horizon} "
          f"discrete_heads={wmcfg.discrete_heads} | missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    return model


def main():
    args = parse_args()
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
    obs_dim = int(env.observation_shape[0])
    action_dim = int(env.action_dim)

    flow = load_flow(args.flow, dev)
    if args.ta_eval > 0:
        assert args.ta_eval <= flow.Tp, f"ta_eval {args.ta_eval} > Tp {flow.Tp}"
        flow.Ta = args.ta_eval
        print(f"[flow] eval Ta overridden -> {flow.Ta} (re-query every {flow.Ta} steps)")
    model = None
    if args.use_ug or args.mppi or args.orig_mppi:
        assert args.wm_ckpt, "use_ug/mppi/orig_mppi require --wm_ckpt"
        model = load_world_model(args.wm_ckpt, obs_dim, action_dim, dev)

    if args.orig_mppi:
        pcfg = PlannerConfig(
            horizon=model.cfg.horizon, discount=0.99,
            num_samples=args.omppi_num_base + args.omppi_num_rand,
            num_base_override=args.omppi_num_base,
            num_elites=args.omppi_num_elites, iterations=args.omppi_iterations,
            temperature=args.omppi_temperature, momentum=0.0,
            init_std=args.omppi_init_std, min_std=args.omppi_min_std,
            base_policy_std=args.omppi_base_policy_std,
            use_reward=args.omppi_use_reward, warm_start=0,
        )
        agent = OrigMppiFlowAgent(flow, model, pcfg, device=dev)
        agent.eval()
    else:
        ug_cfg = FlowUGConfig(
            use_ug=args.use_ug, steering_coeff=args.steering_coeff,
            ug_num_steps=args.ug_num_steps, ug_rollout_H=args.ug_rollout_H,
            use_reward=args.use_reward, stop_grad_velocity=args.stop_grad_velocity,
            discount=0.99,
            mppi_enabled=args.mppi, mppi_candidates=args.mppi_candidates,
            mppi_random=args.mppi_random, mppi_random_std=args.mppi_random_std,
            mppi_select=args.mppi_select, mppi_elites=args.mppi_elites,
            mppi_temperature=args.mppi_temperature,
        )
        agent = FlowUGAgent(flow, model, ug_cfg, device=dev)
        agent.eval()

    wmn = os.path.basename(os.path.dirname(args.wm_ckpt)) if args.wm_ckpt else "none"
    parts = []
    if args.orig_mppi:
        parts.append(f"OrigMPPI(base={args.omppi_num_base},rand={args.omppi_num_rand},"
                     f"iters={args.omppi_iterations})")
    if args.use_ug:
        parts.append(f"UG(coeff={args.steering_coeff})")
    if args.mppi:
        parts.append(f"MPPI(K={args.mppi_candidates},rand={args.mppi_random},"
                     f"sel={args.mppi_select})")
    arm = "A (flow-only)" if not parts else f"{'+'.join(parts)} WM={wmn}"
    print(f"[setup] arm={arm} | shift={args.shift} | games={args.num_game} | "
          f"hand_init_sigma={args.hand_init_sigma} | strict={args.strict_success}")

    if args.mp_eval:
        scores = run_eval_mp(env_params=env_params, agent=agent, num_proc=args.num_proc,
                             num_game=args.num_game, seed=args.seed, verbose=False)
    else:
        scores = run_eval(env_params=env_params, agent=agent, num_game=args.num_game,
                          seed=args.seed, record_dir=None, verbose=False)
    sr = float(np.mean(scores))
    se = float(np.std([float(s) for s in scores]) / max(len(scores) ** 0.5, 1))
    print(f"[result] {arm}: strict_success = {sr:.3f} ± {se:.3f}  "
          f"(n={len(scores)})   [MLP policy-only reference ≈ 0.42]")

    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
