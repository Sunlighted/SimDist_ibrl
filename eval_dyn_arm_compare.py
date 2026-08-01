"""RAW vs LATENT dynamics for MPPI planning — the clean single-variable comparison.

Motivation (CLAUDE.md §27 + the 2026-07-17 session). Every prior raw-vs-latent
target number was confounded: the raw arm (train_dynamics_adapt.py) borrowed a
SOURCE value + a freshly-BC'd policy head, while the latent arm (§24 SimDist
adapt) had a fully distilled value + policy. So "raw target 0.62 < latent 0.85"
compared planning-source maturity, not the representation. §27 also showed the
planner is a *proposal improver* whose score tracks the proposal, so an unequal
proposal invalidates the comparison outright.

Here BOTH arms share, bit-for-bit:
  * the SAME value head   (the adapted latent SimDist's),
  * the SAME policy head  (ditto) as the MPPI proposal,
  * the SAME planner params, env, seeds, and episode count.
The ONLY difference is which dynamics rolls the candidates:
  * arm=latent : the latent SimDist's own `next` in SimNorm latent space,
  * arm=raw    : the raw-state WorldModel from train_dynamics{,_adapt}.py.

Fairness detail that makes this legitimate (both arms at their best, neither
paying a decode tax):
  * the RAW arm goes through the planner's EXTERNAL hooks, which are called on
    `decode(z)` — but in the raw arm `decode` is the EXACT denormalisation
    inverse, so there is no decode error. Terminal scoring is
    `latent_value(latent_encode(s_H))`, i.e. the shared head reading the raw
    arm's own terminal state.
  * the LATENT arm runs NATIVELY (value_fn=None -> `predict_value(z_H)`;
    proposal=None + shared_encoder -> `policy_from_latent(z)`), so it never
    touches `decode` (whose 53mm error would otherwise unfairly sink it).
  Both therefore feed the SAME value head the best terminal latent each arm can
  produce; the comparison isolates the rollout dynamics.

`policy_only` (the shared policy head alone, planner off) is the bar BOTH arms
must clear — per §27 the planner's value is `score - proposal`, not `score`.

Run:
  python eval_dyn_arm_compare.py --arm latent --num_game 300 --shift target
  python eval_dyn_arm_compare.py --arm raw    --num_game 300 --shift target \
      --dyn_ckpt exps/mimicgen/threading_dyn_adapt_goodpolicy_seed1/model0.pt
"""
import argparse
import dataclasses
import json
import os

import numpy as np
import torch
import yaml

from env.domain_shift import DomainShiftConfig
from env.robosuite_wrapper import PixelRobosuite
from env.strict_success import StrictSuccessConfig
from evaluate import run_eval
from rl.simdist_agent import SimDistAgent, SimDistConfig
from rl.simdist_planner import PlannerConfig
from rl.world_model import WorldModel, WorldModelConfig

# §12 dynamics-only TARGET
TARGET_SHIFT = dict(
    mode="fixed", friction_scale=0.5, mass_scale=3.0, solref_scale=1.0,
    solimp_scale=1.0, gripper_kp_scale=0.12, hole_offset=[0.0, 0.0, 0.0],
    ring_size_scale=0.85, ring_friction_scale=2.0, mask_hole_offset_in_obs=False,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["raw", "latent"], required=True)
    # the LATENT adapted SimDist: supplies encoder + value + policy for BOTH arms,
    # and the dynamics for arm=latent.
    p.add_argument("--latent_agent", type=str,
                   default="exps/mimicgen/threading_simdist_adapt_30k_seed1/model0.pt")
    p.add_argument("--latent_stage1_cfg", type=str, default="",
                   help="stage-1 cfg.yaml giving the latent WorldModel arch "
                        "(default: read the adapt run's cfg.yaml -> stage1_ckpt)")
    # the RAW dynamics (arm=raw only); its sibling dyn_cfg.json gives the arch
    p.add_argument("--dyn_ckpt", type=str,
                   default="exps/mimicgen/threading_dyn_adapt_goodpolicy_seed1/model0.pt")
    p.add_argument("--dyn_cfg", type=str,
                   default="exps/mimicgen/threading_dyn_diverse_v2_seed1/dyn_cfg.json")
    p.add_argument("--num_game", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shift", type=str, default="target", choices=["source", "target"])
    p.add_argument("--hand_init_sigma", type=float, default=0.1)
    p.add_argument("--strict_success", type=int, default=1)
    p.add_argument("--skip_policy_only", type=int, default=0)
    # planner (shared by both arms)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--num_elites", type=int, default=64)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--mixture_coef", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--init_std", type=float, default=0.15)
    p.add_argument("--use_wb", type=int, default=0)
    p.add_argument("--wb_project", type=str, default="mimicgen")
    p.add_argument("--wb_group", type=str, default="dyn_arm_compare")
    p.add_argument("--wb_run", type=str, default="",
                   help="wandb run name (default: 'dynarm_<arm>_<shift>_seed<seed>')")
    return p.parse_args()


def _wm_cfg_from_yaml(path):
    wm = yaml.safe_load(open(path))["simdist"]["world_model"]
    fields = {f.name for f in dataclasses.fields(WorldModelConfig)}
    return WorldModelConfig(**{k: v for k, v in wm.items() if k in fields})


def main():
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- resolve the latent stage-1 arch ----
    stage1_cfg = args.latent_stage1_cfg
    if not stage1_cfg:
        adapt_cfg = yaml.safe_load(
            open(os.path.join(os.path.dirname(args.latent_agent), "cfg.yaml")))
        stage1_cfg = os.path.join(os.path.dirname(adapt_cfg["stage1_ckpt"]), "cfg.yaml")
    lat_wm_cfg = _wm_cfg_from_yaml(stage1_cfg)
    print(f"[setup] latent arch from {stage1_cfg} "
          f"(encoder={lat_wm_cfg.encoder_type}, H={lat_wm_cfg.horizon}, "
          f"shared_encoder={getattr(lat_wm_cfg, 'shared_encoder', 'n/a')})")
    assert lat_wm_cfg.encoder_type == "latent", "latent_agent must be a latent arm"

    # ---- env ----
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

    def planner_cfg(H):
        return PlannerConfig(
            horizon=H, num_samples=args.num_samples, num_elites=args.num_elites,
            iterations=args.iterations, mixture_coef=args.mixture_coef,
            temperature=args.temperature, init_std=args.init_std,
            use_reward=False, discount=0.99)

    # ---- the LATENT agent: encoder + value + policy for BOTH arms ----
    lat_agent = SimDistAgent(
        True, env.observation_shape, env.prop_shape, env.action_dim,
        "robot0_eye_in_hand",
        SimDistConfig(world_model=lat_wm_cfg, planner=planner_cfg(lat_wm_cfg.horizon),
                      use_planner=1),
        device=dev)
    sd = torch.load(args.latent_agent, map_location=dev, weights_only=False)
    missing, unexpected = lat_agent.load_state_dict(sd, strict=False)
    print(f"[setup] latent agent {args.latent_agent} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    lat_agent.eval()
    for p_ in lat_agent.parameters():
        p_.requires_grad_(False)
    lat_wm: WorldModel = lat_agent.model

    # ---- the arm under test ----
    if args.arm == "latent":
        # NATIVE: value_fn/proposal left as None => predict_value(z_H) and
        # policy_from_latent(z) (shared_encoder) — no decode anywhere.
        agent = lat_agent
        H = lat_wm_cfg.horizon
        print("[arm] latent: native latent rollout + native value head + native policy")
    else:
        dcfg = json.load(open(args.dyn_cfg))
        H = dcfg["horizon"]
        raw_cfg = SimDistConfig(
            world_model=WorldModelConfig(
                encoder_type="identity", mlp_hidden_dim=dcfg["hidden_dim"],
                num_layer=dcfg["num_layer"], layer_norm=dcfg["layer_norm"],
                horizon=H, activation=dcfg.get("activation", "relu")),
            planner=planner_cfg(H), use_planner=1)
        agent = SimDistAgent(True, env.observation_shape, env.prop_shape,
                             env.action_dim, "robot0_eye_in_hand", raw_cfg, device=dev)
        rsd = torch.load(args.dyn_ckpt, map_location=dev, weights_only=False)
        if any(k.startswith("model.") for k in rsd):     # SimDistAgent-style ckpt
            rsd = {k[len("model."):]: v for k, v in rsd.items() if k.startswith("model.")}
        m_, u_ = agent.model.load_state_dict(rsd, strict=False)
        assert agent.model.raw_state, "arm=raw needs an identity-encoder world model"
        agent.eval()
        print(f"[arm] raw: {args.dyn_ckpt} (missing={len(m_)} unexpected={len(u_)}); "
              f"decode is exact => external hooks are lossless")

        @torch.no_grad()
        def value_fn(state):        # [N,obs] -> [N]; the SHARED latent value head
            return lat_wm.predict_value(lat_wm.encode(state))

        @torch.no_grad()
        def proposal_policy(state):  # [N,obs] -> [N,act]; the SHARED policy head
            return lat_wm.policy_action(state)

        agent.set_planning_sources(value_fn=value_fn, reward_fn=None,
                                   proposal_policy=proposal_policy)

    assert H == lat_wm_cfg.horizon, (
        f"horizon mismatch: arm H={H} vs latent H={lat_wm_cfg.horizon}; the two arms "
        f"must plan over the same horizon to be comparable")

    print(f"[setup] shift={args.shift} sigma={args.hand_init_sigma} H={H} "
          f"games={args.num_game} seed={args.seed} | planner: {args.num_samples} "
          f"samples x {args.iterations} iters, mixture={args.mixture_coef}")

    # ---- the bar: the SHARED policy head alone (same for both arms) ----
    if not args.skip_policy_only:
        prev = lat_agent.cfg.use_planner
        lat_agent.cfg.use_planner = 0
        pol = run_eval(env_params=env_params, agent=lat_agent, num_game=args.num_game,
                       seed=args.seed, record_dir=None, verbose=False)
        lat_agent.cfg.use_planner = prev
        print(f"[bar]     policy_only (shared head): {np.mean(pol):.3f}")
    else:
        pol = None

    scores = run_eval(env_params=env_params, agent=agent, num_game=args.num_game,
                      seed=args.seed, record_dir=None, verbose=False)
    mppi = float(np.mean(scores))
    se = float(np.std(scores) / np.sqrt(args.num_game))
    print(f"[result]  MPPI[{args.arm} dyn]: {mppi:.3f}")
    net = None
    if pol is not None:
        net = mppi - float(np.mean(pol))
        print(f"[compare] planner net contribution = {net:+.3f}")
    print(f"[n] {args.num_game} games, SE≈{se:.3f}")

    if args.use_wb:
        import wandb
        run_name = args.wb_run or f"dynarm_{args.arm}_{args.shift}_seed{args.seed}"
        wandb.init(project=args.wb_project, group=args.wb_group, name=run_name,
                   config={**vars(args), "dynamics_source":
                           (args.latent_agent if args.arm == "latent" else args.dyn_ckpt)})
        wandb.summary["mppi_score"] = mppi
        wandb.summary["mppi_se"] = se
        if pol is not None:
            wandb.summary["policy_only"] = float(np.mean(pol))
            wandb.summary["planner_net"] = net
        wandb.summary["num_game"] = args.num_game
        wandb.finish()
        print(f"[wandb] logged as {run_name} (project={args.wb_project} group={args.wb_group})")

    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    main()
