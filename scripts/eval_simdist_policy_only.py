"""Eval a SimDist ckpt with the POLICY HEAD ALONE (no MPPI) on target + source."""
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_simdist
from train_simdist import Workspace, MainConfig
from env.domain_shift import DomainShiftConfig
from evaluate import run_eval_mp
import pyrallis


def build(weight, use_planner, planner_overrides=None, bc_proposal=None):
    cfg_path = os.path.join(os.path.dirname(weight), "cfg.yaml")
    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))
    ws = Workspace(cfg, from_main=False)
    agent = ws.agent
    agent.load_state_dict(torch.load(weight))
    agent = agent.to("cuda")
    agent.cfg.use_planner = use_planner
    # override planner params (eval-time only; world model is frozen) + rebuild
    if planner_overrides:
        from rl.simdist_planner import MppiPlanner
        pc = agent.planner.cfg
        for k, v in planner_overrides.items():
            if v is not None:
                setattr(pc, k, v)
        agent.planner = MppiPlanner(agent.model, pc, agent.action_dim, device="cuda")
        print(f"[planner] H={pc.horizon} use_reward={pc.use_reward} "
              f"num_samples={pc.num_samples} mixture_coef={pc.mixture_coef:.4f} "
              f"iters={pc.iterations} init_std={pc.init_std} min_std={pc.min_std} "
              f"base_policy_std={pc.base_policy_std} temp={pc.temperature} "
              f"momentum={pc.momentum} elites={pc.num_elites}")
    # optional EXTERNAL proposal policy for the planner (e.g. a standalone BC).
    # Attached AFTER any planner rebuild (rebuild resets proposal_policy).
    if bc_proposal:
        import train_bc
        bc = train_bc.load_model(bc_proposal, "cuda", verbose=False)[0]
        bc.eval()
        for p in bc.parameters():
            p.requires_grad_(False)

        @torch.no_grad()
        def _bc_proposal(state):  # state [N,obs] -> action [N,act]
            return bc.net(state).clamp(-1, 1)

        agent.planner.proposal_policy = _bc_proposal
        print(f"[proposal] external BC policy from {bc_proposal}")
    return agent, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--num_game", type=int, default=50)
    ap.add_argument("--use_planner", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--joint_sigma", type=float, default=None,
                    help="override eval hand_init_perturb_joint_sigma (match BC eval)")
    ap.add_argument("--strict_tripod_disp", type=float, default=None,
                    help="override strict_success.max_tripod_disp (anti ring-shove; default cfg=0.04)")
    ap.add_argument("--strict_needle_speed", type=float, default=None,
                    help="override strict_success.max_needle_speed (anti ramming; default cfg=0=off)")
    # planner overrides (only used when --use_planner 1)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--mixture_coef", type=float, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--init_std", type=float, default=None)
    ap.add_argument("--min_std", type=float, default=None)
    ap.add_argument("--base_policy_std", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--num_elites", type=int, default=None)
    ap.add_argument("--use_reward", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--seed_mean_from_policy", type=int, default=None)
    ap.add_argument("--paper", action="store_true",
                    help="use SimDist Table III manipulation MPPI params")
    ap.add_argument("--bc_proposal", type=str, default=None,
                    help="path to a StateBcPolicy ckpt to use as the MPPI proposal")
    args = ap.parse_args()

    overrides = dict(
        num_samples=args.num_samples, mixture_coef=args.mixture_coef,
        iterations=args.iterations, init_std=args.init_std, min_std=args.min_std,
        base_policy_std=args.base_policy_std, temperature=args.temperature,
        momentum=args.momentum, num_elites=args.num_elites,
        use_reward=args.use_reward, horizon=args.horizon,
        seed_mean_from_policy=args.seed_mean_from_policy)
    if args.paper:
        # Table III: 250 candidate + 100 noised-base = 350 total, mixture 100/350.
        overrides = dict(num_samples=350, mixture_coef=100.0 / 350.0, iterations=3,
                         init_std=1.0, min_std=0.05, base_policy_std=0.1,
                         temperature=0.4, momentum=0.0, num_elites=64)

    agent, ws = build(args.weight, args.use_planner, overrides, bc_proposal=args.bc_proposal)

    tgt_params = dict(ws.eval_env_params)
    src_params = dict(ws.eval_env_params)
    src_params["domain_shift"] = DomainShiftConfig(mode="none")
    if args.joint_sigma is not None:
        tgt_params["hand_init_perturb_joint_sigma"] = args.joint_sigma
        src_params["hand_init_perturb_joint_sigma"] = args.joint_sigma
        print(f"[eval] hand_init_perturb_joint_sigma = {args.joint_sigma}")
    if args.strict_tripod_disp is not None or args.strict_needle_speed is not None:
        import copy
        for pr in (tgt_params, src_params):
            ss = copy.deepcopy(pr["strict_success"])
            if args.strict_tripod_disp is not None:
                ss.max_tripod_disp = args.strict_tripod_disp
            if args.strict_needle_speed is not None:
                ss.max_needle_speed = args.strict_needle_speed
            pr["strict_success"] = ss
        print(f"[eval] strict override: max_tripod_disp={args.strict_tripod_disp} "
              f"max_needle_speed={args.strict_needle_speed}")

    tgt = run_eval_mp(env_params=tgt_params, agent=agent, num_proc=10,
                      num_game=args.num_game, seed=args.seed, verbose=False)
    src = run_eval_mp(env_params=src_params, agent=agent, num_proc=10,
                      num_game=args.num_game, seed=args.seed, verbose=False)
    print(f"\n=== {args.weight} | use_planner={args.use_planner} | n={args.num_game} ===")
    print(f"TARGET score = {np.mean(tgt):.3f}")
    print(f"SOURCE score = {np.mean(src):.3f}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
