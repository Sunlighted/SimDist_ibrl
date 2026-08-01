"""Does the source-value > target-value MPPI gap depend on the PLANNING HORIZON?

Swap ONLY the value head (source stage-1 value  vs  target-retrained value from
the head-ablation) into the SAME target-adapted world model, and run real MPPI
eval on the TARGET env at several horizons H. The dynamics was distilled for H=3,
so H>3 degrades rollout accuracy for BOTH heads (common-mode) — the SOURCE-vs-
TARGET GAP at each fixed H is the signal.

Reads:
  H=1  gap gone  -> the gap needs multi-step rollout (V* mismatch amplified by depth)
  gap ~const in H -> horizon irrelevant; it's the value SEMANTICS at any lookahead
  longer H lifts TARGET -> horizon-reach matters (target value needs deeper plan)

Usage:
  python scripts/diag_horizon_value_sweep.py \
      --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --adapt_ckpt  exps/mimicgen/threading_simdist_adapt_seed1/latest.pt \
      --ablate_ckpt exps/mimicgen/threading_simdist_ablate_val_seed1/model0.pt \
      --horizons 1,2,3,5 --num_game 60 --num_proc 12 --gpu 0
"""
import argparse, copy, os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, pyrallis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--adapt_ckpt", required=True)
    ap.add_argument("--ablate_ckpt", required=True)
    ap.add_argument("--horizons", default="1,2,3,5")
    ap.add_argument("--num_game", type=int, default=60)
    ap.add_argument("--num_proc", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    horizons = [int(x) for x in args.horizons.split(",")]

    from train_simdist import Workspace, MainConfig
    from rl.simdist_planner import MppiPlanner, apply_original_recipe
    from evaluate.multi_process_eval import run_eval as run_eval_mp

    stage1_dir = os.path.dirname(args.stage1_ckpt)
    base_cfg: MainConfig = pyrallis.load(MainConfig, open(os.path.join(stage1_dir, "cfg.yaml")))  # type: ignore
    base_cfg.simdist.use_planner = 1
    apply_original_recipe(base_cfg.simdist.planner)      # broad SimDist recipe (H left alone)
    ws = Workspace(base_cfg, from_main=False)
    agent = ws.agent.to("cuda")
    env_params = ws.eval_env_params                       # TARGET env
    act_dim = agent.action_dim

    # two value heads: SOURCE (adapt ckpt) vs TARGET (ablation ckpt)
    agent.load_state_dict(torch.load(args.adapt_ckpt))
    src_val = copy.deepcopy(agent.model.value.state_dict())
    abl_sd = torch.load(args.ablate_ckpt)
    tgt_val = {k[len("model.value."):]: v for k, v in abl_sd.items()
               if k.startswith("model.value.")}
    print(f"planner recipe: num_samples={base_cfg.simdist.planner.num_samples} "
          f"num_base={base_cfg.simdist.planner.num_base_override} "
          f"init_std={base_cfg.simdist.planner.init_std} "
          f"iters={base_cfg.simdist.planner.iterations} temp={base_cfg.simdist.planner.temperature}")
    print(f"horizons={horizons}  num_game={args.num_game}  (dynamics distilled for H=3)")

    results = {}
    for name, vsd in [("SOURCE-value", src_val), ("TARGET-value", tgt_val)]:
        agent.model.value.load_state_dict(vsd)
        agent.model.value_target.load_state_dict(vsd)
        agent.eval()
        for H in horizons:
            pcfg = copy.deepcopy(base_cfg.simdist.planner)
            pcfg.horizon = H
            agent.planner = MppiPlanner(agent.model, pcfg, act_dim, device="cuda")
            scores = run_eval_mp(env_params=env_params, agent=agent,
                                 num_proc=args.num_proc, num_game=args.num_game,
                                 seed=args.seed, verbose=False)
            sr = float(np.mean(scores))
            results[(name, H)] = sr
            print(f"  {name:13s}  H={H}  success={sr:.3f}  ({int(np.sum(np.array(scores)>0))}/{args.num_game})",
                  flush=True)

    print("\n" + "=" * 60)
    print(f"{'H':>3} | {'SOURCE':>8} | {'TARGET':>8} | {'gap(S-T)':>9} | ratio")
    for H in horizons:
        s = results[("SOURCE-value", H)]; t = results[("TARGET-value", H)]
        ratio = s / t if t > 1e-6 else float("inf")
        print(f"{H:>3} | {s:>8.3f} | {t:>8.3f} | {s-t:>9.3f} | {ratio:.2f}x")
    print("\nInterpretation:")
    print("  gap ~constant across H  -> horizon is NOT the cause (pure value semantics)")
    print("  gap grows with H        -> multi-step rollout amplifies the V* mismatch")
    print("  TARGET rises with H     -> horizon-reach helps the accurate value")
    import sys; sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
