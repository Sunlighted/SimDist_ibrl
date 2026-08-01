"""Ablation: how many Stage-2 policy-BC steps are best (before any dynamics
adaptation)? Reuses train_simdist_adapt.AdaptWorkspace (loads the stage-1 ckpt,
20 TARGET demos, TARGET env, forces MPPI), then BC-finetunes the policy head
CONTINUOUSLY and evaluates (MPPI, TARGET) at a set of milestones along the SAME
BC trajectory. Because BC is cumulative, eval@K == "BC for K steps then eval", so
one run yields the whole steps->score curve. All milestones share ONE eval seed
(paired comparison, low variance).

Run:
  python ablate_bc_steps.py \
      --stage1_ckpt exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt \
      --save_dir exps/mimicgen/threading_simdist_bcablate_seed1 \
      --num_eval_episode 200
  # milestones via env var (default below): BC_MILESTONES=0,200,500,1000,2000,4000
"""

import os
import sys

import pyrallis
import torch

import common_utils
from train_simdist_adapt import AdaptConfig, AdaptWorkspace


def main():
    cfg = pyrallis.parse(config_class=AdaptConfig)  # type: ignore
    milestones = [int(x) for x in
                  os.environ.get("BC_MILESTONES", "0,200,500,1000,2000,4000").split(",")]
    milestones = sorted(set(milestones))

    ws = AdaptWorkspace(cfg)
    # snapshot the freshly-loaded stage-1 policy so milestone 0 = "no BC".
    # EVAL_SEEDS: comma list -> eval at EACH seed per milestone (seed-sensitivity
    # probe). Default a single fixed seed => paired comparison across milestones.
    eval_seeds = [int(x) for x in os.environ.get("EVAL_SEEDS", "12345").split(",")]

    print(common_utils.wrap_ruler(
        f"BC-steps ablation (MPPI eval, sigma="
        f"{ws.base_cfg.eval_hand_init_perturb_joint_sigma}, "
        f"{cfg.num_eval_episode} games, milestones={milestones})"))

    results = []
    trained = 0
    for m in milestones:
        # BC-train continuously up to milestone m
        ws.agent.train()
        for _ in range(m - trained):
            states, actions, _ = ws.demo_replay.sample(cfg.bc_batch_size, ws.H, "cuda")
            metrics = ws.agent.update_bc_policy(states, actions, lr=cfg.bc_lr)
        trained = m
        bc_loss = metrics["bc/policy_loss"] if m > 0 else float("nan")
        scores = {s: ws.eval(seed=s) for s in eval_seeds}
        mean_score = sum(scores.values()) / len(scores)
        results.append((m, mean_score, scores, bc_loss))
        ws.stat["ablate/bc_steps"].append(m)
        ws.stat["score/score"].append(mean_score)
        if m > 0:
            ws.stat["bc/policy_loss"].append(bc_loss)
        ws.stat.summary(m, reset=True)
        per_seed = "  ".join(f"seed{s}={v:.3f}" for s, v in scores.items())
        print(f"[bc-ablate] steps={m:5d}  mean={mean_score:.3f}  ({per_seed})  "
              f"bc_loss={bc_loss:.5f}")

    print(common_utils.wrap_ruler("BC-steps ablation summary"))
    print(f"{'bc_steps':>10} | {'mean_score':>10} | {'bc_loss':>9} | per-seed")
    for m, mean_score, scores, bc_loss in results:
        per_seed = " ".join(f"{s}:{v:.3f}" for s, v in scores.items())
        print(f"{m:>10} | {mean_score:>10.3f} | {bc_loss:>9.5f} | {per_seed}")
    best = max(results, key=lambda r: r[1])
    print(f"\nbest: bc_steps={best[0]} -> mean_score={best[1]:.3f}")

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
    torch.backends.cudnn.benchmark = True   # type: ignore
    main()
