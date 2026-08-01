"""Step A: does multi-step temporal aggregation improve gap LOCALIZATION?

Hypothesis: the dynamics gap is a drift that accumulates along the trajectory,
while per-step noise is ~independent. A window of k steps therefore raises SNR
~sqrt(k). Cheap validation: aggregate the per-step ensemble-mean score over a
centered window of k steps WITHIN each trajectory and re-measure how well it
matches the GT gap (per-traj match metric + pooled AUROC).

rolling MEAN denoises (better for diffuse contact regions); rolling MAX keeps
the spike (better if the gap is a sharp 1-2 step event). We report both.
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from cf_utils import REPO, TARGET_SHIFT
from step10_eval_ensemble import load_ensemble, member_scores, rollout
from match_utils import rolling, mean_match


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--ens_dir", default=f"{os.path.dirname(__file__)}/disc_ens")
    p.add_argument("--num_traj", type=int, default=16)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--tau_gap", type=float, default=0.025)
    p.add_argument("--ks", default="1,3,5,8,12")
    args = p.parse_args()
    dev = "cuda"

    import train_rl
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    agent, _, env_params = train_rl.load_model(args.weight, dev)
    src_params = dict(env_params); src_params["domain_shift"] = DomainShiftConfig(mode="none")
    src_env = PixelRobosuite(**src_params)
    tgt_params = dict(env_params); tgt_params["domain_shift"] = DomainShiftConfig(**TARGET_SHIFT)
    tgt_env = PixelRobosuite(**tgt_params); tgt_env.reset()
    members = load_ensemble(args.ens_dir, dev)

    trajs = rollout(agent, src_env, tgt_env, args.num_traj, args.seed, dev)
    # base per-step localizer = ensemble mean (best AUROC in step12, kappa=0)
    base, gts = [], []
    for t in trajs:
        sc = member_scores(members, t["s"], t["a"], t["ssrc"], dev)
        base.append(sc.mean(0)); gts.append(t["gap"])
    allgap = np.concatenate(gts)
    print(f"IN-DIST source RL: {len(trajs)} trajs, {len(allgap)} transitions, "
          f"GT gap>{args.tau_gap}: {100*(allgap>args.tau_gap).mean():.1f}% positive")

    ks = [int(x) for x in args.ks.split(",")]
    print(f"\n=== STEP A: temporal aggregation (per-traj match vs GT, pooled AUROC) ===")
    print(f"  {'mode':5s} {'k':>3s} {'match rho':>10s} {'match cos':>10s} "
          f"{'peak r@20':>10s} {'pooled AUROC':>13s}")
    results = {}
    for mode in ["mean", "max"]:
        for k in ks:
            preds = [rolling(b, k, mode) for b in base]
            mm = mean_match(preds, gts)
            au = auroc((allgap > args.tau_gap).astype(int), np.concatenate(preds))
            results[(mode, k)] = (preds, mm, au)
            print(f"  {mode:5s} {k:>3d} {mm['rho']:>10.3f} {mm['cos']:>10.3f} "
                  f"{mm['prk']:>10.3f} {au:>13.3f}")

    # pick the best config by match rho
    best = max(results, key=lambda kk: results[kk][1]["rho"])
    print(f"\n  best by match-rho: mode={best[0]} k={best[1]} "
          f"(rho {results[best][1]['rho']:.3f} vs k=1 {results[('mean',1)][1]['rho']:.3f})")

    # ---- visualize: GT gap vs raw(k=1) vs best-aggregated, first 8 trajs ----
    bpreds = results[best][0]
    raw = results[("mean", 1)][0]
    n = min(8, len(trajs)); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]; t = np.arange(len(gts[i]))
        ax.plot(t, raw[i], c="C0", lw=0.8, alpha=0.6, label="ens mean (k=1)")
        ax.plot(t, bpreds[i], c="C2", lw=1.6, label=f"{best[0]} k={best[1]}")
        ax.set_ylim(-0.05, max(0.6, max(raw[i].max(), bpreds[i].max()) * 1.1))
        ax2 = ax.twinx()
        ax2.plot(t, gts[i], c="C3", lw=1.4, alpha=0.8, label="GT gap")
        ax2.plot(t, np.linalg.norm(trajs[i]["s"][:, 9:12] - trajs[i]["s"][:, 23:26], axis=1),
                 c="C4", lw=0.8, ls=":", alpha=0.5, label="needle-ring dist")
        ax2.set_ylim(0, max(0.16, gts[i].max() * 1.1))
        mm = mean_match([bpreds[i]], [gts[i]])
        ax.set_title(f"traj{i} succ={trajs[i]['success']} rho={mm['rho']:.2f}", fontsize=9)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"STEP A: temporal aggregation ({best[0]} k={best[1]}) vs GT gap", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{os.path.dirname(__file__)}/step13_multistep.png"
    fig.savefig(out, dpi=120); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
