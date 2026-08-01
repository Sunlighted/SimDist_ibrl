"""Replot step5 trajectories on a GLOBAL unified (raw, un-normalized) scale so
disc-score / GT-gap amplitudes are comparable ACROSS trajectories and policies.

Shows that the target-only-BC failure trajectories' disc score is NOT larger in
absolute terms than the successful source-BC trajectories — it just fires in the
wrong (no-gap) places, and per-curve normalization made it look huge.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(npz):
    z = np.load(npz, allow_pickle=True)
    return str(z["tag"]), list(z["trajs"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npzs", nargs="+", default=[
        f"{os.path.dirname(__file__)}/step5_sourcebc.npz",
        f"{os.path.dirname(__file__)}/step5_targetbc.npz"])
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step5_global_scale.png")
    args = p.parse_args()

    groups = [load(n) for n in args.npzs]
    # global raw scales
    score_max = max(t["score"].max() for _, trs in groups for t in trs)
    gap_max = max(t["gap"].max() for _, trs in groups for t in trs)
    print(f"GLOBAL raw scales: disc score max = {score_max:.3f}, GT gap max = {gap_max:.4f}")
    for tag, trs in groups:
        sc = np.concatenate([t["score"] for t in trs])
        gp = np.concatenate([t["gap"] for t in trs])
        print(f"  {tag:32s}: disc score mean={sc.mean():.3f} p95={np.quantile(sc,.95):.3f} "
              f"max={sc.max():.3f} | GT gap mean={gp.mean():.4f} max={gp.max():.4f}")

    ncol = max(len(trs) for _, trs in groups)
    nrow = len(groups)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             squeeze=False, sharey=True)
    for r, (tag, trs) in enumerate(groups):
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(trs):
                ax.axis("off"); continue
            t = trs[c]
            x = np.arange(len(t["score"]))
            ax.plot(x, t["score"], color="C0", lw=1.0, label="disc score (raw)")
            ax.set_ylim(0, score_max * 1.05)
            ax2 = ax.twinx()
            ax2.plot(x, t["gap"], color="C3", lw=1.2, label="GT gap (raw)")
            ax2.set_ylim(0, gap_max * 1.05)
            ax2.set_yticklabels([]) if c < ncol - 1 else ax2.set_ylabel("GT gap", color="C3")
            ok = "OK" if t["success"] else "FAIL"
            ax.set_title(f"{tag.split('_')[1] if '_' in tag else tag} "
                         f"traj seed{t['seed']} [{ok}]", fontsize=8)
            if c == 0:
                ax.set_ylabel("disc score", color="C0")
    fig.suptitle("step5 on GLOBAL raw scale — blue=disc score (shared left), red=GT gap (shared right)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
