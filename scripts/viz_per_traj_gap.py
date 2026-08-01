"""Per-trajectory GT-gap plots (one subplot per episode), in the style of
classifier_test/step13_multistep.png.

Reads the .npy saved by scripts/viz_gated_rollout_gap.py
(columns: episode, t, gap, c, needle_ring_dist, success) and draws a grid with
one subplot per episode:
  * left axis  (red)  : GT dynamics gap along the trajectory
  * right axis (gray dotted): needle-ring distance (phase: small = contact)
  * title: traj idx, success, peak gap, and the contact-phase mean gap

Usage:
  python scripts/viz_per_traj_gap.py gated_vs_target__gated_source-RL.npy
  python scripts/viz_per_traj_gap.py gated_vs_target__target_IBRL.npy --ncol 5
  # overlay both as two colors in the same grid (aligned by episode index):
  python scripts/viz_per_traj_gap.py A.npy B.npy --labels gated target
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/Chenyang/ibrl"
GAP_TAU = 0.024
CONTACT_THRESH = 0.15
COLORS = ["tab:red", "tab:blue", "tab:green"]


def load(path):
    d = np.load(path if os.path.isabs(path) else os.path.join(REPO, path))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npy", nargs="+", help="one or more .npy from viz_gated_rollout_gap.py")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--ncol", type=int, default=5)
    p.add_argument("--max_traj", type=int, default=20)
    p.add_argument("--gap_tau", type=float, default=GAP_TAU)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    datas = [load(x) for x in args.npy]
    labels = args.labels or [os.path.basename(x).replace(".npy", "").split("__")[-1] for x in args.npy]

    # episodes to show = union of episode ids, capped
    all_eps = sorted(set(int(e) for d in datas for e in np.unique(d[:, 0])))[: args.max_traj]
    n = len(all_eps)
    ncol = args.ncol
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
    gmax = max(d[:, 2].max() for d in datas) * 1.05

    for k, ep in enumerate(all_eps):
        ax = axes[k // ncol][k % ncol]
        ax2 = ax.twinx()
        title_bits = []
        for di, (d, lab) in enumerate(zip(datas, labels)):
            m = d[:, 0] == ep
            if not m.any():
                continue
            t = d[m, 1]; gap = d[m, 2]; nrd = d[m, 4]; succ = bool(d[m, 5][-1])
            contact = nrd < CONTACT_THRESH
            color = COLORS[di % len(COLORS)]
            ax.plot(t, gap, color=color, lw=1.3, label=f"{lab} gap" if k == 0 else None)
            ax2.plot(t, nrd, color=color, lw=0.9, ls=":", alpha=0.6,
                     label=f"{lab} needle-ring" if k == 0 else None)
            cmean = gap[contact].mean() if contact.any() else float("nan")
            title_bits.append(f"{'S' if succ else 'F'} pk={gap.max():.3f} ct={cmean:.3f}")
        ax.axhline(args.gap_tau, color="k", ls="--", lw=0.7)
        ax.set_ylim(0, gmax)
        ax2.set_ylim(0, 0.55)
        ax.set_title(f"traj{ep}  " + " | ".join(title_bits), fontsize=8)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
        if k % ncol == 0:
            ax.set_ylabel("GT gap", fontsize=8)
        if k % ncol == ncol - 1:
            ax2.set_ylabel("needle-ring dist", fontsize=8)

    # hide unused axes
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # one shared legend
    handles, labs = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[0][0].twinx().get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", fontsize=9, ncol=len(datas))
    fig.suptitle("Per-trajectory GT gap (left, solid) & needle-ring dist (right, dotted)  "
                 f"[gap_tau={args.gap_tau}, dashed; title: S/F  pk=peak gap  ct=contact-phase mean gap]",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.out or ("per_traj_" + "_vs_".join(labels) + ".png")
    out_path = os.path.join(REPO, out)
    fig.savefig(out_path, dpi=120)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
