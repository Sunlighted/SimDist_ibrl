"""Visualize the per-step gap-confidence WEIGHT along random SOURCE trajectories.

The weighted-BC trainer draws each source transition with probability ∝ its
weight `c = exp(-gap/tau)` (or `1[gap<thr]` in binary mode), where `gap` is the
GT per-transition dynamics gap cached by scripts/compute_source_gaps.py. This
script samples a few source episodes and plots, per episode, the weight over the
trajectory together with the needle↔ring distance (the contact-phase proxy), so
you can see WHERE in a trajectory the source is trusted (free-space, weight≈1)
vs distrusted (contact/insertion, weight↓).

  python scripts/viz_source_weights.py --tau 0.01 --n 6 --out docs/source_weights_traj.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gap_cache",
                   default="release/data/mimicgen/threading_d0_source_sigma0.1/gt_gaps_n2000.npz")
    p.add_argument("--tau", type=float, default=0.01, help="c = exp(-gap/tau)")
    p.add_argument("--mode", default="soft", choices=["soft", "binary"])
    p.add_argument("--threshold", type=float, default=0.013, help="binary cutoff")
    p.add_argument("--n", type=int, default=6, help="# random episodes to plot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contact_thresh", type=float, default=0.15)
    p.add_argument("--out", default="docs/source_weights_traj.png")
    args = p.parse_args()

    d = np.load(args.gap_cache)
    ep_id, t_in_ep, gap, dist = d["ep_id"], d["t_in_ep"], d["gap"], d["dist"]

    def to_weight(g):
        if args.mode == "binary":
            return (g < args.threshold).astype(np.float32)
        return np.exp(-g / max(args.tau, 1e-8))

    w_all = to_weight(gap)
    print(f"loaded {len(gap)} transitions over {len(np.unique(ep_id))} episodes")
    print(f"tau={args.tau} mode={args.mode}: mean weight {w_all.mean():.3f}")

    uniq = np.unique(ep_id)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(uniq, size=min(args.n, len(uniq)), replace=False))

    ncol = 2
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.7 * nrow), squeeze=False)

    for k, ep in enumerate(chosen):
        ax = axes[k // ncol][k % ncol]
        m = ep_id == ep
        order = np.argsort(t_in_ep[m])
        t = t_in_ep[m][order]
        w = to_weight(gap[m])[order]
        g = gap[m][order]
        dd = dist[m][order]
        contact = dd < args.contact_thresh

        # weight (left axis)
        ax.plot(t, w, color="tab:blue", lw=1.8, label="weight c")
        ax.fill_between(t, 0, w, color="tab:blue", alpha=0.12)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("weight c", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        # shade contact phase
        in_c = False
        for i in range(len(t)):
            if contact[i] and not in_c:
                start = t[i]; in_c = True
            if in_c and (i == len(t) - 1 or not contact[i]):
                ax.axvspan(start, t[i], color="tab:red", alpha=0.08)
                in_c = False

        # GT gap (right axis) — the raw value the weight is derived from
        ax2 = ax.twinx()
        ax2.plot(t, g, color="tab:orange", lw=1.4, label="GT gap")
        # reference: gap where c=0.5 (gap = tau*ln2)
        ax2.axhline(args.tau * np.log(2), color="tab:orange", lw=0.8, ls=":", alpha=0.6)
        ax2.set_ylabel("GT gap", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # needle-ring distance (third axis, offset) — contact-phase context
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.13))
        ax3.plot(t, dd, color="tab:gray", lw=1.0, ls="--", alpha=0.8, label="needle-ring dist")
        ax3.axhline(args.contact_thresh, color="tab:red", lw=0.8, ls=":", alpha=0.5)
        ax3.set_ylabel("needle-ring dist (m)", color="tab:gray")
        ax3.tick_params(axis="y", labelcolor="tab:gray")

        ax.set_title(f"source demo {ep}  (mean w={w.mean():.2f}, "
                     f"contact mean w={w[contact].mean():.2f})" if contact.any()
                     else f"source demo {ep}  (mean w={w.mean():.2f})", fontsize=9)
        ax.set_xlabel("step")

    # hide unused axes
    for k in range(len(chosen), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(
        f"Source-demo per-step weight c=exp(-gap/{args.tau}) [blue] vs raw GT gap [orange] "
        f"vs needle-ring dist [gray]  (red shade = contact, dist<{args.contact_thresh}m; "
        f"mean w={w_all.mean():.2f})",
        fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.9, hspace=0.55, wspace=0.5, top=0.93, bottom=0.08)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
