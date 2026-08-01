"""Step 2: visualize the ground-truth per-transition gap ||s'_tgt - s'_src||.

Shows the distribution and per-episode gap curves so we can see WHERE the
dynamics gap concentrates (grasp/transport vs free vs insertion).
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/step2_gap.png")
    p.add_argument("--n_show", type=int, default=4, help="example episodes to plot")
    args = p.parse_args()

    from data_utils import state_gap, align_quats
    z = np.load(args.pairs)
    dist, ep, ts = z["dist"], z["ep_id"], z["t_in_ep"]
    s, s_src, s_tgt = z["s"], z["s_src"], z["s_tgt"]
    gap = state_gap(s_src, s_tgt)  # quaternion-aware (recomputed)

    # distribution
    qs = np.quantile(gap, [0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    print(f"gap quantiles [50,75,90,95,99,100]: {np.round(qs,4)}")
    print(f"gap by phase (median): contact {np.median(gap[dist<0.15]):.4f} | "
          f"free {np.median(gap[dist>=0.15]):.4f}")

    # which state dims drive the largest gaps? (per-dim |Δ| averaged over top-5% gap rows)
    top = gap > np.quantile(gap, 0.95)
    perdim = np.abs(s_tgt - align_quats(s_src, s_tgt))[top].mean(0)
    # 37-dim layout: eef_pos[0:3] eef_quat[3:7] grip[7:9] obj[9:37]
    #   obj: needle pos[9:12] quat[12:16] to_eef[16:19] to_eef_quat[19:23]
    #        tripod pos[23:26] quat[26:30] to_eef[30:33] to_eef_quat[33:37]
    groups = {"eef_pos": (0, 3), "eef_quat": (3, 7), "gripper": (7, 9),
              "needle_pos": (9, 12), "needle_quat": (12, 16),
              "needle_to_eef": (16, 19), "tripod_pos": (23, 26)}
    print("top-5%-gap mean |Δ| by group:")
    for k, (a, b) in groups.items():
        print(f"    {k:16s} {perdim[a:b].mean():.4f}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.hist(np.log10(gap + 1e-6), bins=60, color="C0")
    ax.axvline(np.log10(0.006), color="r", ls="--", label="recon noise floor ~0.006")
    ax.set_xlabel("log10(gap)"); ax.set_ylabel("count"); ax.set_title("gap distribution (log)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(dist, gap, s=3, alpha=0.25)
    ax.axhline(0.006, color="r", ls="--", lw=1)
    ax.axvline(0.15, color="C1", ls=":", lw=1, label="contact thresh")
    ax.set_xlabel("needle-ring dist"); ax.set_ylabel("gap"); ax.set_yscale("log")
    ax.set_title("gap vs needle-ring distance"); ax.legend(fontsize=8)

    # per-episode example curves
    uniq = np.unique(ep)[: args.n_show]
    for i, e in enumerate(uniq):
        ax = axes[1, 0] if i < 2 else axes[1, 1]
        m = ep == e
        order = np.argsort(ts[m])
        g = gap[m][order]; d = dist[m][order]; t = ts[m][order]
        ln = ax.plot(t, g, lw=1.5, label=f"ep{e} gap")[0]
        ax.plot(t, d, lw=1.0, ls=":", alpha=0.6, color=ln.get_color())
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlabel("step in episode"); ax.set_ylabel("gap (solid) / dist (dotted)")
        ax.set_title("per-episode: ground-truth gap vs needle-ring dist")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
