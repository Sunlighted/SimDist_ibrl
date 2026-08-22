"""Threading_D0: the source-anchor sweep (docs/simdist.md 42), seed-averaged.

`bc_anchor_coef = lambda` adds  lambda * MSE(pi_theta(s_src), pi_src(s_src))  on
SOURCE states to the stage-2 BC loss, on top of the usual demo term. Everything
else is held fixed across the 12 runs: the same stage-1 ckpt (dynens, H=5), the
same weak target demo set (`sat_160000_s201`), the same anchor pool
(`source_diverse_v2`, 2000 eps / 40k states / batch 256), bc_steps=1000,
adapt_value=0, max_transitions=40000.

Read-out is the mean over the 20-40k window of the in-training evals (100
episodes each), NOT the endpoint and NOT model0.pt -- see the methodology rules
in CLAUDE.md.

Caveats drawn on the figure, because they change how it should be read:
  * lambda=0 seed 1 is a REUSED older baseline run (bc_anchor_coef absent =>
    0 by the bit-compatible default) logged every 1k instead of every 2k.
  * lambda=3 and lambda=10 have only 2 seeds => df=1, so lambda=10's t=2.7 is
    NOT significant (critical t=12.7).

Colours: one-hue ordinal blue ramp from the dataviz reference palette, light ->
dark with lambda (steps 250/350/450/550/700). Validated on the ordinal checks:
monotone L, adjacent dL >= 0.093, hue spread 3.6 deg, light-end contrast 2.06:1.

Usage:
    python viz/plot_lambda_sweep.py
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(_REPO, "exps/mimicgen")

# lambda -> (run dirs, ordinal ramp step)
ARMS = [
    (0.0,  ["threading_simdist_dynens_adapt_sat160000_s201_seed1",
            "threading_lamadapt_0.0_s2", "threading_lamadapt_0.0_s3"],
     "#86b6ef", (5, 2)),
    (0.1,  ["threading_lamadapt_0.1_s1", "threading_lamadapt_0.1_s2",
            "threading_lamadapt_0.1_s3"], "#5598e7", (6, 1.6, 1.4, 1.6)),
    (1.0,  ["threading_lamadapt_1.0_s1", "threading_lamadapt_1.0_s2",
            "threading_lamadapt_1.0_s3"], "#2a78d6", ()),
    (3.0,  ["threading_lamadapt_3.0_s1", "threading_lamadapt_3.0_s2"],
     "#1c5cab", (2.6, 1.8)),
    (10.0, ["threading_lamadapt_10.0_s1", "threading_lamadapt_10.0_s2"],
     "#0d366b", (1.2, 1.8)),
]
WINDOW_LO, WINDOW_HI = 20_000, 40_000
GRID = np.arange(0, WINDOW_HI + 1, 2000)          # every run has these points

# chart chrome (dataviz reference palette, light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"

PAT = re.compile(r"\[adapt @ (\d+) transitions\] target_score=([0-9.]+)")


def curve(run):
    with open(os.path.join(EXP, run, "train.log"), errors="ignore") as f:
        pts = {int(a): float(b) for a, b in PAT.findall(f.read())}
    k = np.array(sorted(pts))
    return k, np.array([pts[i] for i in k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_REPO, "viz/lambda_sweep_threading.png"))
    args = p.parse_args()

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), dpi=200, gridspec_kw=dict(width_ratios=[1.55, 1]))
    fig.patch.set_facecolor(SURFACE)

    stats = []
    for lam, runs, color, dashes in ARMS:
        Y = np.stack([np.interp(GRID, *curve(r)) for r in runs])
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        axL.plot(GRID / 1000, mu, color=color, lw=2.6 if lam == 1 else 2,
                 dashes=dashes, zorder=4 if lam == 1 else 3,
                 solid_capstyle="round",
                 label=f"$\\lambda$ = {lam:g}" + ("  (no anchor)" if lam == 0 else ""))
        axL.fill_between(GRID / 1000, mu - sd, mu + sd, color=color, alpha=0.13, lw=0)
        axL.scatter([GRID[-1] / 1000], [mu[-1]], s=30, color=color, zorder=5,
                    edgecolor=SURFACE, linewidth=1.6)
        sel = (GRID >= WINDOW_LO) & (GRID <= WINDOW_HI)
        stats.append((lam, color, Y[:, sel].mean(1), len(runs)))

    # ---- left: adapt curves -------------------------------------------------
    axL.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color=GRIDLINE, alpha=0.55, zorder=0)
    axL.text(30, 1.015, "read-out window   (mean of 11 evals x 100 episodes)",
             ha="center", va="top", fontsize=8, color=MUTED)
    axL.set_xlim(0, 41)
    axL.set_ylim(0.42, 1.03)
    axL.set_xlabel("stage-3 target transitions (k)", fontsize=9.5, color=INK2)
    axL.set_ylabel("target success rate (MPPI)", fontsize=9.5, color=INK2)
    axL.set_title("Adapt curves, mean over seeds ($\\pm$1 sd)",
                  fontsize=10.5, color=INK, pad=8, loc="left")
    leg = axL.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK2,
                     borderpad=0.2, labelspacing=0.35, handlelength=2.2)
    leg.set_title("source anchor", prop=dict(size=8.5))
    leg.get_title().set_color(MUTED)

    # ---- right: the inverted U ----------------------------------------------
    xs = np.arange(len(stats))
    mus = np.array([s[2].mean() for s in stats])
    sds = np.array([s[2].std(ddof=1) for s in stats])
    axR.plot(xs, mus, color=AXIS, lw=2, zorder=2, solid_capstyle="round")
    for x, (lam, color, vals, n) in zip(xs, stats):
        axR.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), color=color,
                     elinewidth=1.4, capsize=4, capthick=1.4, zorder=3)
        axR.scatter(np.full(n, x) + np.linspace(-.09, .09, n), vals, s=22,
                    color=color, alpha=.45, lw=0, zorder=4)          # per-seed
        axR.scatter([x], [vals.mean()], s=95, color=color, zorder=5,
                    edgecolor=SURFACE, linewidth=2)                  # 2px surface ring

    peak = int(np.argmax(mus))
    # paired by seed index against lambda=0 (same 3 stage-3 seeds)
    dp = stats[peak][2] - stats[0][2]
    t_paired = dp.mean() / (dp.std(ddof=1) / np.sqrt(len(dp)))
    axR.annotate(f"{mus[peak]:.3f}", (xs[peak], mus[peak]), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=10, color=INK, weight="bold")
    axR.annotate(f"{mus[0]:.3f}\nbaseline", (xs[0], mus[0]), textcoords="offset points",
                 xytext=(4, -30), ha="center", fontsize=9, color=INK2)
    axR.text(0.025, 0.965,
             f"$\\lambda$=1 vs $\\lambda$=0, paired by seed:   $\\Delta$ = {dp.mean():+.3f}"
             f"   (t = {t_paired:.1f},  {int((dp > 0).sum())}/{len(dp)} seeds same sign)",
             transform=axR.transAxes, ha="left", va="top", fontsize=9, color=INK2)

    axR.set_xticks(xs)
    axR.set_xticklabels([f"{s[0]:g}\n" + (f"n={s[3]}" if s[3] < 3 else "n=3")
                         for s in stats], fontsize=9)
    axR.set_xlim(-0.5, len(stats) - 0.5)
    axR.set_ylim(0.70, 1.02)
    axR.set_xlabel("source-anchor coefficient  $\\lambda$   (not to scale)",
                   fontsize=9.5, color=INK2)
    axR.set_ylabel("target success, 20-40k window mean", fontsize=9.5, color=INK2)
    axR.set_title("Inverted U: peak at $\\lambda$ = 1", fontsize=10.5, color=INK,
                  pad=8, loc="left")

    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=3)

    fig.suptitle(
        "Source anchor on SimDist stage-2  --  Threading_D0, weak base (sat_160000_s201)",
        fontsize=12.5, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.918,
             "L = MSE($\\pi_\\theta$(s_demo), a_demo) + $\\lambda\\cdot$MSE($\\pi_\\theta$(s_src), "
             "$\\pi_{src}$(s_src));  identical stage-1 ckpt, demo set, anchor pool and planner across all 12 runs.",
             fontsize=9, color=INK2, ha="left")
    fig.text(0.008, 0.022,
             "Caveats:  $\\lambda$=0 seed 1 is a reused older baseline run (logged every 1k, not 2k).   "
             "$\\lambda$=3 and $\\lambda$=10 have 2 seeds => df=1, so $\\lambda$=10's t=2.7 is not significant.   "
             "Single demo set, single stage-1 seed, single task.",
             fontsize=8, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.05, 1, 0.885])
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)
    print(f"  paired delta(lambda=1 vs 0) = {dp.mean():+.4f}  t = {t_paired:.1f}  "
          f"per-seed {[round(float(v), 3) for v in dp]}")
    for lam, _, vals, n in stats:
        print(f"  lambda={lam:<5g} n={n}  mean={vals.mean():.3f} sd={vals.std(ddof=1):.3f}"
              f"  seeds={[round(float(v), 3) for v in vals]}")


if __name__ == "__main__":
    main()
