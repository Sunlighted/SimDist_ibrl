"""Square rotw1: does the stage-2 demo collector's training stage matter?

Both arms are `sq_dynens_*` -- identical stage-1 (5-head dynamics ensemble, H=5),
identical planner (`init_std=0.15, base_policy_std=0.0`, the Square-specific trust
region from docs/tasks.md 41), identical optimistic collection and eval sweep.
The ONLY difference is which target-B expert snapshot collected that seed's 20
stage-2 demos:

    400k  expert eval 0.94  -- the run had NOT yet touched 1.00  (unsaturated)
    600k  expert eval 1.00  -- the first step at which it did    (just saturated)

Threading (docs/demo-collection.md 38) found the unsaturated collector better by
+0.139; this is the same contrast on a second task, so the sign is the question.

Style matches viz/plot_saturation_curves.py: solid = mean over seeds, shaded =
+-1 SEM, faint = individual seeds, all smoothed with a centred rolling mean. SEM
(not sd) because the question is where the group mean sits; each seed here uses a
DIFFERENT demo set, so the seed spread contains the demo-draw variance itself.

Usage:
    python viz/plot_square_demo_source.py
    python viz/plot_square_demo_source.py --band std --window 5
"""
import argparse
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

EXP = os.path.join(_REPO, "exps/square_rotw1_6arm")
SEEDS = [1, 101, 201, 301, 401, 501, 601, 701, 801, 901]
GROUPS = [
    ("400k snapshot  --  collector still improving (eval 0.94)", "#1f77b4",
     "sq_dynens_d400000"),
    ("600k snapshot  --  collector just saturated (eval 1.00)", "#d62728",
     "sq_dynens_d600000"),
]
# `score_best` = max over the eval-time pessimism sweep {0, 0.05, 0.1, 0.3}. Both
# arms take the same max over the same 4 coefficients, so the winner's-curse bias
# is identical on both sides and cancels in the comparison.
KEY = "score/score_best"
WINDOW_LO, WINDOW_HI = 60_000, 100_000


def curve(run):
    path = os.path.join(EXP, run, "train.log")
    pts = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.match(rf"\s*(\d+): {re.escape(KEY)}\s*:\s*([0-9.]+)\s*$", line)
            if m:
                pts[int(m.group(1))] = float(m.group(2))
    k = np.array(sorted(pts))
    return k, np.array([pts[i] for i in k])


def smooth(y, w):
    if w <= 1:
        return y
    out = np.empty_like(y, dtype=float)
    h = w // 2
    for i in range(len(y)):
        out[i] = y[max(0, i - h):i + h + 1].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--band", choices=["sem", "std"], default="sem")
    p.add_argument("--out", default=os.path.join(_REPO, "viz/square_demo_source.png"))
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary = []
    for label, color, pref in GROUPS:
        xs, ys = None, []
        for s in SEEDS:
            k, v = curve(f"{pref}_seed{s}")
            if xs is None:
                xs = k
            ys.append(smooth(np.interp(xs, k, v), args.window))
        Y = np.stack(ys)
        mu, spread = Y.mean(0), Y.std(0, ddof=1)
        band = spread / np.sqrt(len(SEEDS)) if args.band == "sem" else spread
        ax.plot(xs / 1000, mu, color=color, lw=2.2,
                label=f"{label}   (n={len(SEEDS)} seeds)")
        ax.fill_between(xs / 1000, mu - band, mu + band, color=color, alpha=0.18, lw=0)
        for y in Y:
            ax.plot(xs / 1000, y, color=color, lw=0.6, alpha=0.25)
        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        summary.append(Y[:, sel].mean(1))

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color="0.85", alpha=0.35, zorder=0)
    ax.text((WINDOW_LO + WINDOW_HI) / 2000, 0.045, "60-100k window\n(quoted numbers)",
            ha="center", va="bottom", fontsize=8, color="0.35")

    a, b = summary
    d = b.mean() - a.mean()                      # 600k - 400k
    dp = b - a                                   # paired by seed (same seed index)
    se = dp.std(ddof=1) / np.sqrt(len(dp))
    ax.set_title(
        "Square_D0 (rotw1): stage-2 demo collector, unsaturated vs saturated\n"
        "SimDist + dynamics ensemble; everything else identical\n"
        f"60-100k:  400k {a.mean():.3f} +- {a.std(ddof=1):.3f}   vs   "
        f"600k {b.mean():.3f} +- {b.std(ddof=1):.3f}   "
        f"delta = {d:+.3f}  (paired z = {dp.mean()/se:+.2f}, "
        f"{int((dp > 0).sum())}/{len(dp)})",
        fontsize=10)
    ax.set_xlabel("stage-3 TARGET transitions collected (k)")
    ax.set_ylabel("TARGET success (MPPI, best over pessimism sweep)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, xs.max() / 1000)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"saved -> {args.out}", file=sys.stderr)
    for (lab, _, _), v in zip(GROUPS, summary):
        print(f"  {lab[:14]:<16} {np.round(v, 3)}  mean {v.mean():.3f}", file=sys.stderr)
    print(f"  paired delta (600k - 400k): {dp.mean():+.3f}  SE {se:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
