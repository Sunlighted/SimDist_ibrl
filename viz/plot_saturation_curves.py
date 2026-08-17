"""Multi-seed adapt curves: demos from a NOT-YET-SATURATED collector vs a SATURATED one.

The one intervention that survived a preregistered replication (CLAUDE.md 38):
collect stage-2 demos from a policy that is already competent but whose training
curve has NOT yet touched eval=1.00.

    A (not saturated)  N3@160k   the run first hits eval=1.00 at 240k
    B (saturated)      N3@400k   deep past that point

Everything else is identical: same network, same sigma=0.02, same collection
script, same 25-demos-take-20, same adapt config (dynens stage-1, bc_steps 1000,
adapt_interval 1000, no explore). Only the collecting checkpoint and the
collection seed differ, 5 seeds per group.

Style is the usual RL-curve one: solid line = mean over seeds, shaded band =
+-1 SEM, both smoothed with a centred rolling mean. The band is SEM (not std)
because the question is "where is the group mean", and a single 20-demo
collection has a large draw variance of its own (std ~0.092, CLAUDE.md 38) --
plotting std would just re-draw that noise floor. Set --band std to see it.

Usage:
    python viz/plot_saturation_curves.py                    # -> viz/saturation_curves.png
    python viz/plot_saturation_curves.py --window 9 --band std
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

plt.rcParams["axes.unicode_minus"] = False   # keep the minus sign a real glyph

EXP = os.path.join(_REPO, "exps/mimicgen")
GROUPS = [
    ("N3@160k  --  collector still improving", "#1f77b4",
     ["snapB160_sig002", "sat160000_s101", "sat160000_s201",
      "sat160000_s301", "sat160000_s401"]),
    ("N3@400k  --  collector plateaued", "#d62728",
     ["sat400000_s1", "sat400000_s101", "sat400000_s201",
      "sat400000_s301", "sat400000_s401"]),
]
WINDOW_LO, WINDOW_HI = 25_000, 50_000     # the window every number in 38 is quoted on


def curve(arm):
    """(transitions, target_score) for one adapt run, sorted by transitions."""
    path = os.path.join(EXP, f"threading_simdist_dynens_adapt_{arm}_seed1/train.log")
    pts = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"\[adapt @ (\d+) transitions\].*?target_score=([0-9.]+)", line)
            if m:
                pts[int(m.group(1))] = float(m.group(2))
    k = np.array(sorted(pts))
    return k, np.array([pts[i] for i in k])


def smooth(y, w):
    """Centred rolling mean; edges shrink the window instead of padding."""
    if w <= 1:
        return y
    out = np.empty_like(y, dtype=float)
    half = w // 2
    for i in range(len(y)):
        out[i] = y[max(0, i - half):i + half + 1].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=7, help="rolling-mean width in eval points")
    p.add_argument("--band", choices=["sem", "std"], default="sem")
    p.add_argument("--out", default=os.path.join(_REPO, "viz/saturation_curves.png"))
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary = []

    for label, color, arms in GROUPS:
        xs, ys = None, []
        for a in arms:
            k, v = curve(a)
            if xs is None:
                xs = k
            # all runs share the same eval cadence; guard anyway
            v = np.interp(xs, k, v)
            ys.append(smooth(v, args.window))
        Y = np.stack(ys)
        mu = Y.mean(0)
        spread = Y.std(0, ddof=1)
        band = spread / np.sqrt(len(arms)) if args.band == "sem" else spread

        ax.plot(xs / 1000, mu, color=color, lw=2.2, label=f"{label}   (n={len(arms)} seeds)")
        ax.fill_between(xs / 1000, mu - band, mu + band, color=color, alpha=0.18, lw=0)
        for y in Y:                      # faint individual seeds
            ax.plot(xs / 1000, y, color=color, lw=0.6, alpha=0.25)

        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        per_seed = Y[:, sel].mean(1)     # each seed's own window mean
        summary.append((label, per_seed))

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color="0.85", alpha=0.35, zorder=0)
    ax.text(37.5, 0.045, "25-50k window\n(all quoted numbers)", ha="center", va="bottom",
            fontsize=8, color="0.35")

    (la, a), (lb, b) = summary
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    wins = sum(1 for x in a for y in b if x > y)
    ax.set_title("SimDist stage-2 demo collector: still improving vs plateaued\n"
                 "Threading_D0 target, strict success\n"
                 f"25-50k:  {a.mean():.3f} +- {a.std(ddof=1):.3f}   vs   "
                 f"{b.mean():.3f} +- {b.std(ddof=1):.3f}   "
                 f"delta = {d:+.3f}  (z = {d/se:+.2f}, {wins}/{len(a)*len(b)} pairwise)",
                 fontsize=10)
    ax.set_xlabel("stage-3 TARGET transitions collected (k)")
    ax.set_ylabel("TARGET strict success (MPPI)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, xs.max() / 1000)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"saved -> {args.out}", file=sys.stderr)
    for lab, v in summary:
        print(f"  {lab:<42} {np.round(v, 3)}  mean {v.mean():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
