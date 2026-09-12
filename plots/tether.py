"""Threading_D0: what the stage-2 proposal warm-up is worth (plain dynamics).

Three arms, identical single-head SimDist stage-1 checkpoint, identical stage-3
dynamics adaptation, 10 seeds each, differing ONLY in how the MPPI proposal is
warmed up on the 20 target demos:

    plain_anch   bc_steps=1000, bc_anchor_coef=1   tethered fine-tuning   (deep)
    plain_nobc   bc_steps=0                        frozen proposal        (mid)
    plain_h5     bc_steps=1000, bc_anchor_coef=0   untethered fine-tuning (light)

One family, three lightness tiers: the arms differ in a single knob, so spending
three hues on them would claim a difference in kind that is not there. Deep =
the arm we keep, and it is the only one carrying a triangle marker.

Data
----
W&B cao1233/simdist_threading_3arm. x = adapt/transitions, logged every 5k
transitions (eval_every_rounds=5), 100 episodes per eval point.

  * y = score/score (the MPPI planner) at every x > 0, but y = score/policy_only
    (the proposal executed WITHOUT planning) at x = 0. THESE ARE TWO DIFFERENT
    QUANTITIES. The x=0 point answers "what did the demos buy the proposal",
    the rest of the curve answers "what does the planner reach", so the step
    between x=0 and x=5k is partly the planner switching on, not adaptation.
    Both keys exist at x=0 -- score/score there is 0.70 / 0.51 / 0.35 for
    anch / nobc / h5 versus policy_only's 0.63 / 0.20 / 0.42 -- so this is a
    deliberate choice, not a gap in the logs. Say so in the caption, or set
    ZERO_FROM_POLICY_ONLY = False for a single-metric curve.
  * `plain_h5` was run TWICE under the same group name. The Sep-02 batch has a
    crashed seed (301, died at 15k) so it only carries 9; the Sep-10 batch
    carries all 10 and is contemporaneous with the other two arms, which is the
    one this figure uses. Nothing in the run names distinguishes them -- hence
    the created_after filter. The Sep-10 batch ran to 300k; every arm is
    truncated to 50k here (all three logged at least that far).
  * score/score is the in-training eval (100 episodes), which CLAUDE.md rule 14
    flags as systematically optimistic vs an independent large-sample re-eval.
    Read the SPACING between arms, not the absolute level; headline numbers in
    the text should come from evaluate/eval.py.
  * Band is +/- 1 s.e. over the 10 seeds. Per CLAUDE.md rule 1 the seed noise
    floor on this task is ~0.035, so bands that overlap are not a difference.

    python plots/threading_3arm.py                 # uses plots/cache/
    python plots/threading_3arm.py --refresh       # re-pull from W&B
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from style import (FAMILIES, PALETTE, WIDTH_1COL, setup, clean_axes,
                   save)
from wandb_fetch import fetch_group

ICLR_TEXTWIDTH = 5.5

PROJECT = "simdist_threading_3arm"
XKEY = "adapt/transitions"
YKEY, YKEY0 = "score/score", "score/policy_only"
ZERO_FROM_POLICY_ONLY = True    # see the Data note above
XMAX = 50_000
N_SEEDS = 10

# The runs log every 5k, but the figure reports every 10k: at 5k spacing the
# per-point eval noise (100 episodes, so a s.e. of ~5 points on a single arm)
# dominates the line and the arms weave through each other. Subsampling is a
# claim -- that the 5k detail is noise, not signal -- and it is the right claim
# here, but it means the curve is NOT a running maximum or an average over the
# skipped points: it is the 10k evals, drawn as they were measured.
SAMPLE_AT = [0, 10_000, 20_000, 30_000, 40_000, 50_000]
MARK_AT = SAMPLE_AT                             # including x=0, see the note above
# x=0 gets its own rule: once the axes are padded, the left spine no longer
# stands at 0, so without it there is nothing in the plot saying where zero is.
VGRID_AT = [0, 10_000, 20_000, 30_000, 40_000]
# Breathing room so no mark is bisected by a spine and x=0 reads as a measured
# point rather than as the axis itself. The pad is in data units and is NOT a
# claim that anything was measured outside [0, 50k]; the ticks still stop there.
XPAD, YPAD = 2_200, 3.0

BLUE = FAMILIES["blue"]
# Drawn light -> deep so ours sits on top. LEGEND_ORDER re-sorts the key into
# reading order (what you do not do, what goes wrong, what we do) -- z-order is
# a drawing concern and should not dictate how the key reads.
ARMS = [                          # group, label, colour, lw, marker, markersize
    ("plain_h5",   "untethered fine-tuning", BLUE["mid"], 1.2, "o", 3.0),
    ("plain_nobc", "frozen proposal",        BLUE["light"],   1.2, "o", 3.0),
    ("plain_anch", "tethered fine-tuning",   BLUE["deep"],  1.2, "^", 3.5),
]
LEGEND_ORDER = ["frozen proposal", "untethered fine-tuning",
                "tethered fine-tuning"]
# plain_h5 lives twice under one group name -- see the module docstring.
CREATED_AFTER = {"plain_h5": "2026-09-10"}


def load(group, refresh):
    runs = fetch_group(PROJECT, group, XKEY, [YKEY, YKEY0], refresh=refresh,
                       created_after=CREATED_AFTER.get(group))
    if len(runs) != N_SEEDS:
        raise RuntimeError(
            f"{group}: expected {N_SEEDS} seeds, got {len(runs)} "
            f"({sorted(runs)}) -- check the created_after filter")

    # All arms log on the same 5k grid, but assert it rather than assume it:
    # a silently misaligned x would average across different budgets.
    grid, ys = None, []
    for seed in sorted(runs):
        x = np.asarray(runs[seed][XKEY], dtype=float)
        y = np.asarray(runs[seed][YKEY], dtype=float)
        if ZERO_FROM_POLICY_ONLY:
            y0 = np.asarray(runs[seed][YKEY0], dtype=float)
            at0 = x == 0
            if not at0.any():
                raise RuntimeError(f"{group} seed {seed}: no x=0 row to splice")
            y = np.where(at0, y0, y)
        keep = np.isin(x, SAMPLE_AT) & (x <= XMAX)
        x, y = x[keep], y[keep]
        if grid is None:
            grid = x
        elif not np.array_equal(grid, x):
            raise RuntimeError(f"{group} seed {seed}: x grid differs from seed "
                               f"{sorted(runs)[0]}")
        ys.append(y)
    return grid, np.vstack(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull from W&B instead of using plots/cache/")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figs"))
    args = ap.parse_args()

    setup(base=8)
    fig, ax = plt.subplots(figsize=(0.4*ICLR_TEXTWIDTH, 2), dpi=200)

    for xv in VGRID_AT:
        ax.axvline(xv, color=PALETTE["axis"], lw=0.5, alpha=0.55, zorder=0)

    handles = {}
    for group, label, color, lw, marker, ms in ARMS:
        x, ys = load(group, args.refresh)
        ys = ys * 100.0                     # success rate in percent
        mean = ys.mean(axis=0)
        se = ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])

        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, lw=lw, zorder=3, solid_capstyle="round")
        # Open markers, stroked at the line's own weight in the line's own
        # colour: the mark reads as the same object as the curve rather than a
        # second encoding, and a hollow centre keeps the three arms legible
        # where they cross.
        mk = np.isin(x, MARK_AT)
        marker_kw = dict(marker=marker, markersize=ms, markerfacecolor="white",
                         markeredgecolor=color, markeredgewidth=lw)
        ax.plot(x[mk], mean[mk], linestyle="none", zorder=4, clip_on=False,
                **marker_kw)
        # One proxy artist so the key shows the line AND its marker.
        handles[label] = plt.Line2D([], [], color=color, lw=lw, **marker_kw)
        print(f"{group:11s} n={ys.shape[0]:2d}  zero-shot {mean[0]:.1f}%  "
              f"final {mean[-1]:.1f}% +/- {se[-1]:.1f}  "
              f"best {mean.max():.1f}% @ {x[mean.argmax()] / 1000:.0f}k")

    ax.set_xlabel(r"Environment Step ($\times 10^{3}$)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(-XPAD, XMAX + XPAD)
    ax.set_ylim(-YPAD, 100 + YPAD)
    ax.set_xticks([0, 10_000, 20_000, 30_000, 40_000, 50_000])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    clean_axes(ax)
    # The two spines span the full padded range so they close at the bottom-left
    # corner. Bounding them to [0, XMAX] / [0, 100] instead leaves the corner
    # open, which reads as a rendering fault rather than as deliberate air.
    leg = ax.legend([handles[k] for k in LEGEND_ORDER], LEGEND_ORDER,
                    loc="lower right", frameon=True, handlelength=1.8,
                    borderaxespad=0.4, labelspacing=0.35, borderpad=0.5)
    fr = leg.get_frame()
    fr.set_linewidth(0.6)
    fr.set_edgecolor(PALETTE["axis"])       # same weight/colour as the spines
    fr.set_facecolor(PALETTE["surface"])    # opaque: the key sits over gridlines
    leg.set_zorder(5)

    save(fig, "threading_3arm", outdir=args.outdir)


if __name__ == "__main__":
    main()
