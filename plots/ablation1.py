"""Ablation 1: SimADEPT minus each of its two additions (Threading_D0).

Same stage-1 checkpoint and same stage-3 budget throughout; the arms differ only
in which of the two SimADEPT modifications is switched on.

    dynens_h5_slow_anchor   ensemble + tethering   SimADEPT              (deep)
    dynens_h5_slow          ensemble only          w/o tethering         (mid)
    plain_anch              tethering only         w/o ensemble          (mid2)
    plain_h5                neither                w/o ensemble & teth.  (light)

One hue, four lightness tiers ordered by how much of the method survives: these
are the same method with pieces removed, so a second hue would claim a
difference in kind. The tier split between the two SINGLE ablations is
arbitrary -- they are peers, and nothing in the figure should suggest a rank
between them before the reader has looked at the curves. Only ours carries a
triangle. Format follows plots/tether.py.

Data
----
W&B cao1233/simdist_threading_3arm. x = adapt/transitions, y = score/score,
except x=0 which is score/policy_only -- the proposal executed WITHOUT planning.
Those are two different quantities (see tether.py); here it also means the arms
start as TWO PAIRS, because at x=0 the ensemble has not been used yet and only
the tethering can have moved the proposal. That is a real property, not an
artefact, and it is the cleanest way to see that tethering acts before the first
transition while the ensemble acts after.

  * dynens_h5_slow_anchor drops seed 401 as instructed -> n=9; the other three
    arms are n=10.
  * `plain_h5` was logged twice under one group name; the Sep-10 batch is the
    one with all 10 seeds (see tether.py). created_after picks it.
  * VINTAGE MISMATCH: the two dynens arms ran 2026-08-09/13, the two plain arms
    2026-09-10/11. Same project and same task, but a month apart, so anything
    that changed in the trainer in between is confounded with the ensemble
    factor. `dynens_h5_slow` predates bc_anchor_coef existing at all (the key is
    absent from its config, i.e. 0 by default). Worth a line in the caption.
  * score/score is the in-training eval (100 episodes/point) -- CLAUDE.md rule
    14 flags it as optimistic; read the spacing, not the level.
  * Band is style.BAND_TYPE over seeds -- currently +/- 1 s.d., the same
    quantity ceiling_3task and tether draw. A s.d. band does not shrink with n
    and is ~3x wider than a s.e. band, so overlap is not evidence of no
    difference; the seed noise floor on this task is ~3.5 points.

    python plots/ablation1.py              # uses plots/cache/
    python plots/ablation1.py --refresh    # re-pull from W&B
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from style import FAMILIES, PALETTE, setup, clean_axes, save, band_of, BAND_TYPE
from wandb_fetch import fetch_group

ICLR_TEXTWIDTH = 5.5

PROJECT = "simdist_threading_3arm"
XKEY = "adapt/transitions"
YKEY, YKEY0 = "score/score", "score/policy_only"
ZERO_FROM_POLICY_ONLY = True
XMAX = 50_000

SAMPLE_AT = [0, 10_000, 20_000, 30_000, 40_000, 50_000]
MARK_AT = SAMPLE_AT
VGRID_AT = [0, 10_000, 20_000, 30_000, 40_000]
XPAD, YPAD = 2_200, 3.0

BLUE = FAMILIES["blue"]
AQUA = FAMILIES["aqua"]
# A fourth tier between mid and light. style.py ships three per family because
# three is what a family figure normally needs; a 2x2 ablation needs four, and
# inventing a fourth HUE for it would be the wrong fix. L* = 23/51/62/76.
BLUE_MID2 = "#7698C3"

ARMS = [                       # group, label, colour, lw, marker, ms, n, excl
    ("plain_h5",  "SimDist", AQUA["deep"], 1.2, "o", 3.5, 10, ()),
    ("plain_anch", "w/o ensemble",            BLUE["light"],     1.2, "o", 3.5, 10, ()),
    ("dynens_h5_slow", "w/o tethering",       BLUE["mid"],   1.2, "o", 3.5, 10, ()),
    ("dynens_h5_slow_anchor", "SimADEPT",     BLUE["deep"],  1.2, "^", 3.8, 9, (401,)),
]
# Bottom-to-top the key adds components back, so ours sits at the bottom next to
# the curve it labels -- same placement as tether.py.
LEGEND_ORDER = ["SimDist", "w/o ensemble", "w/o tethering",
                "SimADEPT"]
CREATED_AFTER = {"plain_h5": "2026-09-10"}


def load(group, refresh, n_expect, exclude):
    runs = fetch_group(PROJECT, group, XKEY, [YKEY, YKEY0], refresh=refresh,
                       created_after=CREATED_AFTER.get(group))
    for seed in exclude:
        if seed not in runs:
            raise RuntimeError(f"{group}: asked to exclude seed {seed}, "
                               f"which is not in the group ({sorted(runs)})")
        del runs[seed]
    if len(runs) != n_expect:
        raise RuntimeError(f"{group}: expected {n_expect} seeds, got "
                           f"{len(runs)} ({sorted(runs)})")

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
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figs"))
    args = ap.parse_args()

    setup(base=8)
    fig, ax = plt.subplots(figsize=(0.4 * ICLR_TEXTWIDTH, 2), dpi=200)

    for xv in VGRID_AT:
        ax.axvline(xv, color=PALETTE["axis"], lw=0.5, alpha=0.55, zorder=0)

    handles = {}
    for group, label, color, lw, marker, ms, n_exp, excl in ARMS:
        x, ys = load(group, args.refresh, n_exp, excl)
        ys = ys * 100.0
        mean = ys.mean(axis=0)
        band = band_of(ys, BAND_TYPE)

        ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, lw=lw, zorder=3, solid_capstyle="round")
        mk = np.isin(x, MARK_AT)
        marker_kw = dict(marker=marker, markersize=ms, markerfacecolor="white",
                         markeredgecolor=color, markeredgewidth=lw)
        ax.plot(x[mk], mean[mk], linestyle="none", zorder=4, clip_on=False,
                **marker_kw)
        handles[label] = plt.Line2D([], [], color=color, lw=lw, **marker_kw)
        print(f"{group:22s} n={ys.shape[0]:2d}  zero-shot {mean[0]:5.1f}%  "
              f"final {mean[-1]:5.1f}% +/- {band[-1]:.1f}  "
              f"best {mean.max():5.1f}% @ {x[mean.argmax()] / 1000:.0f}k")

    ax.set_xlabel(r"Environment Step ($\times 10^{3}$)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(-XPAD, XMAX + XPAD)
    ax.set_ylim(-YPAD, 100 + YPAD)
    ax.set_xticks([0, 10_000, 20_000, 30_000, 40_000, 50_000])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    clean_axes(ax)
    # Four labels, three of them long, in a 0.4*textwidth panel: at tether.py's
    # legend metrics the key covers ~40% of the axes. Everything below is
    # tightened rather than the labels being shortened -- the label is the
    # experiment's name and should not be abbreviated to save ink.
    leg = ax.legend([handles[k] for k in LEGEND_ORDER], LEGEND_ORDER,
                    loc="lower right", frameon=True, handlelength=1.3,
                    handletextpad=0.5, borderaxespad=0.3, labelspacing=0.22,
                    borderpad=0.35, fontsize=6)
    fr = leg.get_frame()
    fr.set_linewidth(0.6)
    fr.set_edgecolor(PALETTE["axis"])
    fr.set_facecolor(PALETTE["surface"])
    leg.set_zorder(5)

    save(fig, "ablation1", outdir=args.outdir)


if __name__ == "__main__":
    main()
