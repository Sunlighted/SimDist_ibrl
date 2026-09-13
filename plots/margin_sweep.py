"""Dyna-IBRL: what the prior-preference margin m buys, and through which channel.

Two stacked panels sharing the m axis:

    top     target success rate at 100k vs m                   -- the outcome
    bottom  fraction of steps the learned actor wins the argmax
            in the COLLECTION channel, vs m                    -- the mechanism

The point of pairing them is that m acts MONOTONICALLY on who wins the argmax
while the outcome is non-monotone: the same low score can come from the actor
taking over before the critic can rank it (m small, win rate high) or from the
actor never taking over at all (m large, win rate 0). A single score-vs-m curve
cannot tell those two apart -- they are both just "a low point on the U".

The collect/backup SPLIT is panel (b)'s job (Tab. tab:margin), so only the
collection channel is drawn here. `dyna/bootstrap_rl_frac` is fetched anyway and
printed, because at m=0 it runs 14 points ABOVE collection (68.2 vs 53.7) while
at m=0.01 the two coincide -- i.e. the backup channel is the one that fails
first. Worth a sentence in (b) even though it is not plotted here.

x is CATEGORICAL, not log: m=0 is one of the levels and cannot sit on a log
axis, and faking one would put a meaningless distance between the levels.

STATUS -- HALF THIS FIGURE IS PLACEHOLDER. DO NOT SHIP AS IS.
-------------------------------------------------------------
Only m in {0, 0.01} exists as a fixed-margin arm in W&B. `dynaibrl_plain`
(unsuffixed) is NOT a third level: its dyna/prior_margin ramps 0 -> 0.05 during
training, so it has no single m to plot against.

m = 0.05 and m = 0.1 are therefore taken from Tab. tab:margin (a), whose entries
are all inside \ph{} -- i.e. the paper itself marks them as placeholders, from
seed 1 only. They are drawn in grey with a dashed connector so that a
provisional point can never be mistaken for a measured one, and the script
prints a warning naming them.

Mixing the two provenances changes the SHAPE, not just the error bars:

    m       SR@100k        source
    0        36.0 +/- 22.6  W&B, 10 seeds      (table placeholder said 0.00)
    0.01     ~73            W&B, 10 seeds      (table placeholder said 0.87)
    0.05     77.0           table placeholder, seed 1
    0.1      62.0           table placeholder, seed 1

With real seeds at the two left levels the peak moves to m=0.05, contradicting
the text's "m=0.01 is the best". That is an artefact of comparing a 10-seed mean
against a 1-seed number, not a finding. Run m=0.05 and m=0.1 at 10 seeds before
this panel says anything.

Data
----
W&B cao1233/simdist_threading_3arm, x = adapt/transitions.
  * min_step=300 keeps the long re-run of each seed; both groups then carry 10
    seeds. The earlier 100k batch under the same group name is dropped.
  * Each level is the seed mean at REPORT_AT = 100k exactly. A single eval
    point carries the +/-6 point noise of a 100-episode eval (CLAUDE.md rule
    15); WINDOW = (80_000, 100_000) averages it down if you want that back.
  * score/score is the in-training eval (100 episodes/point) and is optimistic
    in level (rule 14); read the spacing between levels.
  * Error bars are style.BAND_TYPE over seeds, same as every other figure here.

    python plots/margin_sweep.py              # uses plots/cache/
    python plots/margin_sweep.py --refresh    # re-pull from W&B
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
SCORE = "score/score"
WIN_COLLECT, WIN_BACKUP = "dyna/collect_rl_frac", "dyna/bootstrap_rl_frac"
MIN_STEP = 300
REPORT_AT = 100_000                     # single eval point, per instruction
WINDOW = None                           # or (80_000, 100_000) to average
N_SEEDS = 10

# (m label, W&B group or None, placeholder SR %, placeholder collect win %).
# A group of None means "no run exists" and the placeholder is used; extend or
# replace as the missing arms land. Axis is categorical, so levels are drawn
# evenly spaced in the order listed here.
LEVELS = [
    ("0",    "dynaibrl_plain_m0",   None, None),
    ("0.01", "dynaibrl_plain_m001", None, None),
    ("0.05", None,                  77.0, 1.0),    # Tab. tab:margin (a), \ph{}
    ("0.1",  None,                  62.0, 0.0),    # Tab. tab:margin (a), \ph{}
]

BLUE = FAMILIES["blue"]
C_MEASURED = BLUE["deep"]
LW, MS = 1.2, 3.8


def level_stats(group, refresh):
    """Per-seed score and win rates at REPORT_AT (or averaged over WINDOW)."""
    runs = fetch_group(PROJECT, group, XKEY, [SCORE, WIN_COLLECT, WIN_BACKUP],
                       refresh=refresh, min_step=MIN_STEP)
    if len(runs) != N_SEEDS:
        raise RuntimeError(f"{group}: expected {N_SEEDS} seeds, got "
                           f"{len(runs)} ({sorted(runs)})")
    out = {k: [] for k in (SCORE, WIN_COLLECT, WIN_BACKUP)}
    for seed in sorted(runs):
        x = np.asarray(runs[seed][XKEY], dtype=float)
        sel = ((x >= WINDOW[0]) & (x <= WINDOW[1]) if WINDOW
               else (x == REPORT_AT))
        if not sel.any():
            raise RuntimeError(f"{group} seed {seed}: no eval point at "
                               f"{WINDOW or REPORT_AT}")
        for k in out:
            out[k].append(np.asarray(runs[seed][k], dtype=float)[sel].mean())
    return {k: np.asarray(v) * 100.0 for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figs"))
    args = ap.parse_args()

    pos = np.arange(len(LEVELS), dtype=float)
    sr, sr_band, win, measured, placeholders = [], [], [], [], []
    for m, group, ph_sr, ph_win in LEVELS:
        if group is None:
            sr.append(ph_sr)
            sr_band.append(np.nan)
            win.append(ph_win)
            measured.append(False)
            placeholders.append(m)
            print(f"m={m:>5s}  [PLACEHOLDER Tab.tab:margin, seed 1]   "
                  f"score {ph_sr:5.1f}            win rate: collect {ph_win:5.1f}%")
            continue
        st = level_stats(group, args.refresh)
        s_mean = st[SCORE].mean()
        s_band = band_of(st[SCORE][:, None], BAND_TYPE)[0]
        sr.append(s_mean)
        sr_band.append(s_band)
        win.append(st[WIN_COLLECT].mean())
        measured.append(True)
        print(f"m={m:>5s}  {group:20s} n={N_SEEDS}  score {s_mean:5.1f} "
              f"+/- {s_band:4.1f}   win rate: collect "
              f"{st[WIN_COLLECT].mean():5.1f}%  backup {st[WIN_BACKUP].mean():5.1f}%")
    if placeholders:
        print(f"\n!! PLACEHOLDER LEVELS m={placeholders}: single-seed numbers "
              f"copied from Tab. tab:margin (a), which the paper itself marks "
              f"\\ph{{}}. Drawn grey/dashed. Do not ship this panel until they "
              f"are run at {N_SEEDS} seeds -- see the module docstring for why "
              f"the mixture changes the SHAPE of the curve.")

    sr, sr_band, win = np.array(sr), np.array(sr_band), np.array(win)
    measured = np.array(measured)

    setup(base=8)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=(0.4 * ICLR_TEXTWIDTH, 2.7), dpi=200,
        gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.18))

    def draw(ax, y, band=None):
        # One dashed grey segment wherever either endpoint is provisional, so
        # the eye never traces a solid line through an unmeasured point.
        for i in range(len(pos) - 1):
            solid = measured[i] and measured[i + 1]
            ax.plot(pos[i:i + 2], y[i:i + 2],
                    color=C_MEASURED if solid else PALETTE["muted"],
                    lw=LW, ls="-" if solid else (0, (2.2, 1.6)), zorder=3)
        for i, ok in enumerate(measured):
            col = C_MEASURED if ok else PALETTE["muted"]
            if band is not None and ok and np.isfinite(band[i]):
                ax.errorbar(pos[i], y[i], yerr=band[i], color=col,
                            elinewidth=LW * 0.8, capsize=2, capthick=LW * 0.8,
                            zorder=3, fmt="none")
            ax.plot(pos[i], y[i], marker="o", markersize=MS,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=LW, linestyle="none", zorder=4,
                    clip_on=False)

    draw(ax0, sr, sr_band)
    draw(ax1, win)

    ax0.set_ylabel("Success Rate (%)")
    ax0.set_ylim(-3, 103)
    ax0.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Actor Win Rate (%)")
    ax1.set_ylim(-3, 103)
    ax1.set_yticks([0, 50, 100])
    ax1.set_xlabel(r"Prior-preference margin $m$")
    ax1.set_xlim(-0.35, len(LEVELS) - 0.65)
    ax1.set_xticks(pos)
    ax1.set_xticklabels([m for m, _, _, _ in LEVELS])

    for ax in (ax0, ax1):
        for pp in pos:
            ax.axvline(pp, color=PALETTE["axis"], lw=0.5, alpha=0.55, zorder=0)
        clean_axes(ax)

    save(fig, "margin_sweep", outdir=args.outdir)


if __name__ == "__main__":
    main()
