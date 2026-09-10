"""Where the planner's remaining gap lives (Square rotw1, sigma=0.1).

Start from the learned dynamics at a matched search budget and replace the
planner's inputs one at a time until all of them are the oracle / target-fitted
version. Each floating bar is what that ONE swap bought.

Only the dynamics swap clears the noise floor. That is the finding, so it is the
only bar carrying colour -- the greyed bars are not decoration, they are the
"not significant" verdict.

Source: docs/tasks.md 41b. n=200 evaluation episodes per cell, single training
seed. The two anchor bars carry +/- 1 s.e.; the steps carry a z instead, because
a step's uncertainty is the uncertainty of a DIFFERENCE, not of a level.

CAVEAT to keep in the caption: the ordering of the swaps is a choice, not a
fact. value x proposal interact (a better value is worth +0.330 on a weak
proposal and +0.015 on a fine-tuned one), so a different replacement order
redistributes length between the last two bars. The dynamics bar is the one that
is order-robust.

    python plots/waterfall_bottleneck.py       # -> figs/waterfall_bottleneck.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from style import PALETTE as P, WIDTH_1COL, WIDTH_2COL, setup, rounded_bar, clean_axes, save

# ------------------------------------------------------------------ data
LEVELS = [0.765, 0.825, 0.940, 0.955, 0.985]     # cumulative score after each swap
SE     = [0.030, 0.027, 0.017, 0.015, 0.009]     # +/- 1 s.e. of each level
# Every step carries its z, not just the significant one: reporting a test
# statistic for one bar and a verdict ("n.s.") for the others mixes two kinds of
# label, and hides HOW FAR the others are from significance. z >= 2 is the
# emphasis threshold here.
STEPS  = [                                        # label, delta, z, emphasise
    ("search\nbudget",    +0.060, 1.50, False),
    ("oracle\ndynamics",  +0.175, 5.10, True),
    ("target\nvalue",     +0.015, 0.67, False),
    ("target\nproposal",  +0.030, 1.77, False),
]
BASE_LABEL  = "learned\ndynamics"
FINAL_LABEL = "all\nreplaced"
YLO, YHI    = 0.60, 1.075       # truncated axis -- say so in the caption
GAP         = 0.012             # surface gap between a step and the level below
BW          = 0.52             # bar width; leaves air for the x labels
MIN_H       = 0.011             # floor on a step's drawn height, so +0.015 is visible

# ----------------------------------------------------------------- draw
setup(base=8)
fig, ax = plt.subplots(figsize=(WIDTH_2COL * 0.54, 2.5), dpi=200)

rounded_bar(ax, 0, YLO, LEVELS[0], P["anchor"], width=BW)
for i, (_, _, _, emph) in enumerate(STEPS):
    lo, hi = LEVELS[i] + GAP / 2, LEVELS[i + 1] - GAP / 2
    if hi - lo < MIN_H:                     # keep a tiny step visible as a mark
        mid = (lo + hi) / 2
        lo, hi = mid - MIN_H / 2, mid + MIN_H / 2
    rounded_bar(ax, i + 1, lo, hi, P["accent"] if emph else P["neutral"], width=BW)
rounded_bar(ax, len(STEPS) + 1, YLO, LEVELS[-1], P["anchor"], width=BW)

# dashed connectors carry the eye from one cumulative level to the next
for i in range(len(LEVELS)):
    ax.plot([i + BW/2 + 0.02, i + 1 - BW/2 - 0.02], [LEVELS[i]] * 2,
            color=P["axis"], lw=0.7, ls=(0, (2.5, 2)), zorder=2)
ax.plot([len(STEPS) + BW/2 + 0.02, len(STEPS) + 1 - BW/2 - 0.02], [LEVELS[-1]] * 2,
        color=P["axis"], lw=0.7, ls=(0, (2.5, 2)), zorder=2)

# s.e. only on the anchors (a level has one; a difference is reported as z)
for x, lv, se in ((0, LEVELS[0], SE[0]), (len(STEPS) + 1, LEVELS[-1], SE[-1])):
    ax.plot([x, x], [lv - se, lv + se], color=P["surface"], lw=1.8, zorder=4)
    ax.plot([x, x], [lv - se, lv + se], color=P["ink2"], lw=0.8, zorder=5)

# labels: absolute level on the anchors, delta + significance on the steps
for x, lv, se in ((0, LEVELS[0], SE[0]), (len(STEPS) + 1, LEVELS[-1], SE[-1])):
    ax.text(x, lv + se + 0.014, f"{lv:.3f}", ha="center", va="bottom",
            fontsize=8, color=P["ink"])
for i, (_, d, z, emph) in enumerate(STEPS):
    top = LEVELS[i + 1]
    y = max(top, LEVELS[i]) + 0.013
    ax.text(i + 1, y, f"+{d:.3f}", ha="center", va="bottom",
            fontsize=8, color=P["ink"] if emph else P["ink2"],
            fontweight="bold" if emph else "normal")
    ax.text(i + 1, y + 0.036, f"z\u2009=\u2009{z:.1f}", ha="center", va="bottom",
            fontsize=7, color=P["accent"] if emph else P["muted"],
            fontweight="bold" if emph else "normal")

ax.set_xticks(range(len(STEPS) + 2))
ax.set_xticklabels([BASE_LABEL] + [s[0] for s in STEPS] + [FINAL_LABEL],
                   linespacing=1.15)
ax.set_ylabel("target success rate", color=P["ink2"])
ax.set_ylim(YLO, YHI)
ax.set_xlim(-0.62, len(STEPS) + 1.62)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
clean_axes(ax)

fig.tight_layout(pad=0.3)
save(fig, "waterfall_bottleneck")
