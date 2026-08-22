"""Threading_D0 at matched budget: nine adaptation methods on one axis.

Source: exps/simdist_threading_3arm (docs/on-policy-adaptation.md), 60 runs / 9
groups. Every arm gets the SAME 100k target transitions, the same target (the
dynamics-only §12 definition), strict success, sigma=0.1. The SimDist arms all
share one stage-1 world model.

Read-out defaults to `score/score` for ALL nine arms. docs/on-policy-adaptation.md
quotes `score_best` for the three `dynens_*` arms (a max over 4 plan-time
pessimism coefficients, worth +0.014..+0.045 and unavailable to the other six);
the plain score costs those arms a little and removes that asymmetry -- the
ranking is identical either way. `--key score_best` reproduces the docs' table,
falling back to `score` (marked with a *) on the six arms that have no sweep.

Colour = family (dataviz reference palette slots 1/2/3, all-pairs validated:
worst CVD dE 9.2, worst normal dE 24.0). Dash = arm within the family. Every
line is also labelled directly at its end, so identity never rests on colour.

Usage:
    python viz/plot_3arm_adapt.py
    python viz/plot_3arm_adapt.py --key score_best
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(_REPO, "exps/simdist_threading_3arm")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILIES = [
    ("Model-based planning (MPPI on the world model)", BLUE),
    ("Model-based Dyna (same world model, offline)", ORANGE),
    ("Model-free RL", AQUA),
]
# (group prefix, end label, family index, dashes, linewidth)
ARMS = [
    ("dynens_h5_slow_anchor", "dyn-ens + slow + source anchor", 0, (), 2.6),
    ("dynens_h5_slow",        "dyn-ens + slow",                 0, (7, 2.2), 2.0),
    ("dynens_h5",             "dyn-ens",                        0, (3, 2), 2.0),
    ("plain_h3",              "plain SimDist (H=3)",            0, (1.2, 1.8), 2.0),
    ("dyna_h5",               "Dyna",                           1, (), 2.0),
    ("dyna_sgft",             "Dyna + SGFT shaping",            1, (3, 2), 2.0),
    ("ibrl",                  "IBRL",                           2, (), 2.0),
    ("rlpd",                  "RLPD",                           2, (3, 2), 2.0),
    ("iql",                   "IQL",                            2, (1.2, 1.8), 2.0),
]
WINDOW_LO, WINDOW_HI = 75_000, 100_000

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"


def curve(run, key):
    """Series for `key`, falling back to score/score when the arm never logged it
    (only the dyn-ens arms ran a plan-time pessimism sweep, so only they have a
    score_best at all)."""
    with open(os.path.join(EXP, run, "train.log"), errors="ignore") as f:
        txt = f.read()
    for k_try in (key, "score/score"):
        pat = re.compile(rf"^\s*(\d+): {re.escape(k_try)}\s*:\s*([0-9.]+)\s*$", re.M)
        pts = {int(a): float(b) for a, b in pat.findall(txt)}
        if pts:
            ks = np.array(sorted(pts))
            return ks, np.array([pts[i] for i in ks]), k_try


def declutter(ys, gap):
    """Push end labels apart, keeping their order. Ties must separate too, so
    walk the sorted order and force each label at least `gap` above the previous
    one -- comparing against the raw values (as an earlier version did) leaves
    equal ends stacked on top of each other."""
    order = np.argsort(ys, kind="stable")
    out = np.array(ys, dtype=float)
    for prev, i in zip(order, order[1:]):
        out[i] = max(out[i], out[prev] + gap)
    over = out.max() - 1.0
    if over > 0:                                       # top-down relief
        for i in order[::-1]:
            out[i] -= over
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="score/score",
                   choices=["score/score", "score_best", "score/score_best"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    key = "score/score_best" if args.key == "score_best" else args.key
    args.out = args.out or os.path.join(
        _REPO, "viz/threading_3arm_adapt%s.png" % ("_best" if "best" in key else ""))

    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    drawn = []
    for pref, label, fam, dashes, lw in ARMS:
        runs = sorted(d for d in os.listdir(EXP) if re.fullmatch(rf"{pref}_seed\d+", d))
        xs = None
        ys, used = [], set()
        for r in runs:
            k, v, uk = curve(r, key)
            used.add(uk)
            if xs is None:
                xs = k
            ys.append(np.interp(xs, k, v))
        Y = np.stack(ys)
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        color = FAMILIES[fam][1]
        ax.fill_between(xs / 1000, mu - sd, mu + sd, color=color, alpha=0.10, lw=0, zorder=2)
        ax.plot(xs / 1000, mu, color=color, lw=lw, dashes=dashes, zorder=3,
                solid_capstyle="round")
        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        drawn.append(dict(label=label, color=color, end=mu[-1],
                          score=Y[:, sel].mean(1).mean(),
                          sd=Y[:, sel].mean(1).std(ddof=1), n=len(runs),
                          fell_back=(used != {key})))

    # ---- end labels, decluttered -------------------------------------------
    ty = declutter([d["score"] for d in drawn], gap=0.062)
    for d, y in zip(drawn, ty):
        ax.plot([100, 103], [d["end"], y], color=d["color"], lw=0.9, alpha=0.6,
                clip_on=False, zorder=3)
        ax.scatter([100], [d["end"]], s=26, color=d["color"], zorder=5,
                   edgecolor=SURFACE, linewidth=1.5, clip_on=False)
        ax.text(104, y, d["label"] + (" *" if d["fell_back"] else ""),
                fontsize=9.5, color=d["color"],
                va="center", ha="left",
                weight="bold" if d["label"].endswith("source anchor") else "normal")
        ax.text(104, y - 0.030, f"{d['score']:.3f} $\\pm$ {d['sd']:.3f}   n={d['n']}",
                fontsize=8, color=MUTED, va="center", ha="left")

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color=GRIDLINE, alpha=0.55, zorder=0)
    ax.text((WINDOW_LO + WINDOW_HI) / 2000, 0.44,
            "75-100k\nsteady-state\nwindow\n(quoted numbers)",
            ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.5)

    handles = [plt.Line2D([], [], color=c, lw=2.6) for _, c in FAMILIES]
    leg = ax.legend(handles, [n for n, _ in FAMILIES], frameon=False, fontsize=9.5,
                    loc="center left", bbox_to_anchor=(0.13, 0.25), labelcolor=INK2,
                    borderpad=0.2, labelspacing=0.4, handlelength=2.4)
    leg.set_zorder(6)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("target transitions collected (k)   -- matched budget for all nine arms",
                  fontsize=10, color=INK2)
    ax.set_ylabel("target success rate (100 eval episodes per point)", fontsize=10, color=INK2)
    ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    fig.suptitle("Threading_D0 sim2sim: nine adaptation methods at a matched 100k-transition budget"
                 + ("   [score_best]" if "best" in key else ""),
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.928,
             "Same target domain (dynamics-only gap), strict success, $\\sigma$=0.1; all SimDist arms share one stage-1 world model. "
             "Shaded band = $\\pm$1 sd across seeds.",
             fontsize=9, color=INK2, ha="left")
    if "best" in key:
        notes = [
            "Read-out is score_best = max over the arm's plan-time pessimism sweep, which reproduces the docs' table but gives the three dyn-ens arms a selection bias of +0.014..+0.045.",
            "* the other six arms ran no sweep and have no score_best -- their score/score is shown, so only the dyn-ens curves differ from the score version of this figure.",
        ]
    else:
        notes = [
            "Read-out is score/score for all nine arms; the docs quote score_best for the dyn-ens arms (a max over 4 pessimism coefficients, +0.014..+0.045).",
            "The other six cannot take that max, so the plain score is used here -- the ranking is identical either way.",
        ]
    notes.append("Model-free arms carry 10 warm-up episodes outside the 100k budget.   One task, one stage-1 world model.")
    for i, n in enumerate(notes):
        fig.text(0.008, 0.010 + 0.019 * (len(notes) - 1 - i), n,
                 fontsize=7.6, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.032 + 0.019 * len(notes), 0.735, 0.90])
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out, "| key =", key)
    for d in drawn:
        print(f"  {d['label']:32s} n={d['n']:2d}  75-100k = {d['score']:.3f} +- {d['sd']:.3f}")


if __name__ == "__main__":
    main()
