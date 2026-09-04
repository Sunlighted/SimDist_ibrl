"""Threading_D0: Dyna-IBRL vs MPPI planning, on the SAME dyn-ens world model.

Both families adapt from one stage-1 checkpoint
(`exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1`), each seed uses its
own 20 target demos (`threading_d0_sat_160000_s<seed>`), the same 100k
target-transition budget, 100 eval episodes per point, strict success, sigma=0.1.
The only difference is what CONSUMES the world model at eval time:

  * `dynens_h5_slow[_anchor]` -- MPPI planning over the learned dynamics
    (`score/score` = the planner's score).
  * `dynaibrl_dynens`         -- Dyna: 1-step imagined rollouts train a latent
    actor/critic; eval runs the IBRL hybrid argmax_a Q(z,a) over {actor, frozen
    distilled prior} with a margin annealed 0.05 -> 0 over 50k. NO planner at
    eval (`train_simdist_dyna_ibrl._eval_mppi` overrides the parent's
    `use_planner=1`), so this arm is ~1 network forward per step.

Data comes from the wandb cache, because the `dynaibrl_dynens` seeds ran on the
cluster and never landed in a local `exps/` dir:

    python viz/fetch_wandb_curves.py        # refresh viz/_cache/*.json

Usage:
    python viz/plot_dynaibrl_vs_dynens.py
    python viz/plot_dynaibrl_vs_dynens.py --sweep     # the prior-margin sweep

⚠️ The `--sweep` figure mixes sources: those margin variants are the SINGLE-seed
workstation runs (local `exps/`), and their demo set is `sat_160000_s1` for every
one of them. They are what made "Dyna-IBRL ties planning" look true; with 9 seeds
the cluster arm lands 0.15 below planning. Read them as a knob sweep, not as
estimates.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(_REPO, "exps/simdist_threading_3arm")
CACHE = os.path.join(_REPO, "viz/_cache/simdist_threading_3arm.json")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"

# (wandb group, end label, colour, dashes, linewidth)
ARMS = [
    ("dynens_h5_slow_anchor", "MPPI planning + source anchor", BLUE, (), 2.6),
    ("dynens_h5_slow",        "MPPI planning",                 BLUE, (7, 2.2), 2.2),
    ("dynaibrl_dynens",       "Dyna-IBRL hybrid, no planner",  ORANGE, (), 2.6),
]
# The workstation prior-margin sweep: single seed each, all on sat_160000_s1,
# read out of the local train.log (never synced to wandb).
SWEEP_LOCAL = [
    ("dynaibrl_m001anch_seed1_crashed85k", "m=0.01 const + source anchor"),
    ("dynaibrl_m005anneal_seed1", "m: 0.05$\\to$0.01"),
    ("dynaibrl_m001_seed1", "m=0.01 const"),
    ("dynaibrl_m005anneal0_seed1", "m: 0.05$\\to$0"),
    ("dynaibrl_m005_seed1", "m=0.05 const"),
    ("dynaibrl_m001rr100_seed1", "m=0.01, real_ratio=1.0"),
    ("dynaibrl_m01_seed1", "m=0.1 const"),
    ("dynaibrl_m001sgft_seed1", "m=0.01 + SGFT shaping"),
    ("dynaibrl_m0demo_seed1", "m=0, +demos"),
    ("dynaibrl_m0_seed1", "m=0"),
]
WINDOW_LO, WINDOW_HI = 75_000, 100_000


def group(name):
    """(x, Y[seed, point]) for one wandb group, from the local cache."""
    with open(CACHE) as f:
        runs = json.load(f)[name]
    names = sorted(runs, key=lambda s: int(s.split("seed")[-1]))
    xs, ys = None, []
    for n in names:
        r = runs[n]
        x = np.array(r["x"], dtype=float)
        v = np.array([np.nan if a is None else a for a in r["score"]], dtype=float)
        if len(x) < 21:                       # a seed that died early is not a seed
            print(f"  [skip] {n}: only {len(x)} points, last x={x[-1]:.0f}")
            continue
        if xs is None:
            xs = x
        ys.append(np.interp(xs, x, v))
    return xs, np.stack(ys)


def local_curve(run, key="score/score"):
    with open(os.path.join(EXP, run, "train.log"), errors="ignore") as f:
        txt = f.read()
    pat = re.compile(rf"^\s*(\d+): {re.escape(key)}\s*:\s*([0-9.]+)\s*$", re.M)
    pts = {int(a): float(b) for a, b in pat.findall(txt)}
    ks = np.array(sorted(pts))
    return ks, np.array([pts[i] for i in ks])


def declutter(ys, gap):
    """Push end labels apart while keeping their order (ties separate too)."""
    order = np.argsort(ys, kind="stable")
    out = np.array(ys, dtype=float)
    for prev, i in zip(order, order[1:]):
        out[i] = max(out[i], out[prev] + gap)
    over = out.max() - 1.0
    if over > 0:
        for i in order[::-1]:
            out[i] -= over
    return out


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se, se


def line_figure(out):
    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    drawn, per_seed = [], {}
    for name, label, color, dashes, lw in ARMS:
        xs, Y = group(name)
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(xs / 1000, mu - sd, mu + sd, color=color, alpha=0.10,
                        lw=0, zorder=2)
        ax.plot(xs / 1000, mu, color=color, lw=lw, dashes=dashes, zorder=3,
                solid_capstyle="round")
        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        w = Y[:, sel].mean(1)
        per_seed[name] = w
        drawn.append(dict(label=label, color=color, end=mu[-1], score=w.mean(),
                          sd=w.std(ddof=1), n=len(Y)))

    # ---- end labels, decluttered -------------------------------------------
    ty = declutter([d["score"] for d in drawn], gap=0.075)
    for d, y in zip(drawn, ty):
        ax.plot([100, 103], [d["end"], y], color=d["color"], lw=0.9, alpha=0.6,
                clip_on=False, zorder=3)
        ax.scatter([100], [d["end"]], s=26, color=d["color"], zorder=5,
                   edgecolor=SURFACE, linewidth=1.5, clip_on=False)
        ax.text(104, y, d["label"], fontsize=9.5, color=d["color"], va="center",
                ha="left", weight="bold" if d["color"] == ORANGE else "normal")
        ax.text(104, y - 0.034,
                f"{d['score']:.3f} $\\pm$ {d['sd']:.3f}   n={d['n']} seeds",
                fontsize=8, color=MUTED, va="center", ha="left")

    t, se = welch(per_seed["dynens_h5_slow"], per_seed["dynaibrl_dynens"])
    diff = per_seed["dynens_h5_slow"].mean() - per_seed["dynaibrl_dynens"].mean()
    ax.annotate("", xy=(100, per_seed["dynens_h5_slow"].mean()),
                xytext=(100, per_seed["dynaibrl_dynens"].mean()),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(98.4, (per_seed["dynens_h5_slow"].mean() + per_seed["dynaibrl_dynens"].mean()) / 2 - 0.008,
            f"planning $-$ Dyna-IBRL\n= {diff:+.3f}   (Welch t = {t:.1f})",
            fontsize=8.5, color=INK2, ha="right", va="center", linespacing=1.4,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color=GRIDLINE, alpha=0.55, zorder=0)
    ax.text((WINDOW_LO + WINDOW_HI) / 2000, 0.30,
            "75-100k\nsteady-state\nwindow\n(quoted numbers)",
            ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.5)

    handles = [plt.Line2D([], [], color=BLUE, lw=2.6),
               plt.Line2D([], [], color=ORANGE, lw=2.6),
               plt.Line2D([], [], color=INK2, lw=2.2, dashes=()),
               plt.Line2D([], [], color=INK2, lw=2.0, dashes=(7, 2.2))]
    labels = ["MPPI planning over the world model at eval time",
              "Dyna: latent actor+critic trained on imagined rollouts, IBRL hybrid at eval",
              "stage-2 source anchor ON  (bc_anchor_coef=1)",
              "stage-2 source anchor OFF  (bc_anchor_coef=0)"]
    leg = ax.legend(handles, labels, frameon=False, fontsize=9, loc="lower right",
                    bbox_to_anchor=(0.995, 0.02), labelcolor=INK2,
                    borderpad=0.2, labelspacing=0.45, handlelength=2.4)
    leg.set_zorder(6)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("target transitions collected (k)   -- matched 100k budget for every arm",
                  fontsize=10, color=INK2)
    ax.set_ylabel("target success rate (100 eval episodes per point)", fontsize=10, color=INK2)
    ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    fig.suptitle("Threading_D0 sim2sim: Dyna-IBRL vs MPPI planning on the SAME dyn-ens world model",
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.928,
             "One stage-1 ckpt (threading_simdist_dynens_diversev2_h5_seed1); per-seed target demos (sat_160000_s<seed>); dynamics-only gap, "
             "strict success, $\\sigma$=0.1. Band = $\\pm$1 sd across seeds.",
             fontsize=9, color=INK2, ha="left")
    notes = [
        "The Dyna-IBRL arm anneals its prior margin 0.05 -> 0 over 50k and runs NO planner at eval -- it is ~1 forward pass per step, against the planner's full MPPI search.",
        "So the 0.15 it gives up is the price of that speed, and the anchored planner is 0.24 above it. Both planning arms are n=10; the Dyna arm is n=9 (seed 901 is missing).",
        "Source: viz/_cache/simdist_threading_3arm.json (wandb) -- the Dyna-IBRL seeds ran on the cluster and have no local exps/ dir.",
    ]
    for i, n in enumerate(notes):
        fig.text(0.008, 0.010 + 0.019 * (len(notes) - 1 - i), n,
                 fontsize=7.6, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.032 + 0.019 * len(notes), 0.735, 0.90])
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)
    for d in drawn:
        print(f"  {d['label']:34s} n={d['n']:2d}  75-100k = {d['score']:.3f} +- {d['sd']:.3f}")
    print(f"  planning - dyna-ibrl = {diff:+.3f}  (se {se:.3f}, Welch t = {t:.2f})")


def sweep_figure(out):
    """Ranked dot plot: the prior-margin knob, against the two planning lines.

    Overlaying ten extra single-seed curves on the line figure made an unreadable
    tangle of end labels; ranking them carries the same information."""
    rows = []
    xs, Y = group("dynaibrl_dynens")
    sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
    w = Y[:, sel].mean(1)
    rows.append(("m: 0.05$\\to$0   (cluster, n=9)", w.mean(), len(Y), w.std(ddof=1)))
    for run, lab in SWEEP_LOCAL:
        k, v = local_curve(run)
        s = v[k >= WINDOW_LO].mean()
        rows.append((lab + ("  (85k)" if k[-1] != 100_000 else ""), s, 1, None))
    rows.sort(key=lambda r: r[1])

    refs = []
    for name, lab in [("dynens_h5_slow_anchor", "MPPI planning + source anchor"),
                      ("dynens_h5_slow", "MPPI planning, no anchor")]:
        gx, gY = group(name)
        p = gY[:, (gx >= WINDOW_LO) & (gx <= WINDOW_HI)].mean(1)
        refs.append((lab, p.mean(), p.std(ddof=1), len(gY)))

    fig, ax = plt.subplots(figsize=(11.4, 5.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for lab, mu, sd, n in refs:
        ax.axvspan(mu - sd, mu + sd, color=BLUE, alpha=0.10, lw=0, zorder=0)
        ax.axvline(mu, color=BLUE, lw=2.0, zorder=1)
        ax.text(mu - 0.013, 0.35, f"{lab}   {mu:.3f} $\\pm$ {sd:.3f}",
                fontsize=8, color=BLUE, va="bottom", ha="left", rotation=90)

    ys = np.arange(len(rows))
    for y, (lab, sc, n, sd) in zip(ys, rows):
        multi = n > 1
        ax.plot([0, sc], [y, y], color=ORANGE, lw=1.4 if multi else 1.2,
                alpha=0.5 if multi else 0.3, zorder=2, solid_capstyle="round")
        if multi:
            ax.plot([sc - sd, sc + sd], [y, y], color=ORANGE, lw=2.4, alpha=0.45, zorder=3)
        ax.scatter([sc], [y], s=90 if multi else 58, color=ORANGE, zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        # values in one fixed column: next to the dot they collided with the
        # rotated reference labels the mid rows pass through
        txt = f"{sc:.3f}" + (f" $\\pm$ {sd:.3f}" if multi else "")
        ax.text(1.02, y, txt, fontsize=8.5, color=INK2 if multi else MUTED,
                va="center", ha="left", weight="bold" if multi else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK2)
    for lab, t in zip(ax.get_yticklabels(), rows):
        if t[2] > 1:
            lab.set_color(INK)
            lab.set_weight("bold")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 1.19)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("target success rate, mean over the 75-100k steady-state window",
                  fontsize=10, color=INK2)
    ax.grid(True, axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    fig.suptitle("Dyna-IBRL's prior-margin knob on the dyn-ens world model, vs MPPI planning on the same model",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.975)
    fig.text(0.008, 0.912,
             "Threading_D0, 100k target transitions, strict success, $\\sigma$=0.1. m = the IBRL margin the actor must beat the frozen prior by. "
             "Only the bold row is multi-seed; the blue lines are n=10.",
             fontsize=9, color=INK2, ha="left")
    for i, n in enumerate([
        "The faint rows are ONE seed each, all on sat_160000_s1 -- a knob sweep, not estimates. The two best of them (0.933, 0.882) sit on the planner's line; the n=9 arm sits 0.15 below it.",
        "m=0 collapses to 0.00: with no margin the argmax is decided by critic noise, and ~6% action deviation is enough to lose the episode. The margin is what keeps the arm alive at all.",
    ]):
        fig.text(0.008, 0.012 + 0.024 * (1 - i), n, fontsize=7.6, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.070, 1.0, 0.885])
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)
    for lab, sc, n, sd in rows[::-1]:
        print(f"  {lab:34s} n={n:2d}  {sc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true",
                   help="draw the ranked prior-margin sweep instead of the curves")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(
        _REPO, "viz/threading_dynaibrl_vs_dynens%s.png" % ("_sweep" if args.sweep else ""))
    (sweep_figure if args.sweep else line_figure)(out)


if __name__ == "__main__":
    main()
