"""Square_D0 (rotw1): the three SimDist adapt arms built on the 400k demo collector.

Source: exps/square_rotw1_6arm, the `*_d400000` half only (the `*_d600000` arms --
the saturated-collector contrast of viz/plot_square_demo_source.py -- are ignored
here on purpose). Three arms, an ordered method ladder:

    plain SimDist            single dynamics head, stage-1 `square_simdist_rotw1_h5`
    + dynamics ensemble      5 heads, plan-time mean, stage-1 `..._dynens_h5`
    + source anchor          same stage-1 as dyn-ens, plus bc_anchor_coef = 0.1

Everything else is matched: 100k target transitions, the Square trust region
`init_std=0.15, base_policy_std=0.0` (tasks.md §41), 100 eval episodes per point,
and -- arm for arm -- the SAME 20-demo set per seed (`square_sat_400000_s<seed>`).

Read-out defaults to `score/score` for all three. The arms ran different
plan-time pessimism sweeps (none / 3 coefficients / 4), so `score_best` gives
each a different selection bias and `plain` has none at all -- the plain score is
the only common ruler. `--key score_best` switches to the docs' read-out, falling
back to `score` (and marking it with a *) on any arm that has no sweep.

Usage:
    python viz/plot_square_3arm_adapt.py
    python viz/plot_square_3arm_adapt.py --key score_best
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(_REPO, "exps/square_rotw1_6arm")

# one-hue ordinal ramp (dataviz reference palette, blue steps 250/450/700):
# monotone L 0.764/0.575/0.338, adjacent dL 0.189/0.237, hue spread 3.6 deg,
# light end 2.06:1 vs surface -- passes the ordinal checks.
ARMS = [
    ("sq_plain_d400000",      "plain SimDist",        "#86b6ef", (1.2, 1.8), 2.0),
    ("sq_dynens_d400000",     "+ dynamics ensemble",  "#2a78d6", (5, 2.2), 2.2),
    ("sq_dynens_anch_d400000", "+ source anchor ($\\lambda$=0.1)", "#0d366b", (), 2.6),
]
WINDOW_LO, WINDOW_HI = 60_000, 100_000

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"


def curve(run, key):
    """Series for `key`, falling back to score/score when the arm never logged it
    (an arm with no plan-time pessimism sweep has no score_best at all)."""
    with open(os.path.join(EXP, run, "train.log"), errors="ignore") as f:
        txt = f.read()
    for k_try in (key, "score/score"):
        pat = re.compile(rf"^\s*(\d+): {re.escape(k_try)}\s*:\s*([0-9.]+)\s*$", re.M)
        pts = {int(a): float(b) for a, b in pat.findall(txt)}
        if pts:
            ks = np.array(sorted(pts))
            return ks, np.array([pts[i] for i in ks]), k_try


def declutter(ys, gap):
    """Push end labels apart, keeping their order (ties must separate too)."""
    order = np.argsort(ys, kind="stable")
    out = np.array(ys, dtype=float)
    for prev, i in zip(order, order[1:]):
        out[i] = max(out[i], out[prev] + gap)
    over = out.max() - 1.0
    if over > 0:
        for i in order[::-1]:
            out[i] -= over
    return out


def demo_of(run):
    with open(os.path.join(EXP, run, "cfg.yaml")) as f:
        return re.search(r"square_sat_400000_(s\d+)", f.read()).group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="score/score",
                   choices=["score/score", "score_best", "score/score_best"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    key = "score/score_best" if args.key == "score_best" else args.key
    args.out = args.out or os.path.join(
        _REPO, "viz/square_3arm_adapt%s.png" % ("_best" if "best" in key else ""))

    fig, ax = plt.subplots(figsize=(11.4, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    drawn = []
    for pref, label, color, dashes, lw in ARMS:
        runs = sorted(d for d in os.listdir(EXP) if re.fullmatch(rf"{pref}_seed\d+", d))
        xs, ys, per_demo, used = None, [], {}, set()
        for r in runs:
            k, v, uk = curve(r, key)
            used.add(uk)
            if xs is None:
                xs = k
            y = np.interp(xs, k, v)
            ys.append(y)
            sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
            per_demo[demo_of(r)] = y[sel].mean()
        Y = np.stack(ys)
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        ax.fill_between(xs / 1000, mu - sd, mu + sd, color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(xs / 1000, mu, color=color, lw=lw, dashes=dashes, zorder=3,
                solid_capstyle="round", label=label)
        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        w = Y[:, sel].mean(1)
        drawn.append(dict(label=label, color=color, end=mu[-1], score=w.mean(),
                          sd=w.std(ddof=1), n=len(runs), per_demo=per_demo,
                          fell_back=(used != {key})))

    ty = declutter([d["score"] for d in drawn], gap=0.075)   # label rows, not values
    for d, y in zip(drawn, ty):
        ax.plot([100, 103], [d["end"], y], color=d["color"], lw=0.9,
                alpha=0.6, clip_on=False, zorder=3)
        ax.scatter([100], [d["end"]], s=28, color=d["color"], zorder=5,
                   edgecolor=SURFACE, linewidth=1.5, clip_on=False)
        ax.text(104, y, d["label"] + (" *" if d["fell_back"] else ""),
                fontsize=10, color=d["color"],
                va="center", ha="left",
                weight="bold" if "anchor" in d["label"] else "normal")
        ax.text(104, y - 0.034,
                f"{d['score']:.3f} $\\pm$ {d['sd']:.3f}   n={d['n']}",
                fontsize=8.5, color=MUTED, va="center", ha="left")

    # paired deltas, on the demo sets the two arms actually share
    def paired(a, b):
        common = sorted(set(a["per_demo"]) & set(b["per_demo"]), key=lambda s: int(s[1:]))
        v = np.array([a["per_demo"][c] - b["per_demo"][c] for c in common])
        return v.mean(), v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), int((v > 0).sum()), len(v)

    d1, t1, s1, n1 = paired(drawn[1], drawn[0])
    d2, t2, s2, n2 = paired(drawn[2], drawn[1])
    ax.text(0.025, 0.975,
            f"paired on the same demo set:\n"
            f"  ensemble  $\\Delta$ = {d1:+.3f}   (t = {t1:.1f}, {s1}/{n1})\n"
            f"  anchor      $\\Delta$ = {d2:+.3f}   (t = {t2:.1f}, {s2}/{n2})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=INK2,
            linespacing=1.6)

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color=GRIDLINE, alpha=0.55, zorder=0)
    ax.text((WINDOW_LO + WINDOW_HI) / 2000, 1.005,
            "60-100k steady-state window   (quoted numbers)",
            ha="center", va="top", fontsize=8, color=MUTED)

    leg = ax.legend(frameon=False, fontsize=9.5, loc="lower left",
                    bbox_to_anchor=(0.02, 0.03), labelcolor=INK2,
                    borderpad=0.2, labelspacing=0.4, handlelength=2.6)
    leg.set_title("SimDist adapt arm", prop=dict(size=8.5))
    leg.get_title().set_color(MUTED)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("target transitions collected (k)", fontsize=10, color=INK2)
    ax.set_ylabel("target success rate (100 eval episodes per point)", fontsize=10, color=INK2)
    ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    fig.suptitle("Square_D0 (rotw1): three SimDist adapt arms on the 400k collector's demos"
                 + ("   [score_best]" if "best" in key else ""),
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.928,
             "Matched 100k-transition budget, same planner trust region (init_std=0.15, base_policy_std=0.0), "
             "and the same 20-demo set per seed. Shaded band = $\\pm$1 sd over 10 seeds.",
             fontsize=9, color=INK2, ha="left")
    if "best" in key:
        notes = [
            "Read-out is score_best = max over each arm's plan-time pessimism sweep (dyn-ens: 3 coefficients, anchor: 4), so the two carry different selection biases.",
            "* plain SimDist ran no sweep and has no score_best at all -- its score/score is shown instead, which is why its curve is identical to the score version.",
        ]
    else:
        notes = [
            "Read-out is score/score: the three arms ran different plan-time pessimism sweeps (none / 3 / 4 coefficients), so score_best would carry a different selection bias in each.",
        ]
    notes.append(
        "The anchor arm uses $\\lambda$=0.1, not the $\\lambda$=1 that peaked on Threading, and its seed-2 run drew demo set s2 where the other two arms drew s1"
        " -- hence 9 paired demo sets for the anchor delta, 10 for the ensemble.")
    for i, n in enumerate(notes):
        fig.text(0.008, 0.010 + 0.019 * (len(notes) - 1 - i), n,
                 fontsize=7.6, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.032 + 0.019 * len(notes), 0.755, 0.90])
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out, "| key =", key)
    for d in drawn:
        print(f"  {d['label']:34s} n={d['n']:2d}  60-100k = {d['score']:.3f} +- {d['sd']:.3f}")
    print(f"  paired: ensemble {d1:+.3f} (t={t1:.2f}, {s1}/{n1})   anchor {d2:+.3f} (t={t2:.2f}, {s2}/{n2})")


if __name__ == "__main__":
    main()
