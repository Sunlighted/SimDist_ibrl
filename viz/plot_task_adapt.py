"""One matched-budget figure per task: Threading_D0, Square_D0 (rotw1), Coffee_D0 (G5x).

Same axes, same palette, same read-out for all three, so the panels can sit side
by side. Data comes from viz/_cache/*.json (viz/fetch_wandb_curves.py) -- the
Square baselines ran on the cluster and exist only on wandb, so reading local
train.log files would silently drop five of its eight arms.

Every arm gets the SAME 100k target transitions. A run is included only if it
actually reached the steady-state window; short/crashed runs are dropped and
counted in the console output rather than being flat-extrapolated by np.interp.

Read-out defaults to `score/score` for every arm. `--key score_best` reproduces
the docs' table for the dyn-ens arms (a max over their plan-time pessimism sweep,
which the other arms never ran and therefore cannot take); arms that fall back to
`score/score` are marked with a *.

    python viz/plot_task_adapt.py                 # all three, score/score
    python viz/plot_task_adapt.py --key score_best
    python viz/plot_task_adapt.py --task coffee
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_REPO, "viz/_cache")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILIES = [
    ("Model-based planning (MPPI on the world model)", BLUE),
    ("Model-based Dyna (same world model, offline)", ORANGE),
    ("Model-free RL", AQUA),
]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS = "#e1e0d9", "#c3c2b7"

WINDOW_LO, WINDOW_HI = 75_000, 100_000

# (group, end label, family index, dashes, linewidth)
PLANNING_TAIL = [("Dyna", 1, (), 2.0), ("Dyna + SGFT shaping", 1, (3, 2), 2.0),
                 ("IBRL", 2, (), 2.0), ("RLPD", 2, (3, 2), 2.0),
                 ("IQL", 2, (1.2, 1.8), 2.0)]

TASKS = {
    "threading": dict(
        project="simdist_threading_3arm",
        title="Threading_D0 sim2sim: nine adaptation methods at a matched 100k-transition budget",
        subtitle="Target = the dynamics-only §12 definition, strict success, $\\sigma$=0.1; all SimDist arms share one stage-1 world model.",
        arms=[("dynens_h5_slow_anchor", "dyn-ens + slow + source anchor", 0, (), 2.6),
              ("dynens_h5_slow", "dyn-ens + slow", 0, (7, 2.2), 2.0),
              ("dynens_h5", "dyn-ens", 0, (3, 2), 2.0),
              ("plain_h3", "plain SimDist (H=3)", 0, (1.2, 1.8), 2.0),
              ("dyna_h5", "Dyna", 1, (), 2.0),
              ("dyna_sgft", "Dyna + SGFT shaping", 1, (3, 2), 2.0),
              ("ibrl", "IBRL", 2, (), 2.0),
              ("rlpd", "RLPD", 2, (3, 2), 2.0),
              ("iql", "IQL", 2, (1.2, 1.8), 2.0)],
        out="threading_matched_budget",
        note="Model-free arms carry 10 warm-up episodes outside the 100k budget.",
    ),
    "square": dict(
        project="square_rotw1_6arm",
        title="Square_D0 (rotw1): eight adaptation methods at a matched 100k-transition budget",
        subtitle="Trust region init_std=0.15 / base_policy_std=0 (§41), $\\sigma$=0.1 (strict_success is OFF on this task); the SimDist arms share one stage-1 world model and one 20-demo set per seed.",
        arms=[("sq_dynens_anch_d400000", "dyn-ens + source anchor ($\\lambda$=0.1)", 0, (), 2.6),
              ("sq_dynens_d400000", "dyn-ens", 0, (3, 2), 2.0),
              ("sq_plain_d400000", "plain SimDist", 0, (1.2, 1.8), 2.0),
              ("dyna_h5", "Dyna", 1, (), 2.0),
              ("dyna_sgft", "Dyna + SGFT shaping", 1, (3, 2), 2.0),
              ("ibrl", "IBRL", 2, (), 2.0),
              ("rlpd", "RLPD", 2, (3, 2), 2.0),
              ("iql", "IQL", 2, (1.2, 1.8), 2.0)],
        out="square_matched_budget",
        note="Demos from the 400k (unsaturated) collector; the 600k half is the collector contrast, not shown here.",
    ),
    "coffee": dict(
        project="coffee_g5x_3arm",
        title="Coffee_D0 (G5x): the SimDist ladder at a matched 100k-transition budget",
        subtitle="Target = G5x (§40e), strict success, $\\sigma$=0.1; all arms share one stage-1 world model and one 20-demo set per seed.",
        arms=[("cof_dynens_anch", "dyn-ens + source anchor ($\\lambda$=1.0)", 0, (), 2.6),
              ("cof_dynens", "dyn-ens", 0, (3, 2), 2.0),
              ("cof_plain", "plain SimDist", 0, (1.2, 1.8), 2.0),
              ("dyna_h5", "Dyna", 1, (), 2.0),
              ("dyna_sgft", "Dyna + SGFT shaping", 1, (3, 2), 2.0),
              ("ibrl", "IBRL", 2, (), 2.0),
              ("rlpd", "RLPD", 2, (3, 2), 2.0),
              ("iql", "IQL", 2, (1.2, 1.8), 2.0)],
        out="coffee_matched_budget",
        note="Baseline arms that have not reached 100k yet are omitted -- an absent family means unfinished, not zero.",
    ),
}


def declutter(ys, gap):
    """Push end labels apart, keeping their order. Ties must separate too, so walk
    the sorted order and force each label at least `gap` above the previous one."""
    order = np.argsort(ys, kind="stable")
    out = np.array(ys, dtype=float)
    for prev, i in zip(order, order[1:]):
        out[i] = max(out[i], out[prev] + gap)
    over = out.max() - 1.0
    if over > 0:
        for i in order[::-1]:
            out[i] -= over
    return out


def arm_curves(runs, key):
    """(x grid, [n_runs, n_points], fell_back, n_dropped). Runs that never reached
    the window are dropped: np.interp would otherwise hold their last value flat
    across the whole quoted window."""
    short = [r for r in runs.values() if not r["x"] or max(r["x"]) < WINDOW_HI]
    keep = [r for r in runs.values() if r["x"] and max(r["x"]) >= WINDOW_HI]
    if not keep:
        return None
    kname = key.split("/")[-1]
    fell_back = any(kname not in r or r.get(kname) is None for r in keep)
    xs = np.array(max((r["x"] for r in keep), key=len))
    ys = []
    for r in keep:
        v = r[kname] if (kname in r and r[kname] is not None) else r["score"]
        v = [b if b is not None else a for a, b in zip(r["score"], v)]
        ys.append(np.interp(xs, r["x"], v))
    return xs, np.stack(ys), fell_back, len(short)


def plot(task, key, outdir):
    spec = TASKS[task]
    data = json.load(open(os.path.join(CACHE, spec["project"] + ".json")))

    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    drawn, fams_used, dropped_any = [], set(), 0
    for group, label, fam, dashes, lw in spec["arms"]:
        if group not in data:
            print(f"  !! {task}: group {group} missing from cache -- skipped")
            continue
        got = arm_curves(data[group], key)
        if got is None:
            print(f"  .. {task}: {group} has no run past {WINDOW_HI//1000}k yet -- omitted")
            continue
        xs, Y, fell_back, n_short = got
        dropped_any += n_short
        mu, sd = Y.mean(0), Y.std(0, ddof=1)
        color = FAMILIES[fam][1]
        fams_used.add(fam)
        ax.fill_between(xs / 1000, mu - sd, mu + sd, color=color, alpha=0.10, lw=0, zorder=2)
        ax.plot(xs / 1000, mu, color=color, lw=lw, dashes=dashes, zorder=3,
                solid_capstyle="round")
        sel = (xs >= WINDOW_LO) & (xs <= WINDOW_HI)
        per_seed = Y[:, sel].mean(1)
        drawn.append(dict(label=label, color=color, end=mu[-1], score=per_seed.mean(),
                          sd=per_seed.std(ddof=1), n=Y.shape[0], fell_back=fell_back,
                          dropped=n_short, bold=("anchor" in label)))

    ty = declutter([d["score"] for d in drawn], gap=0.062 if len(drawn) > 4 else 0.075)
    for d, y in zip(drawn, ty):
        ax.plot([100, 103], [d["end"], y], color=d["color"], lw=0.9, alpha=0.6,
                clip_on=False, zorder=3)
        ax.scatter([100], [d["end"]], s=26, color=d["color"], zorder=5,
                   edgecolor=SURFACE, linewidth=1.5, clip_on=False)
        ax.text(104, y, d["label"] + (" *" if d["fell_back"] else ""), fontsize=9.5,
                color=d["color"], va="center", ha="left",
                weight="bold" if d["bold"] else "normal")
        ax.text(104, y - 0.030, f"{d['score']:.3f} $\\pm$ {d['sd']:.3f}   n={d['n']}",
                fontsize=8, color=MUTED, va="center", ha="left")

    ax.axvspan(WINDOW_LO / 1000, WINDOW_HI / 1000, color=GRIDLINE, alpha=0.55, zorder=0)
    ax.text((WINDOW_LO + WINDOW_HI) / 2000, 0.44,
            "75-100k\nsteady-state\nwindow\n(quoted numbers)",
            ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.5)

    fams = [f for f in range(3) if f in fams_used]
    handles = [plt.Line2D([], [], color=FAMILIES[f][1], lw=2.6) for f in fams]
    leg = ax.legend(handles, [FAMILIES[f][0] for f in fams], frameon=False, fontsize=9.5,
                    loc="center left", bbox_to_anchor=(0.13, 0.25), labelcolor=INK2,
                    borderpad=0.2, labelspacing=0.4, handlelength=2.4)
    leg.set_zorder(6)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel(f"target transitions collected (k)   -- matched budget for all {len(drawn)} arms",
                  fontsize=10, color=INK2)
    ax.set_ylabel("target success rate (100 eval episodes per point)", fontsize=10, color=INK2)
    ax.grid(True, color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)

    fig.suptitle(spec["title"] + ("   [score_best]" if "best" in key else ""),
                 fontsize=13, color=INK, x=0.008, ha="left", y=0.982)
    fig.text(0.008, 0.928, spec["subtitle"], fontsize=9, color=INK2, ha="left")

    notes = ["Shaded band = $\\pm$1 sd across seeds; quoted numbers are the per-seed mean over the shaded window, $\\pm$1 sd across seeds."]
    if "best" in key:
        notes.append("Read-out is score_best = max over the arm's plan-time pessimism sweep; it reproduces the docs' table but gives the dyn-ens arms a selection bias the other arms cannot take.")
        notes.append("* this arm ran no sweep and has no score_best, so its score/score is shown.")
    else:
        notes.append("Read-out is score/score for every arm -- the only common ruler, since only the dyn-ens arms ran a plan-time pessimism sweep (score_best, worth +0.014..+0.045 to them).")
    if dropped_any:
        notes.append(f"{dropped_any} run(s) never reached 100k and are excluded rather than flat-extrapolated.")
    notes.append(spec["note"])
    for i, n in enumerate(notes):
        fig.text(0.008, 0.010 + 0.019 * (len(notes) - 1 - i), n, fontsize=7.6,
                 color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.032 + 0.019 * len(notes), 0.735, 0.90])
    out = os.path.join(outdir, spec["out"] + ("_best" if "best" in key else "") + ".png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}  | key = {key}")
    for d in drawn:
        drop = f"  ({d['dropped']} short run(s) dropped)" if d["dropped"] else ""
        print(f"   {d['label']:38s} n={d['n']:2d}  {WINDOW_LO//1000}-{WINDOW_HI//1000}k = "
              f"{d['score']:.3f} +- {d['sd']:.3f}{drop}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all"] + list(TASKS))
    p.add_argument("--key", default="score/score",
                   choices=["score/score", "score_best", "score/score_best"])
    p.add_argument("--outdir", default=os.path.join(_REPO, "viz"))
    a = p.parse_args()
    key = "score/score_best" if a.key == "score_best" else a.key
    for t in (list(TASKS) if a.task == "all" else [a.task]):
        plot(t, key, a.outdir)


if __name__ == "__main__":
    main()
