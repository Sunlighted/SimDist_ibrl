"""Plot the Dyna stage-3 sweeps (trust region + bc_coef) on Threading_D0.

All arms share one stage-1 world model, one stage-2 BC policy, and a 100k target
transition budget; only the named knob differs from `dyna_h5`. Steady state is the
75-100k window (6 eval points x 100 games); runs truncated by the disk-full stop
are drawn to wherever they got and marked.
"""
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "exps/simdist_threading_3arm"
OUT = "viz/_cache"
SEEDS = (1, 101, 201)
PAT = re.compile(r"^(\d+):\s+(\S+)\s+\[\s*\d+\]:\s+avg:\s+([-\d.]+)")


def score_curve(run):
    xs = {}
    p = f"{R}/{run}/train.log"
    if not os.path.isfile(p):
        return xs
    for line in open(p):
        f = line.split(":")
        if len(f) >= 3 and f[1].strip() == "score/score":
            try:
                xs[int(f[0])] = float(f[2])
            except ValueError:
                pass
    return xs


def metric_curve(run, key):
    xs = {}
    p = f"{R}/{run}/train.log"
    if not os.path.isfile(p):
        return xs
    for line in open(p):
        m = PAT.match(line)
        if m and m.group(2) == key:
            xs[int(m.group(1))] = float(m.group(3))
    return xs


def cell(name, seeds=SEEDS):
    """-> list of (seed, {step: score}) for the runs that exist."""
    out = []
    for s in seeds:
        c = score_curve(f"dyna_{name}_seed{s}" if name != "h5" else f"dyna_h5_seed{s}")
        if c:
            out.append((s, c))
    return out


def steady(c):
    v = [c[k] for k in c if k >= 75000]
    return np.mean(v) if v else np.nan


def draw(ax, name, label, color, seeds=SEEDS, lw=2.0, ls="-"):
    runs = cell(name, seeds)
    if not runs:
        return None
    grid = sorted({k for _, c in runs for k in c})
    M = np.full((len(runs), len(grid)), np.nan)
    for i, (_, c) in enumerate(runs):
        for j, k in enumerate(grid):
            if k in c:
                M[i, j] = c[k]
    x = np.array(grid) / 1000.0
    mean = np.nanmean(M, axis=0)
    ok = ~np.isnan(mean)
    for i in range(len(runs)):
        ax.plot(x[~np.isnan(M[i])], M[i][~np.isnan(M[i])],
                color=color, alpha=0.18, lw=0.9)
    # n = runs that actually reach a usable window, not runs that logged step 0
    n = sum(1 for _, c in runs if max(c) >= 20000)
    trunc = "" if grid[-1] >= 100000 else f", cut@{grid[-1]//1000}k"
    ax.plot(x[ok], mean[ok], color=color, lw=lw, ls=ls,
            label=f"{label}  (n={n}{trunc})")
    return steady({k: v for k, v in zip(grid, mean) if not np.isnan(v)})


BASE = "#8a8a8a"
fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
fig.suptitle("Dyna stage-3 on Threading_D0 — one knob at a time off dyna_h5 "
             "(100k target transitions, 100-game evals)", fontsize=13, y=0.985)

# (a) stage-3 policy anchor -------------------------------------------------
ax = axes[0][0]
draw(ax, "h5", "baseline (no anchor)", BASE)
draw(ax, "plr0", "policy frozen (lr=0)", "#000000", ls=":")
for nm, lb, co in [("anc1", "anchor coef 1", "#9ecae1"),
                   ("anc10", "anchor coef 10", "#2171b5"),
                   ("anc100", "anchor coef 100", "#08306b"),
                   ("ancsrc10", "anchor 10 -> pi_src", "#d95f02")]:
    draw(ax, nm, lb, co)
ax.set_title("(a) stage-3 policy anchor  [NEW knob]")

# (b) bc_coef ---------------------------------------------------------------
ax = axes[0][1]
draw(ax, "h5", "bc_coef 0.1 (default)", BASE)
for nm, lb, co in [("bc1", "bc_coef 1", "#c7e9c0"), ("bc10", "bc_coef 10", "#74c476"),
                   ("bc30", "bc_coef 30", "#238b45"), ("bc100", "bc_coef 100", "#00441b"),
                   ("bc300", "bc_coef 300", "#7a0177")]:
    draw(ax, nm, lb, co)
ax.set_title("(b) bc_coef  [original Dyna knob]")

# (c) policy_lr -------------------------------------------------------------
ax = axes[0][2]
draw(ax, "h5", "policy_lr 1e-5 (default)", BASE)
for nm, lb, co in [("plr3e6", "3e-6", "#fdae6b"), ("plr1e6", "1e-6", "#e6550d"),
                   ("plr0", "0 (frozen)", "#000000")]:
    draw(ax, nm, lb, co)
ax.set_title("(c) policy_lr  [original knob]")

# (d) real_ratio ------------------------------------------------------------
ax = axes[1][0]
draw(ax, "h5", "real_ratio 0.5 (default)", BASE)
for nm, lb, co in [("rr025", "0.25 (more imagined)", "#9e9ac8"),
                   ("rr075", "0.75", "#6a51a3"),
                   ("rr100", "1.00 = model OFF", "#3f007d")]:
    draw(ax, nm, lb, co)
ax.set_title("(d) real_ratio  [MBPO knob]")

# (e) q_mean ----------------------------------------------------------------
ax = axes[1][1]
for nm, lb, co in [("h5", "baseline", BASE), ("anc10", "anchor 10", "#2171b5"),
                   ("rr100", "model OFF", "#3f007d"), ("plr0", "policy frozen", "#000000"),
                   ("bc30", "bc_coef 30", "#238b45")]:
    runs = [(s, metric_curve(f"dyna_{nm}_seed{s}", "dyna/q_mean")) for s in SEEDS]
    runs = [(s, c) for s, c in runs if c]
    if not runs:
        continue
    grid = sorted({k for _, c in runs for k in c})
    M = np.full((len(runs), len(grid)), np.nan)
    for i, (_, c) in enumerate(runs):
        for j, k in enumerate(grid):
            if k in c:
                M[i, j] = c[k]
    ax.plot(np.array(grid) / 1000.0, np.nanmean(M, axis=0), color=co, lw=2, label=lb)
ax.set_title("(e) dyna/q_mean — the critic collapse the anchor prevents")
ax.set_ylabel("mean Q over the training batch")

# (f) dose-response ---------------------------------------------------------
ax = axes[1][2]
base_steady = np.nanmean([steady(c) for _, c in cell("h5")])
frozen = np.nanmean([steady(c) for _, c in cell("plr0")])


def dose(names, values, color, label, window=None):
    xs, ys, ns = [], [], []
    for nm, v in zip(names, values):
        runs = cell(nm)
        if not runs:
            continue
        if window:                       # truncated cells: use a shared early window
            sc = [np.mean([c[k] for k in window if k in c]) for _, c in runs
                  if all(k in c for k in window)]
        else:
            sc = [steady(c) for _, c in runs]
            sc = [v for v in sc if not np.isnan(v)]
        if not sc:
            continue
        xs.append(v); ys.append(np.mean(sc)); ns.append(len(sc))
    ax.plot(xs, ys, "o-", color=color, label=label, lw=2)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color=color)


W = (20000, 25000, 30000, 35000)
dose(["h5", "anc1", "anc10", "anc100"], [0.03, 1, 10, 100], "#2171b5",
     "anchor coef (steady 75-100k)")
dose(["h5", "bc1", "bc10", "bc30", "bc100", "bc300"], [0.1, 1, 10, 30, 100, 300],
     "#238b45", "bc_coef (20-35k window)", window=W)
ax.axhline(base_steady, color=BASE, ls="--", lw=1)
ax.axhline(frozen, color="k", ls=":", lw=1)
ax.annotate("baseline dyna_h5", (0.03, base_steady), fontsize=8, color=BASE,
            va="bottom")
ax.annotate("policy frozen (do nothing)", (0.03, frozen), fontsize=8, color="k",
            va="bottom")
ax.set_xscale("log")
ax.set_xlabel("coefficient (log)")
ax.set_title("(f) dose-response: both regularisers invert")

for row in axes:
    for a in row:
        a.grid(alpha=0.25)
        a.legend(fontsize=7.5, loc="upper left")
        if a is not axes[1][2]:
            a.set_xlabel("target transitions (k)")
            a.axvspan(75, 100, color="k", alpha=0.05)
        if a not in (axes[1][1], axes[1][2]):
            a.set_ylabel("target success (100 games)")
            a.set_ylim(-0.02, 1.0)
axes[1][2].set_ylabel("steady-state success")

plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(OUT, exist_ok=True)
plt.savefig(f"{OUT}/dyna_sweep.png", dpi=130)
print(f"wrote {OUT}/dyna_sweep.png")

# --- also dump the real_ratio numbers, which the user asked about ----------
print("\nreal_ratio cells (steady 75-100k):")
for nm, v in [("rr025", 0.25), ("h5", 0.5), ("rr075", 0.75), ("rr100", 1.0)]:
    runs = cell(nm)
    sc = [(s, steady(c), max(c)) for s, c in runs]
    got = ", ".join(f"seed{s}={v2:.3f}{'' if last>=100000 else f'(cut@{last//1000}k)'}"
                    for s, v2, last in sc if not np.isnan(v2))
    print(f"  real_ratio={v:<5} n={len(sc)}  {got}")
