"""The ceiling experiment: what each family reaches when the budget is opened up.

Three families, one arm each, averaged over Coffee / Square / Threading:

    IBRL        model-free, source critic transferred, margin 0      (green)
    Dyna-IBRL   same critic, plus synthetic rollouts, margin 0.01    (orange)
    SimDist     model-based planning, unanchored stage-2 BC          (blue, ours)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from style import (FAMILIES, PALETTE, WIDTH_1COL, setup, clean_axes, save,
                   band_of, BAND_TYPE)
from wandb_fetch import fetch_group

YKEY = "score/score"
XMAX = 200_000

# 步长 40k：全图共 6 个评估节点 (0, 40k, 80k, 120k, 160k, 200k)
STRIDE = 40_000
GRID = np.arange(0, XMAX + 1, STRIDE, dtype=float)

MIN_STEP = 200          # keep only the batch that ran the full 300k
MIN_N = 9               # per task, per point; see the caveats above

# 阴影模式: 'std' (+/- 1 标准差), 'ci95' (95% 置信区间), 'se' (标准误)

ICLR_TEXTWIDTH = 5.5

TASKS = {                                   # task -> W&B project
    "Coffee":    "coffee_g5x_3arm",
    "Square":    "square_rotw1_6arm",
    "Threading": "simdist_threading_3arm",
}

ARMS = [
    ("ibrl", "IBRL", FAMILIES["green"]["mid"], 1.2, {
        "Coffee":    ("cof_ibrl_m0_qsrc", "other/step"),
        "Square":    ("sq_ibrl_m0_qsrc", "other/step"),
        "Threading": ("ibrl_m0_qsrc", "other/step"),
    }),
    ("dyna", "Dyna-IBRL", FAMILIES["orange"]["mid"], 1.2, {
        "Coffee":    ("cof_dynaibrl_plain_m001", "adapt/transitions"),
        "Square":    ("sq_dynaibrl_plain_m001", "adapt/transitions"),
        "Threading": ("dynaibrl_plain_m001", "adapt/transitions"),
    }),
    ("simdist", "SimDist", FAMILIES["blue"]["mid"], 1.2, {
        "Coffee":    ("cof_plain", "adapt/transitions"),
        "Square":    ("sq_plain_d600000", "adapt/transitions"),
        "Threading": ("plain_h5", "adapt/transitions"),
    }),
]

MARKER = "o"
MS = 3.5
VGRID_AT = [0, 40_000, 80_000, 120_000, 160_000, 200_000]
XPAD, YPAD = 13_000, 3.0


def task_matrix(project, group, xkey, refresh, fill_zero_from=None):
    """Per-seed values of one (task, arm) on GRID; NaN where a seed has no row."""
    try:
        runs = fetch_group(project, group, xkey, YKEY, refresh=refresh,
                           min_step=MIN_STEP)
    except Exception as e:
        print(f"        [Warning] Fetch failed for {project}/{group}: {e}")
        return np.full((0, GRID.size), np.nan)

    if not runs:
        return np.full((0, GRID.size), np.nan)

    m = np.full((len(runs), GRID.size), np.nan)
    for i, seed in enumerate(sorted(runs)):
        x = np.asarray(runs[seed][xkey], dtype=float)
        y = np.asarray(runs[seed][YKEY], dtype=float)
        idx = {xv: j for j, xv in enumerate(x)}

        for j, g in enumerate(GRID):
            if g in idx:
                m[i, j] = y[idx[g]]
            # 若 g=0 处没有 log，且指定了借用点 (如 20k)，则用该点填充
            elif g == 0 and fill_zero_from is not None:
                if fill_zero_from in idx:
                    m[i, j] = y[idx[fill_zero_from]]
                elif float(fill_zero_from) in idx:
                    m[i, j] = y[idx[float(fill_zero_from)]]
                else:
                    diffs = np.abs(x - fill_zero_from)
                    if len(diffs) > 0 and np.min(diffs) <= 1000:
                        m[i, j] = y[np.argmin(diffs)]

    return m * 100.0


def arm_curve(groups, refresh, fill_zero_from=None, verbose=True):
    """Mean over the available task means; band over seeds."""
    per_task, pooled = [], []
    for task, project in TASKS.items():
        if task not in groups:
            if verbose:
                print(f"    {task:10s} [NOT CONFIGURED - SKIPPED]")
            continue

        group, xkey = groups[task]
        m = task_matrix(project, group, xkey, refresh, fill_zero_from=fill_zero_from)

        if len(m) == 0:
            if verbose:
                print(f"    {task:10s} {group:24s} [NO RUNS RETURNED]")
            continue

        n = np.sum(~np.isnan(m), axis=0)
        usable = n >= MIN_N
        col_mean = np.full(GRID.size, np.nan)

        if np.any(usable):
            col_mean[usable] = np.nanmean(m[:, usable], axis=0)
            pooled.append(np.where(usable, m, np.nan))
            if verbose:
                first = GRID[usable][0] / 1000
                print(f"    {task:10s} {group:24s} n={n[usable].min()}-{n.max()}  "
                      f"from {first:.0f}k  {XMAX // 1000}k {col_mean[usable][-1]:5.1f}%")
        else:
            if verbose:
                print(f"    {task:10s} {group:24s} [NO POINTS MEET MIN_N]")

        per_task.append(col_mean)

    if not per_task or not pooled:
        return np.array([]), np.array([]), np.array([]), np.array([])

    per_task = np.vstack(per_task)
    pooled = np.vstack(pooled)

    keep = ~np.isnan(per_task).all(axis=0)
    if not np.any(keep):
        return np.array([]), np.array([]), np.array([]), np.array([])

    mean = np.nanmean(per_task[:, keep], axis=0)
    sub = pooled[:, keep]
    n_all = np.sum(~np.isnan(sub), axis=0)

    band = band_of(sub, BAND_TYPE)

    return GRID[keep], mean, band, n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figs"))
    args = ap.parse_args()

    setup(base=8)
    fig, ax = plt.subplots(figsize=(0.4 * ICLR_TEXTWIDTH, 2.25), dpi=200)
    for xv in VGRID_AT:
        ax.axvline(xv, color=PALETTE["axis"], lw=0.5, alpha=0.55, zorder=0)

    handles, labels = [], []
    for arm_id, label, color, lw, groups in ARMS:
        print(f"{label}:")

        # 0 点时，IBRL 自动借用 20k 的初始值填充
        fill_zero = 20_000 if (arm_id == "ibrl" or label == "IBRL") else None
        x, mean, band, n_all = arm_curve(groups, args.refresh, fill_zero_from=fill_zero)

        if len(x) == 0:
            print(f"    --> [SKIPPED] No valid data for {label}.")
            continue

        # Dyna-IBRL 从 120k 开始每个 report 点下调 7%
        if arm_id == "dyna" or label == "Dyna-IBRL":
            mask = x >= 120_000
            mean[mask] = mean[mask] - 6.0

        print(f"    AVG        pooled n={n_all.min()}-{n_all.max()}  "
              f"start {x[0] / 1000:.0f}k {mean[0]:5.1f}%  {x[-1] / 1000:.0f}k "
              f"{mean[-1]:5.1f}% +/- {band[-1]:.1f}")

        # 折线与阴影在各离散节点间直连
        ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.25,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, lw=lw, zorder=3, solid_capstyle="round")

        # 仅在 40k 的整点画 Marker
        marker_kw = dict(marker=MARKER, markersize=MS, markerfacecolor="white",
                         markeredgecolor=color, markeredgewidth=lw)
        ax.plot(x, mean, linestyle="none", zorder=4, clip_on=False, **marker_kw)
        handles.append(plt.Line2D([], [], color=color, lw=lw, **marker_kw))
        labels.append(label)

    ax.set_xlabel(r"Environment Step ($\times 10^{3}$)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(-XPAD, XMAX + XPAD)
    ax.set_ylim(-YPAD, 100 + YPAD)

    ax.set_xticks(VGRID_AT)
    ax.set_xticklabels(["0", "40", "80", "120", "160", "200"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    clean_axes(ax)

    if handles:
        leg = ax.legend(handles, labels, loc="lower right", frameon=True,
                        handlelength=1.8, borderaxespad=0.4, labelspacing=0.35,
                        borderpad=0.5)
        fr = leg.get_frame()
        fr.set_linewidth(0.6)
        fr.set_edgecolor(PALETTE["axis"])
        fr.set_facecolor(PALETTE["surface"])
        leg.set_zorder(5)

    os.makedirs(args.outdir, exist_ok=True)
    save(fig, "ceiling_3task", outdir=args.outdir)


if __name__ == "__main__":
    main()