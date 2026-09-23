"""Plot MimicGen PegInsertion demos' actions over time, next to their source seed.

Rows: translation action (x,y,z), rotation action (x,y,z), gripper action, and
the peg tip height relative to the board top (mm; < 0 = inside the pocket).
Columns: a few generated demos + the source seed of the first one. The dashed
line is the first grasp signal = where MimicGen switches from the peg-relative
segment to the board-relative one, i.e. where a stitching jump would show.

    python scripts/plot_mimicgen_actions.py \
        --dataset release/data/mimicgen/generated/peg_insertion/peg_hard_n200_ins0/demo.hdf5 \
        --out viz/mimicgen_peg/hard_actions.png
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = os.path.join(REPO, "release/data/mimicgen/source/peg_insertion_scripted.hdf5")
PEG_HALF_Z, BOARD_HALF_Z = 0.061471, 0.02049
INSERT_DEPTH = 0.030
# reference palette (dataviz skill, light mode): categorical slots 1-3, text inks
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def traces(g):
    a = g["actions"][()]
    di = g["datagen_info"]
    peg = di["object_poses/peg"][()]
    board = di["object_poses/board"][()]
    tip_z = peg[:, 2, 3] - np.abs(peg[:, 2, 2]) * PEG_HALF_Z
    rel = (tip_z - (board[:, 2, 3] + BOARD_HALF_Z)) * 1000
    grasp = di["subtask_term_signals/grasp"][()]
    t_grasp = int(np.argmax(grasp)) if grasp.any() else None
    return a, rel, t_grasp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--demos", default="0,1,2")
    args = p.parse_args()
    f = h5py.File(args.dataset, "r")
    src = h5py.File(SRC, "r")
    ids = [int(x) for x in args.demos.split(",")]

    cols = []
    for i in ids:
        g = f[f"data/demo_{i}"]
        s = int(g["src_demo_inds"][0]) if "src_demo_inds" in g else None
        cols.append((f"generated demo_{i}" + (f"  (seed = source demo_{s})" if s is not None else ""), traces(g)))
    s0 = int(f[f"data/demo_{ids[0]}"]["src_demo_inds"][0])
    cols.append((f"source demo_{s0} (scripted expert)", traces(src[f"data/demo_{s0}"])))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(4, len(cols), figsize=(3.6 * len(cols), 8.5), sharex="col", sharey="row",
                           facecolor="#fcfcfb", layout="constrained")
    rows = ["translation action", "rotation action", "gripper action", "peg tip height above board top (mm)"]
    for c, (title, (a, rel, tg)) in enumerate(cols):
        t = np.arange(len(a))
        for k, lab in enumerate("xyz"):
            ax[0, c].plot(t, a[:, k], color=C[k], lw=1.5, label=lab)
            ax[1, c].plot(t, a[:, 3 + k], color=C[k], lw=1.5, label=lab)
        ax[2, c].plot(t, a[:, 6], color=C[0], lw=1.5)
        ax[3, c].plot(t, rel, color=C[0], lw=1.5)
        ax[3, c].axhline(0, color=INK2, lw=0.8)
        ax[3, c].axhline(-INSERT_DEPTH * 1000, color=INK2, lw=0.8, ls=":")
        ax[0, c].set_title(title, fontsize=9, color=INK)
        for r in range(4):
            a_ = ax[r, c]
            a_.set_facecolor("#fcfcfb")
            a_.grid(True, color=GRID, lw=0.6)
            a_.set_axisbelow(True)
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
            if tg is not None:
                a_.axvline(tg, color=INK2, lw=0.8, ls="--")
        ax[3, c].set_xlabel("step (20 Hz)")
    for r, lab in enumerate(rows):
        ax[r, 0].set_ylabel(lab)
    ax[0, 0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    ax[1, 0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    ax[3, 0].text(1, -INSERT_DEPTH * 1000 + 2, "success depth (30 mm below top)", color=INK2, fontsize=8)
    ax[3, 0].text(1, 2, "board top", color=INK2, fontsize=8)
    ax[3, 0].text(1, -38, "peg standing on table", color=INK2, fontsize=8)
    fig.suptitle("PegInsertion MimicGen demos: actions vs source seed.  Dashed = first grasp signal "
                 "(switch from peg-relative to board-relative segment)", fontsize=9, color=INK2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110, facecolor=fig.get_facecolor())

    # stitching check over ALL generated demos: largest one-step action jump
    # right after the grasp switch vs anywhere else in the demo
    jumps_sw, jumps_all = [], []
    for k in f["data"]:
        a, _, tg = traces(f[f"data/{k}"])
        d = np.abs(np.diff(a[:, :6], axis=0)).max(1)
        jumps_all.append(np.percentile(d, 99))
        if tg is not None:
            w = slice(max(tg - 2, 0), min(tg + 30, len(d)))
            jumps_sw.append(d[w].max())
    print(f"n={len(jumps_all)} demos; max |delta action| per step: near grasp switch median "
          f"{np.median(jumps_sw):.3f} max {np.max(jumps_sw):.3f}; whole-demo p99 median {np.median(jumps_all):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
