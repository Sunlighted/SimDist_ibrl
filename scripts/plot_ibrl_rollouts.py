"""Plot IBRL PegInsertion rollouts saved by verify_peg_ibrl.py --save_states.

Per episode (one column): executed translation action vs what the BC prior
would have done, gripper, and peg tip height above the board top. Shaded
steps = IBRL executed the ACTOR's action (Q(actor) > Q(prior)).

    python scripts/plot_ibrl_rollouts.py --rollouts <hdf5> --out viz/.../ibrl_actions.png
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURF, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#f3d9cc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--eps", default="0,1,2,3")
    a = p.parse_args()
    f = h5py.File(a.rollouts, "r")
    ids = [int(x) for x in a.eps.split(",")]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(4, len(ids), figsize=(3.6 * len(ids), 8.5), sharex="col", sharey="row",
                           facecolor=SURF, layout="constrained")
    rows = ["translation: executed", "translation: BC prior would do", "gripper (executed)",
            "peg tip height above board top (mm)"]
    for c, i in enumerate(ids):
        g = f[f"data/demo_{i}"]
        act, prior = g["actions"][()], g["prior_actions"][()]
        chosen, tip = g["actor_chosen"][()], g["tip_height"][()] * 1000
        t = np.arange(len(act))
        for k, lab in enumerate("xyz"):
            ax[0, c].plot(t, act[:, k], color=C[k], lw=1.2, label=lab)
            ax[1, c].plot(t, prior[:, k], color=C[k], lw=1.2, label=lab)
        ax[2, c].plot(t, act[:, 6], color=C[0], lw=1.2)
        ax[3, c].plot(t, tip, color=C[0], lw=1.5)
        ax[3, c].axhline(0, color=INK2, lw=0.8)
        ax[3, c].axhline(-30, color=INK2, lw=0.8, ls=":")
        ok = int(g.attrs["success"])
        ax[0, c].set_title(f"episode {i}  ({'success' if ok else 'fail'}, actor {chosen.mean():.0%} of steps)",
                           fontsize=9, color=INK)
        for r in range(4):
            x = ax[r, c]
            x.set_facecolor(SURF)
            # shade the steps where the ACTOR's action was executed
            on = np.flatnonzero(np.diff(np.r_[0, chosen, 0]) == 1)
            off = np.flatnonzero(np.diff(np.r_[0, chosen, 0]) == -1)
            for s0, s1 in zip(on, off):
                x.axvspan(s0, s1, color=SHADE, lw=0, zorder=0)
            x.grid(True, color=GRID, lw=0.6)
            x.set_axisbelow(True)
            for sp in ("top", "right"):
                x.spines[sp].set_visible(False)
        ax[3, c].set_xlabel("step (20 Hz)")
    for r, lab in enumerate(rows):
        ax[r, 0].set_ylabel(lab)
    ax[0, 0].legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    ax[3, 0].text(1, 2, "board top", color=INK2, fontsize=8)
    ax[3, 0].text(1, -28, "success depth", color=INK2, fontsize=8)
    fig.suptitle("IBRL rollouts (peg, expert-demo prior). Shaded = actor's action executed instead of the BC prior's",
                 fontsize=9, color=INK2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=110, facecolor=SURF)

    # where does the actor take over? share of actor steps by phase, all saved episodes
    ph = {"peg low (tip >21 mm below board top: on the table, or deep in the pocket)": [], "carrying (tip above board)": [],
          "near the board top (-21..+5 mm: lifting past it, or at the mouth)": []}
    for k in f["data"]:
        g = f[f"data/{k}"]
        ch, tip = g["actor_chosen"][()], g["tip_height"][()] * 1000
        n = min(len(ch), len(tip))
        ch, tip = ch[:n], tip[:n]
        ph["peg low (tip >21 mm below board top: on the table, or deep in the pocket)"].append(ch[tip < -21])
        ph["carrying (tip above board)"].append(ch[tip > 5])
        ph["near the board top (-21..+5 mm: lifting past it, or at the mouth)"].append(ch[(tip <= 5) & (tip >= -21)])
    for name, v in ph.items():
        v = np.concatenate(v) if v else np.array([])
        print(f"  actor share {name}: {v.mean() if len(v) else float('nan'):.2f} (steps {len(v)})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
