"""Per-demo object placements for REAL stage-2 teleop, in TAPE-MEASURE terms.

The real operator cannot place objects "at base-frame (0.42, 0.31)". This turns
the sim reset distribution into distances from PHYSICAL table edges and headings
in degrees, and samples it the way sim does -- uniform -- but STRATIFIED (Latin
hypercube), so N demos cover the region evenly instead of clumping.

★ The distribution is READ FROM THE CELL YAML (`layout_*`), never restated here.
An earlier version hard-coded it and went stale the moment the layout changed:
it would have told the operator to put the needle on the robot's LEFT and the
tripod at a fixed unrotated pose, which is the PREVIOUS task. One source of
truth -- the same file train_rl.py and Threading_RealCell read.

Frame used for the numbers (robot's point of view, looking out from the base):
  forward = away from the robot;  LEFT = the robot's left (+y).
  "near edge" = the table edge closest to the robot.

Reference points / headings:
  needle : midpoint of its FULL 16 cm length (tip -> far face of the handle).
           heading = direction from tip to handle; 0 deg = handle pointing
           straight AWAY from the robot, + = turned to the robot's LEFT.
  tripod : the point on the table directly BELOW the ring centre.
           heading = the direction the RING'S HOLE AXIS points, 0 deg = straight
           away from the robot, + = to the robot's LEFT. This one matters: it
           decides which side the arm inserts from (docs/infra.md 14f).

  python scripts/make_placement_schedule.py --n 50 --seed 0 --out placements.csv
  python scripts/make_placement_schedule.py --plot figs/sim2real/placement_map.png
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELL = "release/cfgs/sim2real/threading_real_cell.yaml"


def load_layout(path=CELL):
    """Table geometry + the object distribution, straight out of the cell."""
    from env.sim2real import load_sim2real
    cfg = load_sim2real(path)
    assert cfg.base_xy is not None and cfg.table_full_size is not None, path
    centre = -np.asarray(cfg.base_xy, dtype=float)          # table centre in BASE frame
    size = np.asarray(cfg.table_full_size[:2], dtype=float)  # x = depth, y = width
    spec = {}
    for obj in ("needle", "tripod"):
        x = getattr(cfg, f"layout_{obj}_x")
        y = getattr(cfg, f"layout_{obj}_y")
        r = getattr(cfg, f"layout_{obj}_z_rot_deg")
        assert x is not None and y is not None and r is not None, (
            f"{path}: layout_{obj}_* missing -- this script only supports cells "
            "that state the layout explicitly (workspace_offset alone is the "
            "pre-2026-09 recipe and is not converted here)")
        spec[obj] = dict(x=tuple(map(float, x)), y=tuple(map(float, y)),
                         z_rot_deg=tuple(map(float, r)))
    return centre, size, spec


def lhs(n, k, rng):
    """n stratified samples in [0,1)^k (one per stratum on every axis)."""
    return np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(k)], 1)


def schedule(n, seed, path=CELL):
    centre, size, spec = load_layout(path)
    near_edge_x = centre[0] - size[0] / 2
    left_edge_y = centre[1] + size[1] / 2
    rng = np.random.default_rng(seed)
    draws = {o: lhs(n, 3, rng) for o in ("needle", "tripod")}
    rows = []
    for i in range(n):
        row = {"demo": i}
        for obj in ("needle", "tripod"):
            s, u = spec[obj], draws[obj][i]
            x = s["x"][0] + u[0] * (s["x"][1] - s["x"][0])
            y = s["y"][0] + u[1] * (s["y"][1] - s["y"][0])
            z = s["z_rot_deg"][0] + u[2] * (s["z_rot_deg"][1] - s["z_rot_deg"][0])
            # needle body +y runs tip -> handle, so its heading from "straight
            # away" is z_rot + 90; the tripod's hole axis IS its z_rot.
            head = z + 90.0 if obj == "needle" else z
            row.update({
                f"{obj}_from_near_edge_cm": round(100 * (x - near_edge_x), 1),
                f"{obj}_from_left_edge_cm": round(100 * (left_edge_y - y), 1),
                f"{obj}_left_of_centreline_cm": round(100 * (y - centre[1]), 1),
                f"{obj}_heading_deg": round(float(head), 1),
                f"{obj}_x_base": round(float(x), 4),
                f"{obj}_y_base": round(float(y), 4),
                f"{obj}_zrot_sim_deg": round(float(z), 1),
            })
        rows.append(row)
    return rows


def plot(path, cell=CELL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    centre, size, spec = load_layout(cell)
    near_edge_x = centre[0] - size[0] / 2
    left_edge_y = centre[1] + size[1] / 2

    def P(xb, yb):        # operator axes: robot at the bottom, its LEFT to the left
        return (-yb, xb)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.add_patch(Rectangle(P(near_edge_x, left_edge_y), size[1], size[0],
                           fc="#efe9dc", ec="k", lw=1.5))
    ax.plot(*P(0.0, 0.0), "ks", ms=14)
    ax.annotate("robot base", P(0.0, 0.0), np.array(P(0.0, 0.0)) + (0.05, -0.06), fontsize=10)
    cx = P(0, centre[1])[0]
    ax.plot([cx, cx], [near_edge_x, near_edge_x + size[0]], "k:", lw=1)
    ax.text(cx + 0.01, near_edge_x + size[0] - 0.03, "table centreline", fontsize=8)

    for obj, col in (("needle", "#8b2c2c"), ("tripod", "#1f4e8b")):
        s = spec[obj]
        ax.add_patch(Rectangle(P(s["x"][0], s["y"][1]),
                               s["y"][1] - s["y"][0], s["x"][1] - s["x"][0],
                               fc=col, alpha=0.20, ec=col, lw=1.5, ls="--"))
        mx, my = np.mean(s["x"]), np.mean(s["y"])
        lo, hi = s["z_rot_deg"]
        for z, a in ((lo, 1.0), (hi, 1.0)) if lo != hi else ((lo, 1.0),):
            head = z + 90.0 if obj == "needle" else z
            h = np.radians(head)
            dvec = np.array([np.cos(h), np.sin(h)])
            if obj == "needle":     # draw the shaft, blob = handle end
                tip = np.array([mx, my]) - 0.08 * dvec
                ax.plot(*zip(P(*tip), P(*(np.array([mx, my]) + 0.04 * dvec))),
                        color=col, lw=3, alpha=a)
                ax.add_patch(Circle(P(*(np.array([mx, my]) + 0.06 * dvec)), 0.018,
                                    fc=col, ec="k", lw=0.5))
            else:                   # draw the hole axis through the ring centre
                p0 = np.array([mx, my]) - 0.09 * dvec
                p1 = np.array([mx, my]) + 0.09 * dvec
                ax.annotate("", P(*p1), P(*p0),
                            arrowprops=dict(arrowstyle="<->", color=col, lw=2, alpha=a))
                ax.add_patch(Circle(P(mx, my), 0.022, fc="none", ec=col, lw=2))
        note = (f"needle midpoint\nhandle heading {lo+90:.0f}..{hi+90:.0f} deg"
                if obj == "needle" else
                f"ring centre (above +)\nHOLE AXIS {lo:.0f}"
                + (f"..{hi:.0f}" if lo != hi else "") + " deg")
        ax.text(*P(s["x"][1] + 0.02, s["y"][1]), note, fontsize=8, color=col, va="bottom")

    # Why the ring faces the way it does: the arm always inserts from the side
    # the hole axis points at, so that side must NOT be the camera's (infra 14f).
    from env.sim2real import load_sim2real
    cfg = load_sim2real(cell)
    cam = next((c for c in cfg.cameras if c.name == "agentview" and c.pos), None)
    if cam is not None:
        cxy = np.asarray(cam.pos[:2], dtype=float)
        ax.plot(*P(*cxy), "k^", ms=11)
        ax.annotate("scene camera", P(*cxy), np.array(P(*cxy)) + (0.02, -0.05), fontsize=9)
        t = spec["tripod"]
        rc = np.array([np.mean(t["x"]), np.mean(t["y"])])
        h = np.radians(np.mean(t["z_rot_deg"]))
        arm = rc + 0.20 * np.array([np.cos(h), np.sin(h)])
        ax.annotate("", P(*rc), P(*arm),
                    arrowprops=dict(arrowstyle="->", color="#2f7a2f", lw=2))
        ax.text(*P(*(arm + (0.01, -0.04))), "arm inserts\nfrom here", fontsize=8,
                color="#2f7a2f")
        ax.plot(*zip(P(*rc), P(*cxy)), color="k", ls=":", lw=1, alpha=0.5)

    ax.set_aspect("equal")
    pad = 0.08
    ax.set_xlim(-(left_edge_y + pad), -(centre[1] - size[1] / 2 - pad))
    ax.set_ylim(-0.12, near_edge_x + size[0] + pad)
    ax.set_xlabel("← robot's LEFT          (metres)          robot's RIGHT →")
    ax.set_ylabel("away from robot (m)")
    ax.set_title(f"Threading real cell placement (top view, robot at bottom)\n"
                 f"headings: 0 deg = straight away from the robot, + = to its LEFT\n"
                 f"read from {os.path.basename(cell)}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    centre, size, spec = load_layout(args.cell)
    print(f"[cell] {args.cell}\n[cell] table centre in base frame {np.round(centre,4)}, "
          f"size {np.round(size,3)} m")
    for o, s in spec.items():
        print(f"[cell] {o:7s} x {s['x']}  y {s['y']}  z_rot {s['z_rot_deg']} deg")
    if args.plot:
        plot(args.plot, args.cell)
    if args.out:
        rows = schedule(args.n, args.seed, args.cell)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out} ({args.n} placements, seed {args.seed})")
        for r in rows[:3]:
            print("  ", {k: v for k, v in r.items() if not k.endswith("_base")})


if __name__ == "__main__":
    main()
