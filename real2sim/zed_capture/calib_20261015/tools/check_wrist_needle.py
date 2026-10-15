"""Render-based fits on real frames (2026-10-15), MuJoCo geometry throughout.

  1. per frame: needle + tripod pose (base x, y, yaw) from the SCENE camera (validated: fingertips 1-2 px),
     maximising IoU(sim mask, real red mask);
  2. wrist camera correction (6 DoF, in the camera frame, on top of the cell's hand-eye) + Robotiq mount yaw
     correction (1 DoF). The gripper silhouette alone cannot tell the two apart (camera-vs-gripper only), so
     the fit also uses the needle / tripod in the WRIST view of the FIT frames (camera-vs-world);
  3. held-out: needle / tripod in the wrist view of the HOLD-OUT frame, never used in 2.

    python fit_layout_wrist.py [--fit data/layout_shot_000] [--hold data/home_droid] [--out figs/...]
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.expanduser("~/simdist/deploy_simdist_1015")
sys.path.insert(0, REPO); sys.path.insert(0, "/home/tisl-robotics/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261014/tools")
sys.path.insert(0, os.path.join(REPO, "real2sim", "deploy"))
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import load_sim2real, _apply_camera  # noqa: E402
import env.threading_variants as tv  # noqa: E402
from real2sim.deploy.preprocess import Rectifier  # noqa: E402
from live_crop_view import load_cams  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--fit", nargs="+", default=[os.path.join(HERE, "data", "layout_shot_000")])
ap.add_argument("--hold", nargs="*", default=[os.path.join(HERE, "data", "home_droid")])
ap.add_argument("--out", default=os.path.join(HERE, "figs", "fit_layout_wrist_1015.jpg"))
ap.add_argument("--json", default="")
ap.add_argument("--size", type=int, default=448)
ap.add_argument("--free", default="cam,yaw", help="comma list of cam, yaw, dz: which corrections are fitted")
args = ap.parse_args()
S = args.size
WRIST, SCENE = "robot0_eye_in_hand", "agentview"
PX = 224 / S                                     # report pixel errors at the policy's 224

cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
zero = {f.name: f.default for f in dataclasses.fields(cell) if f.name.startswith("dr_")
        and not isinstance(f.default, dataclasses._MISSING_TYPE)}
nom = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **zero)
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                     camera_names=[SCENE, WRIST], rl_cameras=[SCENE, WRIST], use_state=True, obs_stack=1,
                     state_stack=1, prop_stack=1, flip_image=True, device="cpu", hand_init_perturb_joint_sigma=0.0,
                     sim2real=nom)
np.random.seed(0)
env.reset()
e = env.env
m, d = e.sim.model, e.sim.data
BASE = d.body_xpos[m.body_name2id("robot0_base")].copy()
assert np.allclose(d.body_xmat[m.body_name2id("robot0_base")].reshape(3, 3), np.eye(3), atol=1e-6)

names = [m.geom_id2name(i) or "" for i in range(m.ngeom)]
bodies = [m.body_id2name(int(m.geom_bodyid[i])) for i in range(m.ngeom)]
CLS = np.zeros(m.ngeom + 1, np.int8)            # last index = background
for i, (n, b) in enumerate(zip(names, bodies)):
    CLS[i] = 2 if n.startswith("needle") else 3 if n.startswith("tripod") else 4 if b.startswith("gripper0") else 0
OBJ_K = dict(needle=2)


def seg(cam):
    s = e.sim.render(camera_name=cam, width=S, height=S, segmentation=True)[::-1][..., 1]
    return CLS[np.where(s >= 0, s, m.ngeom)]


def rgb(cam):
    return e.sim.render(camera_name=cam, width=S, height=S)[::-1]


ZQ = {}
for obj in OBJ_K:
    a = m.get_joint_qpos_addr(f"{obj}_obj_joint0")[0]
    ZQ[obj] = (a, float(d.qpos[a + 2]))


def place(obj, x, y, yaw_deg):
    a, z = ZQ[obj]
    d.qpos[a:a + 3] = [BASE[0] + x, BASE[1] + y, z]
    h = np.radians(yaw_deg) / 2
    d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]


def redmask(bgr):
    # "redness" R - max(G, B), not an HSV gate: the needle SHAFT is a dark maroon (V < 40) that an HSV gate
    # drops, which leaves only the handle cube and makes the yaw unobservable (a first fit went to yaw -150
    # with the shaft pointing the wrong way).
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return (r - np.maximum(g, b)) > 22


def darkmask(bgr):
    return (cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] < 75) & ~redmask(bgr)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


near = lambda mk, k: cv2.dilate(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
CAMS = load_cams(tv.REAL_CELL_ROBOTIQ_YAML)


class Frame:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path.rstrip("/"))
        self.q = json.load(open(os.path.join(path, "state.json")))["q"]
        self.real = {}
        for ros, c in CAMS.items():
            raw = cv2.imread(os.path.join(path, f"{ros}_left.png"))
            self.real[c["sim"]] = Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])(raw)
        self.red = {c: redmask(v) for c, v in self.real.items()}
        self.dark = darkmask(self.real[WRIST])     # restricted to the gripper's neighbourhood below
        self.obj = {}

    def apply(self):
        for i in range(7):
            d.qpos[m.get_joint_qpos_addr(f"robot0_joint{i + 1}")] = self.q[i]
        for obj, p in self.obj.items():
            place(obj, *p)
        e.sim.forward()


# ---------------------------------------------------------------- 1. objects from the scene camera
INIT = dict(needle=(0.553, -0.108, -104.0), tripod=(0.5269, 0.1758, -40.6))


def fit_objects(fr):
    fr.apply()
    for obj, k in OBJ_K.items():
        place(obj, *INIT[obj])
        e.sim.forward()
        roi = near(seg(SCENE) == k, 61)

        def loss(p):
            place(obj, *p)
            e.sim.forward()
            s = seg(SCENE)
            return 1 - iou(s == k, fr.red[SCENE] & roi & ~(s == (5 - k)))

        # the thin shaft barely moves the IoU: scan the full turn before the local search
        grid = sorted((loss(np.r_[INIT[obj][0] + dx, INIT[obj][1] + dy, yaw]), (INIT[obj][0] + dx, INIT[obj][1] + dy, yaw))
                      for yaw in np.arange(-180, 180, 7.5) for dx in (-0.01, 0, 0.01) for dy in (-0.01, 0, 0.01))
        best = None
        for _, p0 in grid[:4]:
            p0 = np.array(p0)
            r = minimize(loss, p0, method="Nelder-Mead", options=dict(
                xatol=2e-4, fatol=1e-4, maxiter=400,
                initial_simplex=[p0, p0 + [0.006, 0, 0], p0 + [0, 0.006, 0], p0 + [0, 0, 6]]))
            if best is None or r.fun < best.fun:
                best = r
        fr.obj[obj] = tuple(float(v) for v in best.x)
        place(obj, *best.x)
        e.sim.forward()
        print(f"[scene {fr.name}] {obj:6s} base ({best.x[0]:.4f}, {best.x[1]:.4f}) yaw {best.x[2]:7.1f}  IoU {1 - best.fun:.3f}")


fits = [Frame(p) for p in args.fit]
holds = []
fit_objects(fits[0])                      # the needle and the scene camera are static: fit once
for fr in fits[1:]:
    fr.obj = dict(fits[0].obj)


wcam = [c for c in nom.cameras if c.name == WRIST][0]
NEW = json.load(open(args.json))
poses = {"cell": wcam, "new": dataclasses.replace(wcam, pos=tuple(NEW["pos"]), quat=tuple(NEW["quat"]))}
res = {k: [] for k in poses}
rows = []
for fr in fits:
    tiles = []
    for k, cam in poses.items():
        fr.apply()
        _apply_camera(e, m, e.sim, cam, nom, np.random)
        e.sim.forward()
        s = seg(WRIST)
        g = iou(s == 4, fr.dark & near(s == 4, 61))
        mine = s == 2
        if mine.sum() < 30:
            res[k].append((fr.name, g, None, None)); continue
        rr = fr.red[WRIST] & near(mine, 41) & (s != 4)
        cy, cx = np.argwhere(mine).mean(0)
        ry, rx = np.argwhere(rr).mean(0) if rr.any() else (np.nan, np.nan)
        res[k].append((fr.name, g, iou(mine, rr), np.hypot(cx - rx, cy - ry) * PX))
        o = (0.5 * rgb(WRIST)[..., ::-1] + 0.5 * fr.real[WRIST]).astype(np.uint8)
        for mk, col in ((s == 4, (0, 165, 255)), (mine, (255, 255, 0))):
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 2)
        cv2.putText(o, f"{fr.name} {k}", (8, 24), 0, 0.7, (0, 0, 255), 2)
        tiles.append(cv2.resize(o, (224, 224)))
    if len(tiles) == 2:
        rows.append(np.hstack(tiles))
for k in poses:
    v = [r for r in res[k] if r[2] is not None]
    print(f"{k:5s}: gripper IoU median {np.median([r[1] for r in res[k]]):.3f} | needle in view {len(v)}/{len(res[k])}: "
          f"IoU median {np.median([r[2] for r in v]):.3f}, offset median {np.median([r[3] for r in v]):.1f} px @224, "
          f"p90 {np.percentile([r[3] for r in v], 90):.1f}")
for a, b in zip(res["cell"], res["new"]):
    print(a[0], f"grip {a[1]:.2f}->{b[1]:.2f}", "needle", "-" if a[2] is None else f"IoU {a[2]:.2f}->{b[2]:.2f} off {a[3]:.1f}->{b[3]:.1f}px")
if rows:
    cv2.imwrite(args.out, np.vstack([np.hstack(rows[i:i + 3]) if len(rows[i:i + 3]) == 3 else np.hstack(rows[i:i + 3] + [np.zeros_like(rows[0])] * (3 - len(rows[i:i + 3]))) for i in range(0, len(rows), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
