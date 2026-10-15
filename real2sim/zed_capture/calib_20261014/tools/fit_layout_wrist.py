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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
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
OBJ_K = dict(needle=2, tripod=3)


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
holds = [Frame(p) for p in args.hold]
for fr in fits + holds:
    fit_objects(fr)

# ---------------------------------------------------------------- 2. wrist camera + mount yaw
wcam = [c for c in nom.cameras if c.name == WRIST][0]
T0 = np.eye(4)
T0[:3, :3] = Rot.from_quat(np.r_[wcam.quat[1:], wcam.quat[0]]).as_matrix()
T0[:3, 3] = wcam.pos
AD = m.body_name2id("gripper0_robotiq_85_adapter_link")
AD_QUAT0 = m.body_quat[AD].copy()
AD_POS0 = m.body_pos[AD].copy()
CUR_CAM = [wcam]
FREE = set(args.free.split(","))
NP = 8                                            # rx ry rz tx ty tz | dyaw | dz


def set_all(p):
    """p = (rx, ry, rz deg, tx, ty, tz m) camera-frame correction, (dyaw deg) Robotiq mount about the flange z,
    (dz m) Robotiq mount along the flange z (the cell's gripper_mount_z)."""
    m.body_pos[AD] = AD_POS0 + [0, 0, p[7]]
    dT = np.eye(4)
    dT[:3, :3] = Rot.from_euler("xyz", p[:3], degrees=True).as_matrix()
    dT[:3, 3] = p[3:6]
    T = T0 @ dT
    qx, qy, qz, qw = Rot.from_matrix(T[:3, :3]).as_quat()
    CUR_CAM[0] = dataclasses.replace(wcam, pos=tuple(T[:3, 3]), quat=(qw, qx, qy, qz))
    # mount: q = Rz(dyaw) (x) q0, i.e. an extra turn about the hand z (the flange axis)
    w0, x0, y0, z0 = AD_QUAT0
    c, s_ = np.cos(np.radians(p[6]) / 2), np.sin(np.radians(p[6]) / 2)
    m.body_quat[AD] = [c * w0 - s_ * z0, c * x0 - s_ * y0, c * y0 + s_ * x0, c * z0 + s_ * w0]
    return T


def seat(fr):
    fr.apply()
    _apply_camera(e, m, e.sim, CUR_CAM[0], nom, np.random)
    e.sim.forward()


def wrist_terms(fr):
    seat(fr)
    s = seg(WRIST)
    out = dict(gripper=float(iou(s == 4, fr.dark)))
    for obj, k in OBJ_K.items():
        mine = s == k
        if mine.sum() < 30:
            out[obj] = None
            continue
        rr = fr.red[WRIST] & near(mine, 41) & (s != 4)
        cy, cx = np.argwhere(mine).mean(0)
        ry, rx = np.argwhere(rr).mean(0) if rr.any() else (np.nan, np.nan)
        out[obj] = dict(iou=float(iou(mine, rr)), off_px=float(np.hypot(cx - rx, cy - ry)) * PX)
    return out


def loss(p7):
    set_all(p7)
    L = 0.0
    for fr in fits + holds:                      # the gripper silhouette of EVERY frame (camera-vs-gripper)
        L += 1 - wrist_terms(fr)["gripper"]
    for fr in fits:                              # world objects only from the FIT frames
        t = wrist_terms(fr)
        for obj in OBJ_K:
            L += 1 - (t[obj]["iou"] if t[obj] else 0.0)
    return L


def report(tag, p7):
    set_all(p7)
    for fr in fits + holds:
        t = wrist_terms(fr)
        role = "fit " if fr in fits else "HOLD"
        objs = "  ".join(f"{o} " + ("not in view" if t[o] is None else f"IoU {t[o]['iou']:.3f} off {t[o]['off_px']:.1f}px")
                         for o in OBJ_K)
        print(f"[wrist {tag:6s}] {role} {fr.name:16s} gripper IoU {t['gripper']:.3f}  {objs}")


p0 = np.zeros(NP)
# the dark mask also catches the breadboard / ChArUco squares: keep only the gripper's neighbourhood
set_all(p0)
for fr in fits + holds:
    seat(fr)
    fr.dark &= near(seg(WRIST) == 4, 61)
report("before", p0)
idx = [i for i, on in enumerate([*(["cam" in FREE] * 6), "yaw" in FREE, "dz" in FREE]) if on]
STEP = np.r_[1.0, 1.0, 1.0, 0.004, 0.004, 0.004, 1.0, 0.003]


def full(px):
    p = np.zeros(NP)
    p[idx] = px
    return p


best = None
starts = [np.zeros(NP)] + [np.eye(NP)[i] * STEP[i] * s for i in idx for s in (-1.5, 1.5)]
for start in starts:
    s0 = start[idx]
    simplex = [s0] + [s0 + np.eye(len(idx))[j] * STEP[i] for j, i in enumerate(idx)]
    r = minimize(lambda px: loss(full(px)), s0, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-5, maxiter=4000, initial_simplex=simplex))
    if best is None or r.fun < best.fun:
        best = r
pbest = full(best.x)
report("after", pbest)
T = set_all(pbest)
qx, qy, qz, qw = Rot.from_matrix(T[:3, :3]).as_quat()
print(f"[wrist] correction: rot {np.round(pbest[:3], 2)} deg, trans {np.round(1e3 * pbest[3:6], 1)} mm "
      f"(|t| {1e3 * np.linalg.norm(pbest[3:6]):.1f} mm), Robotiq mount yaw {pbest[6]:+.2f} deg, "
      f"mount z {1e3 * pbest[7]:+.1f} mm (=> gripper_mount_z {AD_POS0[2] + pbest[7]:.4f})  loss {best.fun:.3f}")
print(f"[wrist] new hand-frame pose: pos {np.round(T[:3, 3], 5).tolist()}  quat(wxyz) {np.round([qw, qx, qy, qz], 5).tolist()}")


# ---------------------------------------------------------------- figure: per frame, wrist before | after | scene fit
def overlay(img_rgb, real_bgr, masks):
    o = (0.5 * img_rgb[..., ::-1] + 0.5 * real_bgr).astype(np.uint8)
    for mk, col in masks:
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, col, 2)
    return o


rows = []
for fr in fits + holds:
    tiles = []
    for p in (p0, pbest):
        set_all(p)
        seat(fr)
        s = seg(WRIST)
        tiles.append(overlay(rgb(WRIST), fr.real[WRIST], [(s == 4, (0, 165, 255)), ((s == 2) | (s == 3), (255, 255, 0))]))
    s = seg(SCENE)
    tiles.append(overlay(rgb(SCENE), fr.real[SCENE], [((s == 2) | (s == 3), (255, 255, 0))]))
    cv2.putText(tiles[0], ("FIT " if fr in fits else "HOLD-OUT ") + fr.name, (8, 24), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(tiles[0], "wrist: cell", (8, 50), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(tiles[1], "wrist: fitted", (8, 24), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(tiles[2], "scene: object fit", (8, 24), 0, 0.7, (0, 0, 255), 2)
    rows.append(np.hstack(tiles))
cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", args.out)
if args.json:
    json.dump(dict(objects={fr.name: fr.obj for fr in fits + holds}, correction=pbest.tolist(),
                   wrist_pos=T[:3, 3].tolist(), wrist_quat_wxyz=[qw, qx, qy, qz]), open(args.json, "w"), indent=1)
