"""Re-fit each demo's needle pose with the WRIST camera before the grasp (2026-10-17).

The frame-0 scene-camera fit (demo_layout/layout_demos_1015.json) is ~0.7 m away: 1 px at 224 ~ 3 mm, and the sim grasp
flips on 4 mm. Right before the grasp the wrist camera is ~10 cm from the handle (1 px at 448 ~ 0.25 mm) and the arm
pose is known (measured q, FK). Fit (x, y, yaw) of the needle, base frame, to: the scene view of frame 0 + the wrist
views of the last N frames before the gripper command closes (the real fingers do not touch the needle before that).

Masks: real "red" = R - G > 12 and R >= B (the handle in the gripper's shadow is RGB ~(20, 0, 12)); compared only
near the sim needle and outside the sim gripper (it occludes). Kinematic sim: arm at the measured q, Robotiq drivers
at the mapped knuckle, objects at the candidate pose.

    python refit_needle_wrist_1017.py [--raw realdata/threading_robotiq_1015] [--n_wrist 10] [--out ...json]
"""
import argparse
import dataclasses
import glob
import io
import json
import os
import sys

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.optimize import minimize

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
import env.threading_variants as tv  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import Sim2RealConfig, load_sim2real  # noqa: E402
from real2sim.deploy.preprocess import Rectifier, camera_intrinsics  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join(REPO, "realdata/threading_robotiq_1015"))
ap.add_argument("--layout", default=os.path.join(HERE, "demo_layout", "layout_demos_1015.json"))
ap.add_argument("--n_wrist", type=int, default=10, help="wrist frames before the close command")
ap.add_argument("--size", type=int, default=448)
ap.add_argument("--out", default=os.path.join(HERE, "demo_layout", "layout_demos_1015_wristrefit.json"))
ap.add_argument("--fig", default=os.path.join(HERE, "figs", "needle_wrist_refit_1017.jpg"))
args = ap.parse_args()
S = args.size
SCENE, WRIST = "agentview", "robot0_eye_in_hand"
CAM_RAW = {SCENE: "scene", WRIST: "wrist"}

cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64,
                     camera_names=[SCENE, WRIST], rl_cameras=[SCENE, WRIST], use_state=True, device="cpu",
                     hand_init_perturb_joint_sigma=0.0, sim2real=cell, env_kwargs={"exec_layer": False})
np.random.seed(0)
env.reset()
e = env.env
m, d = e.sim.model, e.sim.data
rob = e.robots[0]
jidx = np.array(rob._ref_joint_pos_indexes)
gq = [m.get_joint_qpos_addr(n) for n in tv.ROBOTIQ_DRIVER_JOINTS]
BASE = d.body_xpos[m.body_name2id("robot0_base")].copy()
names = [m.geom_id2name(i) or "" for i in range(m.ngeom)]
bodies = [m.body_id2name(int(m.geom_bodyid[i])) or "" for i in range(m.ngeom)]
CLS = np.zeros(m.ngeom + 1, np.int8)
for i, (n, b) in enumerate(zip(names, bodies)):
    CLS[i] = 2 if n.startswith("needle") else 3 if n.startswith("tripod") else 4 if b.startswith("gripper0") else 0
ZQ = {o: (m.get_joint_qpos_addr(f"{o}_obj_joint0")[0], float(d.qpos[m.get_joint_qpos_addr(f"{o}_obj_joint0")[0] + 2]))
      for o in ("needle", "tripod")}
CELLCAM = {c.name: c for c in cell.cameras}


def place(obj, x, y, yaw):
    a, z = ZQ[obj]
    d.qpos[a:a + 3] = [BASE[0] + x, BASE[1] + y, z]
    h = np.radians(yaw) / 2
    d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]


def seg(cam):
    im = e.sim.render(camera_name=cam, width=S, height=S, segmentation=True)[::-1][..., 1]
    return CLS[np.where(im >= 0, im, m.ngeom)]


def redmask(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return ((r - g) > 12) & (r >= b)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def dil(mk, k):
    return cv2.dilate(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


lay = {k: v for k, v in json.load(open(args.layout)).items() if not k.startswith("_")}
out = {"_source": "refit_needle_wrist_1017.py: needle (x, y, yaw) from the frame-0 scene view + the last "
                  f"{args.n_wrist} wrist views before the close; tripod kept from the scene fit"}
tiles = []
for fn in sorted(glob.glob(os.path.join(args.raw, "episode_*.h5"))):
    ep = os.path.basename(fn)[:-3]
    h = h5py.File(fn, "r")
    q = h["observation/joint_positions"][:]
    kn = h["observation/gripper_knuckle"][:]
    g = h["action/gripper_width_percent"][:]
    intr = json.loads(h.attrs["camera_intrinsics"])
    mid = 0.5 * (g.max() + g.min())
    tc = int(np.argmax(g < mid))                       # first step the command is past mid-range (closing)
    onset = int(np.argmax(kn > 0.02))                  # the real knuckle starts to move
    t_last = min(tc, onset) - 1
    frames = [(SCENE, 0)] + [(WRIST, t) for t in range(max(1, t_last - args.n_wrist + 1), t_last + 1)]
    rect = {}
    real = []
    for cam, t in frames:
        raw = CAM_RAW[cam]
        if raw not in rect:
            k = intr[raw]["K"]
            K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], float)
            D = np.array(intr[raw]["D"], float)
            K, D = camera_intrinsics(CELLCAM[cam], K, D, 1280, 720)
            rect[raw] = Rectifier(K, D, 1280, 720, S, CELLCAM[cam].resolved_fovy())
        img = np.array(Image.open(io.BytesIO(np.asarray(h[f"observation/images/{raw}"][t]).tobytes())).convert("RGB"))
        real.append(rect[raw](img))
    tripod = lay[ep]["tripod"]

    def set_frame(t):
        d.qpos[jidx] = q[t]
        for a_, v in zip(gq, tv.robotiq_real_to_sim(kn[t])):
            d.qpos[a_] = v

    def loss(p, detail=False):
        tot, per = 0.0, []
        for (cam, t), im in zip(frames, real):
            set_frame(t)
            place("needle", *p)
            place("tripod", *tripod)
            e.sim.forward()
            s = seg(cam)
            ns = s == 2
            ok = ~dil(s == 4, 5)                        # outside the sim gripper (it occludes)
            rr = redmask(im) & dil(ns, 41 if cam == WRIST else 21) & ~(s == 3) & ok
            v = 1 - iou(ns & ok, rr)
            w = 1.0 if cam == SCENE else 1.0 / args.n_wrist * 2.0     # scene ~ as much as all wrist frames / 2
            tot += w * v
            if detail:
                c = (np.argwhere(ns & ok).mean(0) - np.argwhere(rr).mean(0)) if (ns & ok).sum() > 30 and rr.sum() > 30 else None
                per.append((cam, t, 1 - v, None if c is None else np.round(c, 1).tolist()))
        return (tot, per) if detail else tot

    p0 = np.array(lay[ep]["needle"], float)
    l0, per0 = loss(p0, True)
    best = None
    for dyaw in (0.0, -4.0, 4.0):
        x0 = p0 + [0, 0, dyaw]
        r = minimize(loss, x0, method="Nelder-Mead", options=dict(
            xatol=1e-4, fatol=1e-5, maxiter=300,
            initial_simplex=[x0, x0 + [0.004, 0, 0], x0 + [0, 0.004, 0], x0 + [0, 0, 3.0]]))
        if best is None or r.fun < best.fun:
            best = r
    l1, per1 = loss(best.x, True)
    dxy = 1e3 * (best.x[:2] - p0[:2])
    out[ep] = {"needle": [round(float(v), 5) for v in best.x], "tripod": tripod,
               "scene_fit_needle": p0.tolist(), "moved_mm_deg": [round(float(dxy[0]), 1), round(float(dxy[1]), 1),
                                                                round(float(best.x[2] - p0[2]), 1)],
               "loss_before_after": [round(l0, 4), round(l1, 4)]}
    w0 = [f"{x[2]:.2f}" for x in per0 if x[0] == WRIST]
    w1 = [f"{x[2]:.2f}" for x in per1 if x[0] == WRIST]
    print(f"{ep}: close cmd t={tc}, knuckle onset t={onset} -> wrist frames {frames[1][1]}..{frames[-1][1]} | "
          f"needle moved {dxy.round(1).tolist()} mm, yaw {best.x[2] - p0[2]:+.1f} deg | scene IoU "
          f"{per0[0][2]:.2f} -> {per1[0][2]:.2f} | wrist IoU median {np.median([x[2] for x in per0[1:]]):.2f} -> "
          f"{np.median([x[2] for x in per1[1:]]):.2f} | last wrist centroid d px@{S} {per0[-1][3]} -> {per1[-1][3]}",
          flush=True)
    # figure: last wrist frame, real with sim needle outline before (yellow) / after (cyan)
    set_frame(frames[-1][1])
    ov = real[-1].copy()
    for p, col in ((p0, (255, 255, 0)), (best.x, (0, 255, 255))):
        place("needle", *p)
        place("tripod", *tripod)
        e.sim.forward()
        s = seg(WRIST) == 2
        edge = s & ~cv2.erode(s.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ov[edge] = col
    tiles.append(ov)
json.dump(out, open(args.out, "w"), indent=1)
cv2.imwrite(args.fig, cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
print("wrote", args.out, args.fig)
