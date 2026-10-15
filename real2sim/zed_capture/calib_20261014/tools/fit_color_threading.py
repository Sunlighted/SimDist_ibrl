"""Per-camera colour correction real ZED -> sim for the 10-14 Robotiq cell (2026-10-15).

Target = the sim's DR-MEAN image (colour + light DR on, geometry DR off, NSEED seeds averaged pixelwise) at the
real home q with needle / tripod at the poses fitted in the scene camera, i.e. what the policy saw on average.
Model = per-channel affine (out_k = a_k * in_k + b_k, the model real2sim/deploy/fit_color_correction.py settled
on), fitted on blurred pixels of the classes both images share: board, red objects, and in the wrist view the
gripper. Written in real2sim/deploy/color_correction.json's format, so policy_server.py --color_json takes it.

    python fit_color_threading.py --objects <fit json> [--out <json>] [--fig <jpg>]
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "real2sim", "deploy"))
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import load_sim2real  # noqa: E402
import env.threading_variants as tv  # noqa: E402
from real2sim.deploy.preprocess import Rectifier, apply_color  # noqa: E402
from live_crop_view import load_cams  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--frame", default=os.path.join(HERE, "data", "home_droid"))
ap.add_argument("--objects", required=True)
ap.add_argument("--check", default=os.path.join(HERE, "data", "exposure_locked"),
                help="exposure-locked frames of another moment (board only, no q). Only the SCENE camera is a "
                     "check: the wrist saw another part of the board there")
ap.add_argument("--nseed", type=int, default=16)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--out", default=os.path.join(HERE, "color_correction_threading_1014.json"))
ap.add_argument("--fig", default=os.path.join(HERE, "figs", "color_correction_threading_1015.jpg"))
args = ap.parse_args()
S = args.size
SIM2ROS = {"agentview": "scene", "robot0_eye_in_hand": "wrist"}

cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
GEOM_DR = ("dr_cam_pos", "dr_cam_rot_deg", "dr_cam_fovy_deg", "dr_base_height", "dr_base_yaw_deg",
           "dr_base_tilt_deg", "dr_base_xy")
cfg = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras),
                          **{k: 0.0 for k in GEOM_DR})
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                     camera_names=list(SIM2ROS), rl_cameras=list(SIM2ROS), use_state=True, obs_stack=1, state_stack=1,
                     prop_stack=1, flip_image=True, device="cpu", hand_init_perturb_joint_sigma=0.0, sim2real=cfg)
objs = json.load(open(args.objects))["objects"][os.path.basename(args.frame.rstrip("/"))]
q = json.load(open(os.path.join(args.frame, "state.json")))["q"]

acc = {c: np.zeros((S, S, 3)) for c in SIM2ROS}
segs = {}
for seed in range(args.nseed):
    np.random.seed(seed)
    env.reset()
    e = env.env
    m, d = e.sim.model, e.sim.data
    for i in range(7):
        d.qpos[m.get_joint_qpos_addr(f"robot0_joint{i + 1}")] = q[i]
    base = d.body_xpos[m.body_name2id("robot0_base")]
    for obj, (x, y, yaw) in objs.items():
        a = m.get_joint_qpos_addr(f"{obj}_obj_joint0")[0]
        d.qpos[a:a + 2] = [base[0] + x, base[1] + y]
        d.qpos[a + 3:a + 7] = [np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]
    e.sim.forward()
    for c in SIM2ROS:
        acc[c] += e.sim.render(camera_name=c, width=S, height=S)[::-1]
        if seed == 0:
            g = e.sim.render(camera_name=c, width=S, height=S, segmentation=True)[::-1][..., 1]
            cls = np.zeros(m.ngeom + 1, np.int8)
            for i in range(m.ngeom):
                n, b = m.geom_id2name(i) or "", m.body_id2name(int(m.geom_bodyid[i]))
                cls[i] = 1 if n in ("table_visual", "table_collision") else 2 if n.startswith(("needle", "tripod")) \
                    else 3 if b.startswith("gripper0") else 0
            segs[c] = cls[np.where(g >= 0, g, m.ngeom)]
sim = {c: acc[c] / args.nseed / 255.0 for c in SIM2ROS}

real = {}
for ros, c in load_cams(tv.REAL_CELL_ROBOTIQ_YAML).items():
    raw = cv2.imread(os.path.join(args.frame, f"{ros}_left.png"))
    real[c["sim"]] = Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])(raw)[..., ::-1]
er = lambda mk, k: cv2.erode(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)

out = {"cameras": {}}
report = {}
for c, ros in SIM2ROS.items():
    R = blur(real[c].astype(np.float64) / 255.0)
    T = blur(sim[c])
    redness = real[c][..., 0].astype(int) - np.maximum(real[c][..., 1], real[c][..., 2])
    classes = {"board": er(segs[c] == 1, 7) & (real[c].min(-1) > 110),
               "red": er(segs[c] == 2, 3) & (redness > 22),
               "gripper": er(segs[c] == 3, 7) & (real[c].max(-1) < 75)}
    if ros == "wrist":
        # the wrist hand-eye is ~1-1.6 cm / ~2 deg off (fit_layout_wrist.py): sim and real red pixels in this
        # view are not the same object, so the red class would fit a misregistration, not a colour
        classes.pop("red")
    classes = {k: v for k, v in classes.items() if v.sum() >= 40}
    # each class weighs the same (the board has 100x the pixels of the red objects)
    X, Y, W = [], [], []
    for k, mk in classes.items():
        X.append(R[mk]); Y.append(T[mk]); W.append(np.full(mk.sum(), 1.0 / mk.sum()))
    X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
    M = np.zeros((3, 4), np.float32)
    for ch in range(3):
        A = np.c_[X[:, ch], np.ones(len(X))] * np.sqrt(W)[:, None]
        a, b = np.linalg.lstsq(A, Y[:, ch] * np.sqrt(W), rcond=None)[0]
        M[ch, ch], M[ch, 3] = a, b
    out["cameras"][ros] = {"M": M.tolist(), "model": "per-channel affine, class-balanced (board/red/gripper)",
                           "classes": {k: int(v.sum()) for k, v in classes.items()}}
    corr = apply_color(real[c], M).astype(np.float64) / 255.0
    rep = {}
    for k, mk in classes.items():
        rep[k] = dict(sim=np.round(255 * np.median(sim[c][mk], 0), 1).tolist(),
                      real=np.round(255 * np.median(real[c][mk] / 255.0, 0), 1).tolist(),
                      corrected=np.round(255 * np.median(corr[mk], 0), 1).tolist())
    report[ros] = rep
    print(f"[{ros}] M diag {np.round(np.diag(M[:, :3]), 3).tolist()} offset {np.round(M[:, 3], 3).tolist()}")
    for k, v in rep.items():
        err0 = np.abs(np.array(v["real"]) - v["sim"]).mean()
        err1 = np.abs(np.array(v["corrected"]) - v["sim"]).mean()
        print(f"   {k:8s} sim {v['sim']}  real {v['real']} (|d| {err0:5.1f})  corrected {v['corrected']} (|d| {err1:4.1f})")

# held-out: the exposure-locked reference frames (different session moment, board only)
if args.check and os.path.isdir(args.check):
    for ros, c in load_cams(tv.REAL_CELL_ROBOTIQ_YAML).items():
        raw = cv2.imread(os.path.join(args.check, f"{ros}_left.png"))
        if raw is None:
            continue
        r = Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])(raw)[..., ::-1]
        mk = er(segs[c["sim"]] == 1, 7) & (r.min(-1) > 110)
        rc = apply_color(r, np.asarray(out["cameras"][ros]["M"], np.float32))
        sm = np.round(255 * np.median(sim[c["sim"]][mk], 0), 1)
        print(f"[held-out {ros}] board: sim {sm.tolist()}  real {np.median(r[mk], 0).tolist()}  "
              f"corrected {np.median(rc[mk], 0).tolist()}")
out["source"] = dict(frame=os.path.relpath(args.frame, REPO), cell=os.path.relpath(tv.REAL_CELL_ROBOTIQ_YAML, REPO),
                     nseed=args.nseed, objects=objs, report=report,
                     note="real ZED crop (after Rectifier) -> sim DR-mean; apply with policy_server.py --color_json")
json.dump(out, open(args.out, "w"), indent=1)
tiles = []
for c, ros in SIM2ROS.items():
    M = np.asarray(out["cameras"][ros]["M"], np.float32)
    tiles.append(np.hstack([real[c], apply_color(real[c], M), (255 * sim[c]).astype(np.uint8)]))
cv2.imwrite(args.fig, np.vstack(tiles)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", args.out, args.fig)
