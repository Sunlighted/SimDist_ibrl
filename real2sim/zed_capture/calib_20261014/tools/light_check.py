"""Lighting variants of the 10-14 Robotiq cell vs the exposure-locked real home frame (2026-10-15).

For each variant: colour + light DR ON (the DR mean is what must match the robot), geometry DR off, real home q,
needle / tripod at the poses fitted in the scene camera (fit_layout_wrist.py), NSEED seeds. Reports the board
luminance median and p90-p10 spread (the shadow-depth criterion) per camera, the render cost, and writes one
sample per variant into a figure.

    python light_check.py --objects <fit json> [--nseed 16] [--out figs/light_check_1015.jpg]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "real2sim", "deploy"))
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import load_sim2real  # noqa: E402
import env.threading_variants as tv  # noqa: E402
from real2sim.deploy.preprocess import Rectifier  # noqa: E402
from live_crop_view import load_cams  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--frame", default=os.path.join(HERE, "data", "home_droid"))
ap.add_argument("--objects", required=True, help="json from fit_layout_wrist.py --json")
ap.add_argument("--nseed", type=int, default=16)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--variants", default="")
ap.add_argument("--out", default=os.path.join(HERE, "figs", "light_check_1015.jpg"))
args = ap.parse_args()
S = args.size
CAMS = ("agentview", "robot0_eye_in_hand")

VARIANTS = {
    "cell": {},
    "10-14 cell (5+-3 casters)": dict(light_shadow_n=5, dr_light_shadow_n=3),
    "no shadows": dict(light_shadow_n=0, dr_light_shadow_n=0),
}
if args.variants:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in args.variants.split(";")}

cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
GEOM_DR = ("dr_cam_pos", "dr_cam_rot_deg", "dr_cam_fovy_deg", "dr_base_height", "dr_base_yaw_deg",
           "dr_base_tilt_deg", "dr_base_xy")
objs = json.load(open(args.objects))["objects"][os.path.basename(args.frame.rstrip("/"))]
q = json.load(open(os.path.join(args.frame, "state.json")))["q"]
real = {}
for ros, c in load_cams(tv.REAL_CELL_ROBOTIQ_YAML).items():
    raw = cv2.imread(os.path.join(args.frame, f"{ros}_left.png"))
    real[c["sim"]] = Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])(raw)[..., ::-1].astype(float)
lum = lambda x: x @ np.array([0.299, 0.587, 0.114])


def bands(img, mk, sigma=12.0):
    """Board luminance split in two bands (iron rule 29: one p90-p10 cannot tell a soft room GRADIENT from
    hard shadow EDGES). low = p90-p10 of the blurred luminance (normalised-convolution inside the mask),
    high = mean |L - blur(L)| on the eroded board (edge energy)."""
    L = lum(img)
    w = cv2.GaussianBlur(mk.astype(float), (0, 0), sigma)
    Lb = cv2.GaussianBlur(L * mk, (0, 0), sigma) / np.maximum(w, 1e-6)
    core = er(mk, 9)
    return (float(np.median(L[mk])), float(np.subtract(*np.percentile(Lb[core], [90, 10]))),
            float(np.mean(np.abs(L - Lb)[core])))
er = lambda mk, k=5: cv2.erode(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def build(over):
    cfg = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras),
                              **{k: 0.0 for k in GEOM_DR}, **over)
    return PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                          camera_names=list(CAMS), rl_cameras=list(CAMS), use_state=True, obs_stack=1, state_stack=1,
                          prop_stack=1, flip_image=True, device="cpu", hand_init_perturb_joint_sigma=0.0, sim2real=cfg)


def pose(e):
    m, d = e.sim.model, e.sim.data
    for i in range(7):
        d.qpos[m.get_joint_qpos_addr(f"robot0_joint{i + 1}")] = q[i]
    base = d.body_xpos[m.body_name2id("robot0_base")]
    for obj, (x, y, yaw) in objs.items():
        a = m.get_joint_qpos_addr(f"{obj}_obj_joint0")[0]
        d.qpos[a:a + 2] = [base[0] + x, base[1] + y]
        d.qpos[a + 3:a + 7] = [np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]
    e.sim.forward()


rows, results = [], {}
real_stats = None
for name, over in VARIANTS.items():
    env = build(over)
    per = {c: [] for c in CAMS}
    sample = None
    t_render = []
    for seed in range(args.nseed):
        np.random.seed(seed)
        env.reset()
        e = env.env
        pose(e)
        m = e.sim.model
        board = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "") in ("table_visual", "table_collision")]
        imgs = {}
        t0 = time.time()
        for c in CAMS:
            imgs[c] = e.sim.render(camera_name=c, width=S, height=S)[::-1].astype(float)
        t_render.append(time.time() - t0)
        for c in CAMS:
            sg = e.sim.render(camera_name=c, width=S, height=S, segmentation=True)[::-1][..., 1]
            mk = er(np.isin(sg, board))
            per[c].append(bands(imgs[c], mk))
            if real_stats is None or c not in real_stats:
                real_stats = real_stats or {}
                real_stats[c] = bands(real[c], mk & (lum(real[c]) > 120))
        if seed == 0:
            sample = np.hstack([imgs[c] for c in CAMS])
    nl = env.env.sim.model.nlight
    res = {c: np.array(per[c]) for c in CAMS}
    results[name] = res
    ms = 1e3 * np.median(t_render[1:])
    print(f"{name:24s} render {ms:4.1f} ms | " + " | ".join(
        f"{c[:5]}: med {res[c][:, 0].mean():5.1f} low {res[c][:, 1].mean():5.1f} high {res[c][:, 2].mean():4.2f} "
        f"[{res[c][:, 2].min():.2f}-{res[c][:, 2].max():.2f}]" for c in CAMS))
    s = sample.astype(np.uint8)[..., ::-1].copy()
    cv2.putText(s, name, (6, 18), 0, 0.5, (0, 0, 255), 1)
    rows.append(s)
print(f"{'real (home_droid)':24s}           | " + " | ".join(
    f"{c[:5]}: med {real_stats[c][0]:5.1f} low {real_stats[c][1]:5.1f} high {real_stats[c][2]:4.2f}" for c in CAMS))
r = np.hstack([real[c] for c in CAMS]).astype(np.uint8)[..., ::-1].copy()
cv2.putText(r, "real", (6, 18), 0, 0.5, (0, 0, 255), 1)
cv2.imwrite(args.out, np.vstack([r] + rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote", args.out)
