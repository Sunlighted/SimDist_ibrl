"""Sim vs real at chosen frames of the converted Robotiq demos, kinematically (2026-10-16).

Arm at the MEASURED q of that frame, gripper at the mapped real knuckle, objects at the demo's frame-0
fit (demo_layout/layout_demos_1015.json) -- valid only before the arm touches them. Renders the scene and
wrist views at 224 (the converter's crops = the server's) and compares masks with the real frames:
  needle / tripod: sim segmentation vs real "redness" (fit_layout_wrist.py redmask)
  gripper (wrist view): sim segmentation vs real dark pixels near the sim gripper
Frames: 0 (initialisation) and the last frame before the gripper command closes (pre-grasp).

  python demo_frame_check_1016.py --dataset .../dataset.hdf5 --out figs/demo_frame_check_1016.jpg
"""
import argparse
import dataclasses
import io
import json
import os
import sys

import cv2
import h5py
import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, REPO)
import env.threading_variants as tv  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import Sim2RealConfig, load_sim2real  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="/user_data/Chenyang/real_stage2/threading_robotiq_1015/dataset.hdf5")
ap.add_argument("--layout", default=os.path.join(HERE, "demo_layout", "layout_demos_1015.json"))
ap.add_argument("--out", default=os.path.join(HERE, "figs", "demo_frame_check_1016.jpg"))
args = ap.parse_args()
S = 224
SCENE, WRIST = "agentview", "robot0_eye_in_hand"

cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                     camera_names=[SCENE, WRIST], rl_cameras=[SCENE, WRIST], use_state=True, device="cpu",
                     hand_init_perturb_joint_sigma=0.0, sim2real=cell)
np.random.seed(0)
env.reset()
e = env.env
sim, m, d = e.sim, e.sim.model, e.sim.data
rob = e.robots[0]
jidx = np.array(rob._ref_joint_pos_indexes)
gq = [m.get_joint_qpos_addr(n) for n in tv.ROBOTIQ_DRIVER_JOINTS]
BASE = d.body_xpos[m.body_name2id("robot0_base")].copy()
names = [m.geom_id2name(i) or "" for i in range(m.ngeom)]
bodies = [m.body_id2name(int(m.geom_bodyid[i])) or "" for i in range(m.ngeom)]
CLS = np.zeros(m.ngeom + 1, np.int8)
for i, (n, b) in enumerate(zip(names, bodies)):
    CLS[i] = (5 if n.startswith("needle_obj_handle") else 2 if n.startswith("needle") else 3 if n.startswith("tripod")
              else 4 if b.startswith("gripper0") else 0)
ZQ = {o: (m.get_joint_qpos_addr(f"{o}_obj_joint0")[0], float(d.qpos[m.get_joint_qpos_addr(f"{o}_obj_joint0")[0] + 2]))
      for o in ("needle", "tripod")}


def place(obj, x, y, yaw):
    a, z = ZQ[obj]
    d.qpos[a:a + 3] = [BASE[0] + x, BASE[1] + y, z]
    h = np.radians(yaw) / 2
    d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]


def render(cam, seg=False):
    im = sim.render(camera_name=cam, width=S, height=S, segmentation=seg)[::-1]
    return CLS[np.where(im[..., 1] >= 0, im[..., 1], m.ngeom)] if seg else im


def redmask(rgb):
    """R - G > 12 and R >= B. Not fit_layout_wrist.py's R - max(G, B) > 22: in the wrist view the handle
    cube sits in the gripper's shadow at RGB ~(20, 0, 12), which that gate drops (it made the pre-grasp
    needle IoU read 0.03). The paint here has G ~ 0; the white board and the black gripper have G ~ R."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return ((r - g) > 12) & (r >= b)


def darkmask(rgb):
    return (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2] < 75) & ~redmask(rgb)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def cdiff(a, b):
    if a.sum() < 20 or b.sum() < 20:
        return None
    return np.round(np.argwhere(a).mean(0) - np.argwhere(b).mean(0), 1).tolist()


def jpeg(buf):
    return np.array(Image.open(io.BytesIO(np.asarray(buf).tobytes())).convert("RGB"))


lay = {k: v for k, v in json.load(open(args.layout)).items() if not k.startswith("_")}
f = h5py.File(args.dataset, "r")
tiles = []
for name in sorted(f["data"], key=lambda k: int(k.split("_")[1])):
    dd = f["data"][name]
    src = dd.attrs["source_episode"]
    q = np.asarray(dd["obs/real_joint_pos"])
    g = np.asarray(dd["obs/real_gripper"])
    a = np.asarray(dd["actions"])[:, 6]
    close = int(np.argmax(a > 0))
    for tag, t in (("init", 0), ("pre-grasp", max(close - 1, 0))):
        d.qpos[jidx] = q[t]
        for adr, v in zip(gq, tv.robotiq_real_to_sim(g[t])):
            d.qpos[adr] = v
        for obj, p in lay[src].items():
            place(obj, *p)
        sim.forward()
        out, row = [], []
        for cam in (SCENE, WRIST):
            s = render(cam, True)
            hmask = s == 5                    # handle on its own (size check), then fold it into "needle"
            s = np.where(s == 5, 2, s)
            real = jpeg(dd[f"obs/{cam}_image"][t])
            red = redmask(real)
            res = dict(needle=iou(s == 2, red & ~(s == 3) & cv2.dilate((s == 2).astype(np.uint8), np.ones((31, 31))).astype(bool)),
                       tripod=iou(s == 3, red & ~(s == 2) & cv2.dilate((s == 3).astype(np.uint8), np.ones((31, 31))).astype(bool)),
                       needle_dpx=cdiff(s == 2, red & cv2.dilate((s == 2).astype(np.uint8), np.ones((41, 41))).astype(bool) & ~(s == 3)))
            if cam == WRIST and hmask.sum() > 200:
                # handle size: widths of sim handle vs real red inside the handle's dilated box, rows/cols
                box = cv2.dilate(hmask.astype(np.uint8), np.ones((25, 25))).astype(bool) & ~(s == 4)
                rr = red & box
                def ext(mk):
                    ys, xs = np.nonzero(mk)
                    return np.percentile(xs, 97) - np.percentile(xs, 3), np.percentile(ys, 97) - np.percentile(ys, 3)
                if rr.sum() > 200:
                    res["handle_w_sim_real"] = [round(float(v), 0) for v in (*ext(hmask), *ext(rr))]
            if cam == WRIST:
                near = cv2.dilate((s == 4).astype(np.uint8), np.ones((15, 15))).astype(bool)
                res["gripper"] = iou(s == 4, darkmask(real) & near)
            row.append(f"{cam[:5]} " + " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in res.items()))
            rgb = render(cam)
            ov = (0.5 * real + 0.5 * rgb).astype(np.uint8)
            ov[(s == 2) & ~red] = [0, 255, 255]       # sim needle where real has none: cyan
            ov[red & ~(s >= 2) & (s != 4)] = [255, 255, 0] if cam == WRIST else ov[red & ~(s >= 2) & (s != 4)]
            out.append(ov)
        print(f"{name} ({src}) {tag:9s} t={t:3d} knuckle {g[t]:.3f} | " + " | ".join(row), flush=True)
        tiles.append(np.concatenate(out, 1))
cv2.imwrite(args.out, cv2.cvtColor(np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)], 0),
                                   cv2.COLOR_RGB2BGR))
print("wrote", args.out)
