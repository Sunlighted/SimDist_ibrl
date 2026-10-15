# usage: TABLE_RGB=r,g,b OBJECT_RGB=r,g,b NSEED=16 python colour_check.py out.png   (defaults = the cell yaml)
"""Render the 10-14 Robotiq cell (nominal, no DR) at the real DROID-home q and compare colours
with the exposure-locked real frames (data/home_droid) in the 224 policy crops."""
import sys, os, dataclasses, json
import numpy as np, cv2
REPO = "/home/Chenyang/ibrl"; sys.path.insert(0, REPO)
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
import env.threading_variants as tv
from real2sim.deploy.preprocess import Rectifier
sys.path.insert(0, REPO + "/real2sim/deploy")
from live_crop_view import load_cams

OUT = sys.argv[1]
D = REPO + "/real2sim/zed_capture/calib_20261014/data/home_droid"
cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
GEOM_DR = ("dr_cam_pos", "dr_cam_rot_deg", "dr_cam_fovy_deg", "dr_base_height", "dr_base_yaw_deg",
           "dr_base_tilt_deg", "dr_base_xy")
over = dict(table_rgb=tuple(float(v) for v in os.environ["TABLE_RGB"].split(",")) if os.environ.get("TABLE_RGB") else cell.table_rgb,
            object_rgb=tuple(float(v) for v in os.environ["OBJECT_RGB"].split(",")) if os.environ.get("OBJECT_RGB") else cell.object_rgb)
cams_nodr = tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras)
nom = dataclasses.replace(cell, cameras=cams_nodr, **{k: 0.0 for k in GEOM_DR}, **over)
print("table_rgb", nom.table_rgb, "object_rgb", nom.object_rgb)
NSEED = int(os.environ.get("NSEED", 16))
q = json.load(open(D + "/state.json"))["q"]
S = 224
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=S, rl_image_size=S,
                     camera_names=["agentview", "robot0_eye_in_hand"], rl_cameras=["agentview", "robot0_eye_in_hand"],
                     use_state=True, obs_stack=1, state_stack=1, prop_stack=1, flip_image=True, device="cpu",
                     hand_init_perturb_joint_sigma=0.0, sim2real=nom)
acc = {}
for seed in range(NSEED):
  np.random.seed(seed); env.reset()
  e = env.env; m, d = e.sim.model, e.sim.data
  for i, n in enumerate([f"robot0_joint{k}" for k in range(1, 8)]):
      d.qpos[m.get_joint_qpos_addr(n)] = q[i]
  e.sim.forward()
  for cam in ["agentview", "robot0_eye_in_hand"]:
    rgb = e.sim.render(camera_name=cam, width=S, height=S)[::-1].astype(float)
    seg = e.sim.render(camera_name=cam, width=S, height=S, segmentation=True)[::-1][..., 1]
    acc.setdefault(cam, []).append((rgb, seg))
# real needle as fitted by the scene camera: (0.553, -0.108) base, z_rot ~ -104
import env.threading_variants as _t
bx, by = cell.base_xy if cell.base_xy is not None else (-0.489, -0.034)
base = d.body_xpos[m.body_name2id("robot0_base")].copy()
e.sim.forward()
fr = {}
for cam, L in acc.items():
    fr[cam] = (L[0][0].astype(np.uint8), L[0][1], L)
cams = load_cams(tv.REAL_CELL_ROBOTIQ_YAML)
real = {}
for ros, c in cams.items():
    raw = cv2.imread(D + f"/{ros}_left.png")
    rect = Rectifier(c["K"], c["D"], raw.shape[1], raw.shape[0], S, c["fovy"])
    real[c["sim"]] = rect(raw)[..., ::-1]   # RGB
gid = {n: m.geom_name2id(n) for n in m.geom_names}
table_ids = {i for n, i in gid.items() if "table" in n and "leg" not in n}
red_ids = {i for n, i in gid.items() if n.startswith(("needle", "tripod"))}
print("geoms table:", sorted(n for n, i in gid.items() if i in table_ids))
tiles = []
for cam, (rgb, seg, L) in fr.items():
    R = real[cam].astype(float)
    for lab, ids in (("table/board", table_ids), ("needle+tripod", red_ids)):
        mk = np.isin(seg, list(ids))
        mk = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        # robust to residual misalignment: keep real pixels that look like the class
        rr = R[mk]
        if lab.startswith("table"):
            keep = (rr.min(1) > 120)
        else:
            keep = (rr[:, 0] > rr[:, 1] + 40)
        simv = np.nanmean([np.median(r[cv2.erode(np.isin(sg, list(ids)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)], 0) for r, sg in L], 0)
        print(f"{cam:20s} {lab:14s} n={mk.sum():5d}  sim RGB {np.round(simv,1)}  "
              f"real RGB {np.round(np.median(rr[keep],0),1) if keep.any() else None} (kept {keep.mean():.2f})")
    mk = cv2.erode(np.isin(seg, list(table_ids)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lum = lambda x: x @ np.array([0.299, 0.587, 0.114])
    rl = lum(R[mk]); rl = rl[rl > 120]
    sp_sim = [np.subtract(*np.percentile(lum(r[mk & np.isin(sg, list(table_ids))]), [90, 10])) for r, sg in L]
    print(f"{cam:20s} board spread p90-p10: sim {np.mean(sp_sim):.1f} (seeds {np.min(sp_sim):.0f}-{np.max(sp_sim):.0f})  real {np.subtract(*np.percentile(rl, [90, 10])):.1f}")
    tiles.append(np.hstack([rgb, real[cam].astype(np.uint8), (0.5 * rgb + 0.5 * real[cam]).astype(np.uint8)]))
cv2.imwrite(OUT, np.vstack(tiles)[..., ::-1])
print("wrote", OUT)
