"""Tripod (x, y, yaw) in the base frame from the SCENE camera at chosen frames of the raw 10-27 demos.
Sim: arm at the measured q, Robotiq at the mapped knuckle, needle parked far away (unoccluded tripod);
real red mask restricted near the sim tripod and outside the dilated sim gripper. Same code for frame 0 and
the last frame, so the difference is the tripod's motion during the episode."""
import dataclasses, glob, io, json, os, sys
sys.path.insert(0, "/home/Chenyang/ibrl")
import cv2, h5py, numpy as np
from PIL import Image
from scipy.optimize import minimize
import env.threading_variants as tv
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import Sim2RealConfig, load_sim2real
from real2sim.deploy.preprocess import Rectifier, camera_intrinsics
RAW, LAY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = 448; SCENE = "agentview"
cell = load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML)
off = {f.name: f.default for f in dataclasses.fields(Sim2RealConfig) if f.name.startswith("dr_")}
cell = dataclasses.replace(cell, cameras=tuple(dataclasses.replace(c, randomize=False) for c in cell.cameras), **off)
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, image_size=64, rl_image_size=64, camera_names=[SCENE],
                     rl_cameras=[SCENE], use_state=True, device="cpu", hand_init_perturb_joint_sigma=0.0, sim2real=cell,
                     env_kwargs={"exec_layer": False})
np.random.seed(0); env.reset(); e = env.env; m, d = e.sim.model, e.sim.data
jidx = np.array(e.robots[0]._ref_joint_pos_indexes); gq = [m.get_joint_qpos_addr(n) for n in tv.ROBOTIQ_DRIVER_JOINTS]
BASE = d.body_xpos[m.body_name2id("robot0_base")].copy()
names = [m.geom_id2name(i) or "" for i in range(m.ngeom)]; bodies = [m.body_id2name(int(m.geom_bodyid[i])) or "" for i in range(m.ngeom)]
CLS = np.zeros(m.ngeom + 1, np.int8)
for i, (n, b) in enumerate(zip(names, bodies)):
    CLS[i] = 2 if n.startswith("needle") else 3 if n.startswith("tripod") else 4 if b.startswith("gripper0") else 0
ZQ = {o: (m.get_joint_qpos_addr(f"{o}_obj_joint0")[0], float(d.qpos[m.get_joint_qpos_addr(f"{o}_obj_joint0")[0] + 2])) for o in ("needle", "tripod")}
CC = {c.name: c for c in cell.cameras}[SCENE]
def place(obj, x, y, yaw):
    a, z = ZQ[obj]; d.qpos[a:a + 3] = [BASE[0] + x, BASE[1] + y, z]; h = np.radians(yaw) / 2; d.qpos[a + 3:a + 7] = [np.cos(h), 0, 0, np.sin(h)]
def seg():
    im = e.sim.render(camera_name=SCENE, width=S, height=S, segmentation=True)[::-1][..., 1]
    return CLS[np.where(im >= 0, im, m.ngeom)]
def red(rgb):
    r, g, b = (rgb[..., i].astype(int) for i in range(3)); return ((r - g) > 12) & (r >= b)
dil = lambda mk, k: cv2.dilate(mk.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
lay = {k: v for k, v in json.load(open(LAY)).items() if not k.startswith("_")}
res = {}
for fn in sorted(glob.glob(os.path.join(RAW, "episode_*.h5"))):
    ep = os.path.basename(fn)[:-3]; h = h5py.File(fn, "r"); T = len(h["observation/joint_positions"])
    intr = json.loads(h.attrs["camera_intrinsics"])["scene"]; k = intr["K"]
    K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], float); D = np.array(intr["D"], float)
    K, D = camera_intrinsics(CC, K, D, 1280, 720); rect = Rectifier(K, D, 1280, 720, S, CC.resolved_fovy())
    res[ep] = {}
    for tag, t in (("first", 0), ("last", T - 1)):
        img = rect(np.array(Image.open(io.BytesIO(np.asarray(h["observation/images/scene"][t]).tobytes())).convert("RGB")))
        rr = red(img)
        d.qpos[jidx] = h["observation/joint_positions"][t]
        for a_, v in zip(gq, tv.robotiq_real_to_sim(h["observation/gripper_knuckle"][t])): d.qpos[a_] = v
        place("needle", 2.0, 2.0, 0.0)
        p0 = np.array(lay[ep]["tripod"], float); place("tripod", *p0); e.sim.forward()
        s0 = seg(); roi = dil(s0 == 3, 61); gr = dil(s0 == 4, 9)
        def loss(p):
            place("tripod", *p); e.sim.forward(); s = seg(); ms = (s == 3) & ~gr
            mr = rr & roi & ~gr
            return 1 - (ms & mr).sum() / max((ms | mr).sum(), 1)
        best = None
        for dyaw in (0, -5, 5):
            x0 = p0 + [0, 0, dyaw]
            r = minimize(loss, x0, method="Nelder-Mead", options=dict(xatol=2e-4, fatol=1e-4, maxiter=300,
                         initial_simplex=[x0, x0 + [0.006, 0, 0], x0 + [0, 0.006, 0], x0 + [0, 0, 4]]))
            if best is None or r.fun < best.fun: best = r
        res[ep][tag] = dict(pose=[float(v) for v in best.x], iou=float(1 - best.fun))
    a, b = np.array(res[ep]["first"]["pose"]), np.array(res[ep]["last"]["pose"])
    res[ep]["moved_mm_deg"] = [round(1e3 * (b[0] - a[0]), 1), round(1e3 * (b[1] - a[1]), 1), round(b[2] - a[2], 1)]
    print(ep, "first", np.round(a, 4).tolist(), f"IoU {res[ep]['first']['iou']:.2f}", "| last", np.round(b, 4).tolist(),
          f"IoU {res[ep]['last']['iou']:.2f}", "| moved (dx, dy mm, dyaw deg)", res[ep]["moved_mm_deg"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
