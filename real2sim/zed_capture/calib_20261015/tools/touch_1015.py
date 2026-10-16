"""10-15 touch-offs: fingertip contact (FK of the measured q, closed Robotiq, sim cell model) vs
(a) the ChArUco corners seen by the SCENE camera at the cell's pose, (b) the sim work surface,
(c) the needle ends from the scene-camera render fit."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.expanduser("~/simdist/SimDist_ibrl"))
sys.path.insert(0, os.path.expanduser("~/simdist/SimDist_ibrl/real2sim/zed_capture/calib_20261010/tools"))
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite, mujoco
from scipy.spatial.transform import Rotation
import solve_calib as sc
import env.threading_variants  # noqa: F401 registers the env
run = sys.argv[1]
raw = robosuite.make("Threading_RealCellRobotiq", robots="Panda", has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False)
np.random.seed(0); raw.reset()
M, D = raw.sim.model._model, raw.sim.data._data
rob = raw.robots[0]
a = np.zeros(raw.action_dim)
for _ in range(60):
    a[-1] = 1.0; raw.step(a)
gidx = rob._ref_gripper_joint_pos_indexes; gq = D.qpos[gidx].copy()
aidx = rob._ref_joint_pos_indexes
base = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
gb = {b for b in range(M.nbody) if mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b).startswith("gripper0")}
geoms = [g for g in range(M.ngeom) if M.geom_bodyid[g] in gb and M.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
tid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
mujoco.mj_kinematics(M, D)
SURF = D.geom_xpos[tid][2] + M.geom_size[tid][2] - D.xpos[base][2]


def contact(q):
    """lowest gripper point (base frame) and the centroid of everything within 2 mm of it"""
    D.qpos[aidx] = q; D.qpos[gidx] = gq; mujoco.mj_kinematics(M, D)
    Rb, pb = D.xmat[base].reshape(3, 3), D.xpos[base]
    P = []
    for g in geoms:
        mid = M.geom_dataid[g]; a0, n = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
        W = (D.geom_xmat[g].reshape(3, 3) @ M.mesh_vert[a0:a0 + n].T).T + D.geom_xpos[g]
        P.append((W - pb) @ Rb)
    P = np.concatenate(P)
    lo = P[:, 2].min()
    patch = P[P[:, 2] < lo + 0.002]
    return lo, patch.mean(0), np.ptp(patch[:, :2], 0)


# board from the scene camera at the cell's (validated, unbumped) pose, pooled over samples
SCENE_POS = np.array([0.23173, 0.55526, 0.52272]); q = [0.15341, -0.29297, 0.85476, -0.40002]
Bs = sc.T(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(), SCENE_POS)
sess = json.load(open(os.path.join(run, "session.json")))
Ks, Ds = np.array(sess["cameras"]["scene"]["K"]), np.array(sess["cameras"]["scene"]["D"])
A = 0.04200                                     # square size from the 10-15 wrist solve
ids, pix, per = [], [], []
for d in sorted(glob.glob(os.path.join(run, "sample_*"))):
    i, p = sc.detect(os.path.join(d, "scene_left.png"))
    if len(i) >= 6:
        Tsb = sc.pnp(sc.UNIT[i] * A, p, Ks, Ds); per.append((os.path.basename(d), Tsb))
    ids.append(i); pix.append(p)
ids, pix = np.concatenate(ids), np.concatenate(pix)
Tsb = sc.pnp(sc.UNIT[ids] * A, pix, Ks, Ds)
for n, T in per:
    print(f"  scene PnP {n}: board moved {np.linalg.norm(T[:3, 3] - Tsb[:3, 3]) * 1e3:.1f} mm / {sc.rot_deg(T, Tsb):.2f} deg vs pooled")
e = np.linalg.norm(sc.project(Tsb, sc.UNIT[ids] * A, Ks, Ds) - pix, axis=1)
print(f"pooled scene PnP: {len(ids)} corners, reprojection median {np.median(e):.2f} px")
Y = Bs @ Tsb                                     # board in base
EXT = {k: (Y @ np.r_[sc.UNIT[k] * A, 1])[:3] for k in (0, 3, 20, 23)}
ALL = np.array([(Y @ np.r_[u * A, 1])[:3] for u in sc.UNIT])
print(f"board plane in base: corners z {ALL[:, 2].min():.4f}..{ALL[:, 2].max():.4f}; sim work surface {SURF:.4f}")
print(f"\nsim work surface (base z) {SURF:.4f}")
out = {}
for d in sorted(glob.glob(os.path.join(run, "sample_*"))):
    st = json.load(open(os.path.join(d, "state.json")))
    lo, c, span = contact(np.array(st["q"]))
    name = st.get("point")
    line = f"{name:13s} tip lowest z {lo:.4f}  patch centre ({c[0]:.4f}, {c[1]:.4f})  patch span {np.round(span * 1e3, 1).tolist()} mm"
    if name and name.startswith("c"):
        dists = np.linalg.norm(ALL[:, :2] - c[:2], axis=1); k = int(np.argmin(dists))
        line += f" | nearest corner id {k} ({ALL[k, 0]:.4f}, {ALL[k, 1]:.4f}, z {ALL[k, 2]:.4f}): dxy {np.round((c[:2] - ALL[k, :2]) * 1e3, 1).tolist()} mm, dz {(lo - ALL[k, 2]) * 1e3:+.1f} mm"
        out[name] = dict(tip=c.tolist(), lo=lo, corner_id=k, corner=ALL[k].tolist())
    else:
        out[name] = dict(tip=c.tolist(), lo=lo)
    print(line)
json.dump(dict(points=out, T_base_board=Y.tolist(), surface=SURF), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "touch_1015.json"), "w"), indent=1)

# ---- projection check: FK fingertip patch -> scene image (cell scene pose)
import cv2
Tcb = sc.inv(Bs)                                  # base -> scene camera
tiles = []
for d in sorted(glob.glob(os.path.join(run, "sample_*"))):
    st = json.load(open(os.path.join(d, "state.json")))
    D.qpos[aidx] = st["q"]; D.qpos[gidx] = gq; mujoco.mj_kinematics(M, D)
    Rb, pb = D.xmat[base].reshape(3, 3), D.xpos[base]
    P = []
    for g in geoms:
        mid = M.geom_dataid[g]; a0, n = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
        P.append(((D.geom_xmat[g].reshape(3, 3) @ M.mesh_vert[a0:a0 + n].T).T + D.geom_xpos[g] - pb) @ Rb)
    P = np.concatenate(P)
    low = P[P[:, 2] < P[:, 2].min() + 0.004]
    uv = sc.project(Tcb, np.ascontiguousarray(P[::7], np.float64), Ks, Ds); uvl = sc.project(Tcb, np.ascontiguousarray(low, np.float64), Ks, Ds)
    im = cv2.imread(os.path.join(d, "scene_left.png"))
    c = uvl.mean(0).astype(int)
    for p in uv.astype(int):
        cv2.circle(im, tuple(p), 1, (255, 255, 0), -1)
    for p in uvl.astype(int):
        cv2.circle(im, tuple(p), 1, (0, 0, 255), -1)
    x0, y0 = max(0, c[0] - 90), max(0, c[1] - 70)
    t = cv2.resize(im[y0:y0 + 140, x0:x0 + 180], (360, 280), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, st["point"], (4, 20), 0, 0.6, (0, 255, 255), 2); tiles.append(t)
tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "touch_proj.jpg"),
            np.vstack([np.hstack(tiles[i:i + 4]) for i in (0, 4)]))
