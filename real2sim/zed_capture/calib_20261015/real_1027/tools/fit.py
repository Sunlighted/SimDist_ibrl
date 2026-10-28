"""Same state, real vs sim images: real demo frame t -> (a) real images, (b) sim render of the same q/knuckle with the
needle at the SIM in-hand pose and the tripod at its nominal pose, (c) real scene + sim wrist, (d) sim scene + real wrist."""
import os, sys, json, glob, h5py, numpy as np, torch, cv2
D = os.path.expanduser("~/simdist/deploy_simdist_1015"); os.chdir(D); sys.path.insert(0, D)
import env.robosuite_wrapper as rw
_PR = rw.PixelRobosuite
def _PRcam(*args, **kw):                      # the reference arm's env, but with both cameras rendering
    kw.update(camera_names=["agentview", "robot0_eye_in_hand"], rl_cameras=["agentview", "robot0_eye_in_hand"],
              state_only_no_render=False)
    return _PR(*args, **kw)
rw.PixelRobosuite = _PRcam
import real2sim.deploy.policy_server as ps
import robosuite.utils.transform_utils as T
NOM = {}
_pk = ps.ReferenceArm._park
def _park_rec(self):                          # remember the objects' reset pose (z, upright) before the first park
    if not NOM:
        NOM["q"] = self.sim.data.qpos.copy()
    _pk(self)
ps.ReferenceArm._park = _park_rec
demo, out, tripod_yaw = sys.argv[1], sys.argv[2], -35.0
inh = json.load(open(sys.argv[3])); GRASPS = sys.argv[4].split(","); P_in = np.array(inh["pos"]); R_in = np.array(inh["rot"]).reshape(3, 3)
U, _, Vt = np.linalg.svd(R_in); R_in = U @ Vt
a = ps.build_parser().parse_args(["--ckpt", "ckpts/robotiq_s1/model0.pt", "--z_min", "0.96",
                                  "--color_json", "real2sim/zed_capture/calib_20261014/color_correction_threading_1014.json"])
srv = ps.Server(a); arm = srv.arm; sim = arm.sim; m = sim.model
h = h5py.File(demo); intr = json.loads(h.attrs["camera_intrinsics"])
srv.hello({"hz": arm.control_freq, "camera_info": {c: {"K": np.array(v["K"]).ravel().tolist(), "D": list(v["D"]),
           "width": int(v["width"]), "height": int(v["height"])} for c, v in intr.items()}})
base = sim.data.body_xpos[m.body_name2id(arm.rob.robot_model.root_body)].copy()
jn = [m.joint_id2name(i) for i in range(m.njnt)]
adr = {p: m.get_joint_qpos_addr([n for n in jn if n.startswith(p)][0])[0] for p in ("needle", "tripod")}
gs = m.site_name2id("gripper0_grip_site")

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
nbody = [b for b in range(m.nbody) if "needle" in (m.body_id2name(b) or "")]
def seg(cam):
    sg = sim.render(camera_name=cam, width=224, height=224, segmentation=True)[::-1]
    gid = sg[..., 1] if sg.ndim == 3 else sg
    return np.isin(m.geom_bodyid[np.clip(gid, 0, m.ngeom - 1)], nbody) & (gid >= 0)
def realmask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 0] < 8) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 35)
CAMS = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}
na = adr["needle"]
def place(x, Rg, pg):
    R = Rot.from_rotvec(np.radians(x[3:])).as_matrix() @ R_in
    sim.data.qpos[na:na + 3] = pg + Rg @ (P_in + x[:3] / 1000)
    sim.data.qpos[na + 3:na + 7] = T.convert_quat(T.mat2quat(Rg @ R), to="wxyz")
    sim.forward()
    return R
results = {}
for gdir in GRASPS:
    st = json.load(open(os.path.join(gdir, "state.json")))
    q = np.array(st["q"]); fingers = np.array([st["knuckle"]])
    arm.set_state(q, np.zeros(7), fingers); sim.forward()
    Rg = sim.data.site_xmat[gs].reshape(3, 3).copy(); pg = sim.data.site_xpos[gs].copy()
    real = srv.obs_images([open(os.path.join(gdir, "scene_0.jpg"), "rb").read(), open(os.path.join(gdir, "wrist_0.jpg"), "rb").read()], ["scene", "wrist"])
    # ROI from a generous union of plausible poses so the tripod (also red) is excluded
    roi = {c: np.zeros((224, 224), bool) for c in CAMS}
    for dx in (-30, 0, 30):
        for dy in (-30, 0, 30):
            place(np.array([dx, dy, 0, 0, 0, 0.0]), Rg, pg)
            for c, sc in CAMS.items(): roi[c] |= seg(sc)
    roi = {c: cv2.dilate(v.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0 for c, v in roi.items()}
    RM = {c: realmask(real[c]) & roi[c] for c in CAMS}
    def loss(x):
        place(x, Rg, pg); l = 0
        for c, sc in CAMS.items():
            sm = seg(sc); l += 1 - (sm & RM[c]).sum() / max((sm | RM[c]).sum(), 1)
        return l
    best = None
    for x0 in ([0, 0, 0, 0, 0, 0], [0, 15, 0, 0, 0, 0], [0, -15, 0, 0, 0, 0], [0, 0, 10, 0, 0, 0], [0, 0, 0, 8, 0, 0], [0, 0, 0, -8, 0, 0]):
        r = minimize(loss, np.array(x0, float), method="Nelder-Mead",
                     options=dict(initial_simplex=np.array(x0, float) + np.vstack([np.zeros(6), np.diag([6, 6, 6, 5, 5, 5.0])]), xatol=0.2, fatol=1e-4, maxiter=800))
        if best is None or r.fun < best.fun: best = r
    x = best.x; R = place(x, Rg, pg)
    ious = {}
    pan = []
    for c, sc in CAMS.items():
        sm = seg(sc); ious[c] = float((sm & RM[c]).sum() / max((sm | RM[c]).sum(), 1))
        ov = real[c].copy(); ov[sm] = (0.5 * ov[sm] + [0, 120, 0]).astype(np.uint8)
        e = cv2.Canny(RM[c].astype(np.uint8) * 255, 50, 100) > 0; ov[e] = [255, 0, 255]; pan.append(ov)
    # needle axis in the grip-site frame: the needle shaft geom's long axis
    ng = m.geom_name2id("needle_obj_needle"); hg = m.geom_name2id("needle_obj_handle")
    u = sim.data.geom_xpos[ng] - sim.data.geom_xpos[hg]; u /= np.linalg.norm(u)
    hc_grip = Rg.T @ (sim.data.geom_xpos[hg] - pg)
    results[os.path.basename(gdir)] = dict(dx_mm=x[:3].round(1).tolist(), drot_deg=x[3:].round(1).tolist(), iou=ious,
        shaft_pitch_deg=float(np.degrees(np.arcsin(-u[2]))), shaft_dir_grip=(Rg.T @ u).round(3).tolist(),
        handle_center_in_grip_mm=(hc_grip * 1000).round(1).tolist(), q=q.tolist(), knuckle=st["knuckle"])
    print(os.path.basename(gdir), json.dumps(results[os.path.basename(gdir)]), flush=True)
    cv2.imwrite(out + "_" + os.path.basename(gdir) + ".jpg", cv2.resize(np.hstack(pan), None, fx=2, fy=2)[..., ::-1])
# the sim reference itself, at the last grasp's arm pose
place(np.zeros(6), Rg, pg)
u = sim.data.geom_xpos[ng] - sim.data.geom_xpos[hg]; u /= np.linalg.norm(u)
print("SIM_REF", json.dumps(dict(shaft_pitch_deg=float(np.degrees(np.arcsin(-u[2]))), shaft_dir_grip=(Rg.T @ u).round(3).tolist(),
      handle_center_in_grip_mm=(Rg.T @ (sim.data.geom_xpos[hg] - pg) * 1000).round(1).tolist())))
json.dump(results, open(out + ".json", "w"), indent=1)
