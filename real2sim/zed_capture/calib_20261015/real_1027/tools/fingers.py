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
demo, out, tripod_yaw = sys.argv[1], sys.argv[2], float(sys.argv[3])
inh = json.load(open(sys.argv[4])); P_in = np.array(inh["pos"]); R_in = np.array(inh["rot"]).reshape(3, 3)
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
Q, DQ, K = h["observation/joint_positions"][:], h["observation/joint_velocities"][:], h["observation/gripper_knuckle"][:]
SC, WR = h["observation/images/scene"], h["observation/images/wrist"]
kc = int(np.argmax(K > 0.3))
def act(ims, prop):
    obs = {srv.cammap[c]: torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) for c, im in ims.items()}
    obs["prop"] = torch.from_numpy(prop)
    with torch.no_grad():
        return np.clip(srv.agent.act(obs, eval_mode=True).numpy().astype(np.float64), -1, 1)

res = []; panels = []
for t in list(range(0, kc - 5, 12)):
    fingers = np.array([K[t]]); arm.set_state(Q[t], DQ[t], fingers); sim.forward()
    seg = sim.render(camera_name="robot0_eye_in_hand", width=224, height=224, segmentation=True)[::-1]
    gid = seg[..., 1] if seg.ndim == 3 else seg
    names = np.array([m.body_id2name(b) or "" for b in range(m.nbody)])[m.geom_bodyid[np.clip(gid, 0, m.ngeom - 1)]]
    ms = np.vectorize(lambda n: "gripper" in n)(names) & (gid >= 0)
    rw = srv.obs_images([bytes(SC[t]), bytes(WR[t])], ["scene", "wrist"])["wrist"]
    hsv = cv2.cvtColor(rw, cv2.COLOR_RGB2HSV); mr = (hsv[..., 2] < 70)
    mr[:112] &= False                                 # fingers live in the lower half
    ms2 = ms.copy(); ms2[:112] = False
    iou = (mr & ms2).sum() / max((mr | ms2).sum(), 1)
    cy = lambda M: np.array(np.nonzero(M)).mean(1) if M.any() else np.array([np.nan, np.nan])
    top = lambda M: np.nonzero(M.any(1))[0].min() if M.any() else -1
    res.append((t, iou, *(cy(mr) - cy(ms2)), top(mr), top(ms2), float(K[t])))
    if len(panels) < 4:
        ov = rw.copy(); ov[ms2] = (0.5 * ov[ms2] + [0, 127, 0]).astype(np.uint8)
        e = cv2.Canny(mr.astype(np.uint8) * 255, 50, 100) > 0; ov[e] = [255, 0, 255]
        panels.append(ov)
for r in res: print("t %3d IoU %.2f  centroid real-sim (row %+.1f, col %+.1f) px  top row real %d sim %d  knuckle %.2f" % r)
cv2.imwrite(out + "_fingers.jpg", cv2.resize(np.hstack(panels), None, fx=2, fy=2)[..., ::-1])
