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

cid = m.camera_name2id("robot0_eye_in_hand"); cp0 = m.cam_pos[cid].copy(); ls0 = m.light_castshadow.copy()
def wr():
    return sim.render(camera_name="robot0_eye_in_hand", width=224, height=224)[::-1].copy()
def cmatch(src, ref):
    s, r = src.astype(np.float32), ref.astype(np.float32)
    o = (s - s.mean((0, 1))) / (s.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
    return np.clip(o, 0, 255).astype(np.uint8)
rows = []
for t in range(kc + 8, len(Q), 3):
    fingers = np.array([K[t]])
    arm.set_state(Q[t], DQ[t], fingers)
    prop = arm.prop(fingers)
    arm.q_ref, arm.v_ref = np.array(Q[t], np.float64), np.array(DQ[t], np.float64)
    prop = np.concatenate([prop, arm.exec_obs(Q[t], np.zeros(7))]).astype(np.float32)
    arm.set_state(Q[t], DQ[t], fingers)
    Rg = sim.data.site_xmat[gs].reshape(3, 3); pg = sim.data.site_xpos[gs].copy()
    sim.data.qpos[adr["needle"]:adr["needle"] + 3] = pg + Rg @ P_in
    sim.data.qpos[adr["needle"] + 3:adr["needle"] + 7] = T.convert_quat(T.mat2quat(Rg @ R_in), to="wxyz")
    ta = adr["tripod"]
    sim.data.qpos[ta:ta + 3] = [base[0] + 0.521, base[1] + 0.183, NOM["q"][ta + 2]]
    sim.data.qpos[ta + 3:ta + 7] = T.convert_quat(T.axisangle2quat(np.array([0, 0, np.radians(tripod_yaw)])), to="wxyz")
    sim.forward()
    real = srv.obs_images([bytes(SC[t]), bytes(WR[t])], ["scene", "wrist"])
    sc = real["scene"]
    W = {"sim": wr()}
    W["blur1"] = cv2.GaussianBlur(W["sim"], (0, 0), 0.9); W["blur1_col"] = cmatch(W["blur1"], real["wrist"])
    W["colmatch"] = cmatch(W["sim"], real["wrist"])
    W["real"] = real["wrist"]
    rows.append({"t": t, **{k: act({"scene": sc, "wrist": v}, prop).round(3).tolist() for k, v in W.items()}})
    pass
json.dump(rows, open(out + ".json", "w")); print("frames", len(rows))
