"""Open-loop: feed real demo frames through the deploy server's preprocessing + the stage-1 policy, frame by frame.
exec_obs: reference arm == measured arm (q_ref = q, v_ref = dq); last action = 'zero' or 'prev' (policy's own previous)."""
import os, sys, json, glob, h5py, numpy as np, torch
D = os.path.expanduser("~/simdist/deploy_simdist_1015"); os.chdir(D); sys.path.insert(0, D)
import real2sim.deploy.policy_server as ps
demo_glob, out, last_mode = sys.argv[1], sys.argv[2], sys.argv[3]
a = ps.build_parser().parse_args(["--ckpt", "ckpts/robotiq_s1/model0.pt", "--z_min", "0.96",
                                  "--color_json", "real2sim/zed_capture/calib_20261014/color_correction_threading_1014.json"])
srv = ps.Server(a)
res = {}
for f in sorted(glob.glob(demo_glob)):
    h = h5py.File(f)
    intr = json.loads(h.attrs["camera_intrinsics"])
    srv.hello({"hz": srv.arm.control_freq, "camera_info": {c: {"K": np.array(v["K"]).ravel().tolist(), "D": list(v["D"]),
               "width": int(v["width"]), "height": int(v["height"])} for c, v in intr.items()}})
    Q, DQ, K = h["observation/joint_positions"][:], h["observation/joint_velocities"][:], h["observation/gripper_knuckle"][:]
    SC, WR = h["observation/images/scene"], h["observation/images/wrist"]
    last = np.zeros(7); rows = []
    for t in range(len(Q)):
        ims = srv.obs_images([bytes(SC[t]), bytes(WR[t])], ["scene", "wrist"])
        fingers = np.array([K[t]])
        srv.arm.set_state(Q[t], DQ[t], fingers)
        prop = srv.arm.prop(fingers); eef = srv.arm.eef().copy()
        srv.arm.q_ref, srv.arm.v_ref = np.array(Q[t], np.float64), np.array(DQ[t], np.float64)
        prop = np.concatenate([prop, srv.arm.exec_obs(Q[t], last if last_mode == "prev" else np.zeros(7))]).astype(np.float32)
        obs = {srv.cammap[c]: torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) for c, im in ims.items()}
        obs["prop"] = torch.from_numpy(prop)
        with torch.no_grad():
            act = np.clip(srv.agent.act(obs, eval_mode=True).numpy().astype(np.float64), -1, 1)
        last = act
        rows.append(dict(t=t, eef=eef.tolist(), a=act.round(4).tolist(), k=float(K[t])))
    res[os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)] = rows
    print(f, len(rows), flush=True)
json.dump(res, open(out, "w"))
