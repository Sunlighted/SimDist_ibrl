import os, sys, json, numpy as np, torch
D = os.path.expanduser("~/simdist/deploy_simdist_1015"); os.chdir(D); sys.path.insert(0, D)
sys.path.insert(0, os.path.join(D, "real2sim/deploy"))
import policy_server as ps
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
from env.strict_success import StrictSuccessConfig
agent, cfg = ps.load_agent("ckpts/robotiq_s1/model0.pt", "cuda")
cams = cfg.rl_camera.split("+")
env = PixelRobosuite(env_name=cfg.task_name, robots="Panda", episode_length=400, image_size=224, rl_image_size=224,
                     camera_names=cams, rl_cameras=cams, use_state=False, extra_state=True, device="cpu",
                     strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04),
                     sim2real=__import__("dataclasses").replace(load_sim2real(cfg.sim2real_cell), dr_background_image=""), exec_obs=True, hand_init_perturb_joint_sigma=0.1)
N, out = int(sys.argv[2]), sys.argv[1]
res = []
for ep in range(N):
    np.random.seed(1000 + ep); torch.manual_seed(1000 + ep)
    obs, _ = env.reset()
    if ep == 0: print({k: (tuple(v.shape), v.dtype) for k, v in obs.items()})
    raw = env.env; sim = raw.sim; m = sim.model
    base = sim.data.body_xpos[m.body_name2id(raw.robots[0].robot_model.root_body)].copy()
    gs = m.site_name2id("gripper0_grip_site")
    rg = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith("tripod_obj_ring_") and not m.geom_id2name(i).endswith("_vis")]
    ng, hg = m.geom_name2id("needle_obj_needle"), m.geom_name2id("needle_obj_handle")
    traj = []
    for k in range(400):
        o = {c: obs[c] for c in cams}; o["prop"] = obs["prop"]
        with torch.no_grad():
            a = agent.act(o, eval_mode=True).cpu().numpy().astype(np.float64)
        a = np.clip(a, -1, 1)
        obs, r, term, succ, _ = env.step(torch.from_numpy(a).float())
        nc, hc = sim.data.geom_xpos[ng] - base, sim.data.geom_xpos[hg] - base
        tip = nc + (nc - hc) / np.linalg.norm(nc - hc) * 0.06
        traj.append(dict(k=k, grip=(sim.data.site_xpos[gs] - base).tolist(), ring=(sim.data.geom_xpos[rg].mean(0) - base).tolist(),
                         tip=tip.tolist(), handle=hc.tolist(), a=np.round(a, 3).tolist(), succ=bool(succ)))
        if term: break
    res.append(dict(ep=ep, success=bool(traj[-1]["succ"]), traj=traj))
    print(ep, "success", traj[-1]["succ"], "steps", len(traj), flush=True)
    json.dump(res, open(out, "w"))
