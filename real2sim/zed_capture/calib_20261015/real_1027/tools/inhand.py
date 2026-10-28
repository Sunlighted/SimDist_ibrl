"""In-hand needle transform in sim (grip-site frame), from policy rollouts after the lift."""
import os, sys, json, dataclasses, numpy as np, torch
D = os.path.expanduser("~/simdist/deploy_simdist_1015"); os.chdir(D); sys.path.insert(0, D)
import real2sim.deploy.policy_server as ps
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
from env.strict_success import StrictSuccessConfig
agent, cfg = ps.load_agent("ckpts/robotiq_s1/model0.pt", "cuda")
cams = cfg.rl_camera.split("+")
env = PixelRobosuite(env_name=cfg.task_name, robots="Panda", episode_length=400, image_size=224, rl_image_size=224,
                     camera_names=cams, rl_cameras=cams, use_state=False, extra_state=True, device="cpu",
                     strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04),
                     sim2real=dataclasses.replace(load_sim2real(cfg.sim2real_cell), dr_background_image=""), exec_obs=True,
                     hand_init_perturb_joint_sigma=0.1)
rels = []
for ep in range(4):
    np.random.seed(2000 + ep); torch.manual_seed(2000 + ep); obs, _ = env.reset()
    sim = env.env.sim; m = sim.model; gs = m.site_name2id("gripper0_grip_site"); nb = m.body_name2id("needle_obj_root")
    for k in range(120):
        o = {c: obs[c] for c in cams}; o["prop"] = obs["prop"]
        with torch.no_grad(): a = np.clip(agent.act(o, eval_mode=True).cpu().numpy(), -1, 1)
        obs, r, term, succ, _ = env.step(torch.from_numpy(a).float())
        Rg = sim.data.site_xmat[gs].reshape(3, 3); pg = sim.data.site_xpos[gs]
        Rn = sim.data.body_xmat[nb].reshape(3, 3); pn = sim.data.body_xpos[nb]
        if a[6] > 0 and pn[2] - 0.9405 > 0.12:     # lifted well off the table
            rels.append(np.r_[Rg.T @ (pn - pg), (Rg.T @ Rn).ravel()])
        if term or len(rels) >= 10 * (ep + 1): break
    print(ep, len(rels), flush=True)
R = np.array(rels); print("in-hand pos (grip frame) mean", R[:, :3].mean(0).round(4), "std", R[:, :3].std(0).round(4))
json.dump(dict(pos=R[:, :3].mean(0).tolist(), rot=R[:, 3:].mean(0).tolist(), pos_std=R[:, :3].std(0).tolist()), open(sys.argv[1], "w"))
