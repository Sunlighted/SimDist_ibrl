import os, sys, numpy as np, torch, dataclasses
D = os.path.expanduser("~/simdist/deploy_simdist_1015"); os.chdir(D); sys.path.insert(0, D)
sys.path.insert(0, os.path.join(D, "real2sim/deploy"))
import policy_server as ps
from env.robosuite_wrapper import PixelRobosuite
from env.sim2real import load_sim2real
_, cfg = ps.load_agent("ckpts/robotiq_s1/model0.pt", "cpu")
env = PixelRobosuite(env_name=cfg.task_name, robots="Panda", episode_length=400, camera_names=[], rl_cameras=[],
                     use_state=True, state_only_no_render=True, device="cpu", exec_obs=True, hand_init_perturb_joint_sigma=0.1,
                     sim2real=dataclasses.replace(load_sim2real(cfg.sim2real_cell), dr_background_image=""))
for ep in range(6):
    np.random.seed(1000 + ep); env.reset()
    sim = env.env.sim; m = sim.model
    ng, hg = m.geom_name2id("needle_obj_needle"), m.geom_name2id("needle_obj_handle")
    tid = m.geom_name2id("table_collision"); top = sim.data.geom_xpos[tid][2] + m.geom_size[tid][2]
    def meas(tag):
        R = sim.data.geom_xmat[ng].reshape(3, 3)
        ax = int(np.argmax(m.geom_size[ng]));  u = R[:, ax]
        nc, hc = sim.data.geom_xpos[ng], sim.data.geom_xpos[hg]
        if u @ (nc - hc) < 0: u = -u
        tipc = nc + u * m.geom_size[ng][ax]
        # lowest point of the shaft box at the tip end
        corners = [tipc + R @ (s * np.eye(3)[i] * m.geom_size[ng][i]) for i in range(3) if i != ax for s in (-1, 1)]
        tipz = min(c[2] for c in corners) - top
        Rh = sim.data.geom_xmat[hg].reshape(3, 3)
        hz = min((hc + Rh @ (np.array([sx, sy, sz]) * m.geom_size[hg]))[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)) - top
        print(f"ep{ep} {tag}: shaft pitch {np.degrees(np.arcsin(-u[2])):+.2f} deg (tip down +) | tip bottom {tipz*1000:+.1f} mm | "
              f"handle bottom {hz*1000:+.1f} mm | shaft size {np.round(m.geom_size[ng],4)} handle {np.round(m.geom_size[hg],4)}")
    meas("reset")
    for _ in range(20):
        env.step(torch.zeros(7))
    meas("after 2 s")
