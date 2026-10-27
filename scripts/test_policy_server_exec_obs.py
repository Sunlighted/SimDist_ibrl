"""Parity test: policy_server's exec_obs block (ReferenceArm.exec_obs) vs the training env's (SimRefController).

Runs the frozen state expert in the exec_obs training env (the cell's DR on, as in training). Every step the
"robot" is the sim arm: its measured q / knuckle go to the deploy Server exactly as robot_client would send them,
the Server's reference is stepped with the same action through the deploy sim_ref path (step_ref -> _clamp_box ->
_speed_caps), and the 27 dims the Server would append to prop are compared with obs["state"][-27:].

  python scripts/test_policy_server_exec_obs.py --stage1 exps/sim2real/threading_robotiq_pixel_stage1_seed1/model0.pt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

ap = argparse.ArgumentParser()
ap.add_argument("--expert", default="exps/sim2real/threading_robotiq_execobs_expert/expert.pt")
ap.add_argument("--stage1", required=True, help="an exec_obs pixel ckpt: the Server is built from it")
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--teacher_force", type=int, default=1,
                help="1: copy the env's q_ref / v_ref into the Server's reference before every exec_obs (tests the "
                     "FEATURE). 0: let the Server's reference run free (also tests that the two references evolve "
                     "alike; they drift apart by MuJoCo solver noise, ~1e-3 rad per step, compounding)")
args = ap.parse_args()

import torch  # noqa: E402
import train_rl  # noqa: E402
import env.threading_variants as tv  # noqa: E402
from env.robosuite_wrapper import EXEC_OBS_DIM  # noqa: E402
from real2sim.deploy.policy_server import Server, build_parser  # noqa: E402

agent, env, _ = train_rl.load_model(args.expert, "cuda")
agent.eval()
srv = Server(build_parser().parse_args(["--ckpt", args.stage1, "--no_color"]))
assert srv.exec_obs, "the stage-1 ckpt has no exec_obs"
arm = srv.arm
# t = 0 is reported apart: the env's shadow takes the episode's base-DR geometry only at its first set_goal, so its
# first observation carries the base DR as a fake ref-vs-measured pose offset (the robot's is 0 there, as it should be).
# The steps after it differ only by the env reading site_xpos one substep stale (~0.1 mm at normal speed).
names = (["dp"] * 3 + ["drot"] * 3 + ["dq"] * 7 + ["v_ref"] * 7 + ["last_a"] * 7)
rows, first, n_clamp, n_resync, n_ref_clamp = [], [], 0, 0, 0
for ep in range(args.episodes):
    np.random.seed(args.seed + ep)
    obs, _ = env.reset()
    srv._reset_ref()
    raw = env.env
    rob = raw.robots[0]
    jidx = np.array(rob._ref_joint_pos_indexes)
    gidx = np.array(rob._ref_gripper_joint_pos_indexes)
    terminal, t = False, 0
    while not terminal:
        sim = raw.sim
        q = np.array(sim.data.qpos[jidx])
        knuckle = float(sim.data.qpos[gidx[0]]) / tv.ROBOTIQ_REAL_TO_SIM
        arm.set_state(q, None, [knuckle])
        ctrl = raw.robots[0].controller
        if args.teacher_force and t > 0:
            arm.q_ref, arm.v_ref = np.array(ctrl.q_ref), np.array(ctrl.v_ref)
        mine = arm.exec_obs(q, srv.last_action)
        ref = obs["state"].cpu().numpy()[-EXEC_OBS_DIM:]
        (rows if t > 0 else first).append(np.abs(mine - ref))
        with torch.no_grad():
            a = agent.act(obs, eval_mode=True).cpu().numpy().astype(np.float64)
        a = np.clip(a, -1.0, 1.0)
        srv.last_action = a.copy()
        warn = []
        q_ref, resynced = arm.step_ref(q, a, [knuckle], srv.a.resync_m)
        n_resync += resynced
        q_ref = srv._clamp_box(q_ref, arm.eef(), warn)
        srv._speed_caps(q, q_ref, warn)
        n_clamp += bool(warn)
        if warn: print(f"  server ep {ep} t {t}: {warn}", flush=True)
        n_ref_clamp += ctrl.clamps
        ctrl.clamps = 0
        obs, _, terminal, success, _ = env.step(torch.from_numpy(a).float())
        t += 1
    print(f"ep {ep}: {t} steps, success {success}, max |diff| t>0 {np.max(rows[-(t - 1):]):.4f}, t=0 {first[-1].max():.4f}",
          flush=True)
d = np.array(rows)
print(f"t = 0 (base DR artefact, see above): max |diff| dp {np.max(np.array(first)[:, :3]):.4f} "
      f"drot {np.max(np.array(first)[:, 3:6]):.4f} rest {np.max(np.array(first)[:, 6:]):.4f}")
print(f"\n{len(d)} steps | server clamps/caps {n_clamp}, resyncs {n_resync} | sim clamps {n_ref_clamp}")
for k in dict.fromkeys(names):
    cols = [i for i, n in enumerate(names) if n == k]
    print(f"  {k:7s} median {np.median(d[:, cols]):.5f}  p99 {np.percentile(d[:, cols], 99):.5f}  max {d[:, cols].max():.5f}")
