"""Can the state expert insert with the REAL in-hand needle pose? (real_1027/README.md, sim to-do step 1)

Runs the frozen exec_obs IBRL expert in its own env. When the needle is first lifted in a closed gripper, the needle
is moved once to a given pose in the gripper0_grip_site frame (velocity zeroed) and the episode continues closed loop.
  --pose none      no move (baseline)
  --pose sim       the sim's own mean in-hand pose (controls for the move itself)
  --pose real      the real 6-grasp mean (results/inhand_real_1027.json)
  --pose percycle  cycles over the 6 real grasps
  --pose self      re-places the needle at its own current pose (velocity zeroed): controls for the move procedure

  python scripts/test_inhand_pose_expert.py --pose real --episodes 100
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

ap = argparse.ArgumentParser()
ap.add_argument("--expert", default="exps/sim2real/threading_robotiq_execobs_expert/expert.pt")
ap.add_argument("--inhand", default="real2sim/zed_capture/calib_20261015/real_1027/results/inhand_real_1027.json")
ap.add_argument("--pose", default="real", choices=["none", "self", "sim", "real", "percycle"])
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lift_m", type=float, default=0.03, help="needle root this far above its reset height = lifted")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch  # noqa: E402
import train_rl  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

js = json.load(open(args.inhand))
poses = {"sim": [js["sim_reference"]], "real": [dict(pos=js["real_summary"]["pos_mean"], rot=js["real_summary"]["rot_mean"])],
         "percycle": [js["real_per_grasp"][k] for k in sorted(js["real_per_grasp"])], "none": [None], "self": ["self"]}[args.pose]


def orthonormal(r):
    u, _, vt = np.linalg.svd(np.asarray(r, float).reshape(3, 3))   # the mean of rotations is not a rotation
    return u @ vt


agent, env, _ = train_rl.load_model(args.expert, "cuda")
agent.eval()
rows = []
for ep in range(args.episodes):
    np.random.seed(args.seed + ep)
    torch.manual_seed(args.seed + ep)
    obs, _ = env.reset()
    raw = env.env
    sim, m = raw.sim, raw.sim.model
    gs, nb = m.site_name2id("gripper0_grip_site"), m.body_name2id("needle_obj_root")
    jid = m.joint_name2id("needle_obj_joint0")
    qa, va = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    z0 = float(sim.data.body_xpos[nb][2])
    pose = poses[ep % len(poses)]
    moved_at, slip, terminal, success, t = None, None, False, False, 0
    while not terminal:
        with torch.no_grad():
            a = agent.act(obs, eval_mode=True).cpu().numpy()
        obs, _, terminal, success, _ = env.step(torch.from_numpy(a).float())
        t += 1
        Rg, pg = sim.data.site_xmat[gs].reshape(3, 3), sim.data.site_xpos[gs]
        if moved_at is None and a[6] > 0 and sim.data.body_xpos[nb][2] - z0 > args.lift_m:
            moved_at = t
            sim.forward()   # after mj_step the x* poses are one substep behind qpos: writing them back drags the needle
            Rg, pg = sim.data.site_xmat[gs].reshape(3, 3), sim.data.site_xpos[gs]   # into the fingers
            if pose == "self":
                pose = dict(pos=Rg.T @ (sim.data.body_xpos[nb] - pg), rot=Rg.T @ sim.data.body_xmat[nb].reshape(3, 3))
            if pose is not None:
                sim.data.qpos[qa:qa + 3] = pg + Rg @ np.asarray(pose["pos"])
                q = Rotation.from_matrix(Rg @ orthonormal(pose["rot"])).as_quat()     # xyzw
                q = np.array([q[3], q[0], q[1], q[2]])
                if q @ sim.data.qpos[qa + 3:qa + 7] < 0:   # the state carries the raw quaternion: keep its sign
                    q = -q                                    # (a flipped sign alone: self 0.87 -> 0.18)
                sim.data.qpos[qa + 3:qa + 7] = q
                sim.data.qvel[va:va + 6] = 0
                sim.forward()
            rel0 = Rg.T @ (sim.data.body_xpos[nb] - pg)
        if moved_at is not None and t == moved_at + 10:   # does the hand keep the pose? (mm, grip frame)
            slip = float(np.linalg.norm(Rg.T @ (sim.data.body_xpos[nb] - pg) - rel0) * 1000)
    rows.append(dict(ep=ep, success=bool(success), steps=t, moved_at=moved_at, slip_mm=slip))
    print(f"ep {ep}: success {success} steps {t} lifted@{moved_at} slip10 {slip}", flush=True)
sr = np.mean([r["success"] for r in rows])
grasped = [r for r in rows if r["moved_at"] is not None]
print(f"\npose={args.pose}: success {sr:.3f} ({sum(r['success'] for r in rows)}/{len(rows)}), "
      f"lifted {len(grasped)}, success|lifted {np.mean([r['success'] for r in grasped]) if grasped else float('nan'):.3f}, "
      f"median slip {np.median([r['slip_mm'] for r in grasped if r['slip_mm'] is not None]):.1f} mm")
if args.out:
    json.dump(dict(args=vars(args), rows=rows), open(args.out, "w"))
