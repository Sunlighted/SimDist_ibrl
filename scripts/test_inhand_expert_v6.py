"""The frozen v6 expert (10-28 cell, trained with in-hand DR) under forced in-hand poses (2026-10-28).
  --mode off       no re-placement
  --mode uniform   every episode re-placed, the cell's uniform DR draw
  --mode real      every episode re-placed at the real 6-grasp mean (results/inhand_real_1027.json)
  --mode realdist  every episode re-placed, drawn from N(real mean, real std) per axis
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
ap = argparse.ArgumentParser()
ap.add_argument("--expert", default="exps/sim2real/threading_robotiq_v6_expert/expert.pt")
ap.add_argument("--mode", default="real", choices=["off", "uniform", "real", "realdist"])
ap.add_argument("--episodes", type=int, default=150)
ap.add_argument("--seed", type=int, default=5000)
ap.add_argument("--range_cell", default="", help="draw --mode uniform from this cell's inhand_dr_* ranges")
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch  # noqa: E402
import train_rl  # noqa: E402

R = json.load(open("real2sim/zed_capture/calib_20261015/real_1027/results/inhand_real_1027.json"))["real_summary"]
mu = dict(along=R["handle_center_in_grip_mm"][0][1], depth=R["handle_center_in_grip_mm"][0][2], pitch=R["pitch_deg"][0], yaw=R["yaw_deg"][0])
sd = dict(along=R["handle_center_in_grip_mm"][1][1], depth=R["handle_center_in_grip_mm"][1][2], pitch=R["pitch_deg"][1], yaw=R["yaw_deg"][1])
agent, env, _ = train_rl.load_model(args.expert, "cuda")
agent.eval()
c = env.sim2real if not args.range_cell else __import__("env.sim2real", fromlist=["x"]).load_sim2real(args.range_cell)
print(f"[cell] needle_com_y {c.needle_com_y} inhand_dr_prob {c.inhand_dr_prob} | real mean {mu} sd {sd}", flush=True)
rows = []
for ep in range(args.episodes):
    np.random.seed(args.seed + ep)
    torch.manual_seed(args.seed + ep)
    obs, _ = env.reset()
    if args.mode == "off":
        env._inhand_pending = False
    else:
        if args.mode == "uniform":
            w = dict(along=np.random.uniform(*c.inhand_dr_along_mm), depth=np.random.uniform(*c.inhand_dr_depth_mm),
                     pitch=np.random.uniform(*c.inhand_dr_pitch_deg), yaw=np.random.uniform(*c.inhand_dr_yaw_deg))
        elif args.mode == "real":
            w = dict(mu)
        else:
            w = {k: np.random.normal(mu[k], sd[k]) for k in mu}
        env._inhand_draw = dict(along=w["along"] / 1000, depth=w["depth"] / 1000, pitch=np.radians(w["pitch"]), yaw=np.radians(w["yaw"]))
        env._inhand_pending = True
        env._inhand_z0 = float(env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("needle_obj_root")][2])
    term, succ, t = False, False, 0
    while not term:
        with torch.no_grad():
            a = agent.act(obs, eval_mode=True)
        obs, _, term, succ, _ = env.step(a)
        t += 1
    rows.append(dict(ep=ep, success=bool(succ), steps=t, draw=env.last_inhand_dr))
    print(f"ep {ep}: {'OK  ' if succ else 'FAIL'} {t} steps | {env.last_inhand_dr}", flush=True)
re = [r for r in rows if r["draw"]]
print(f"\nmode={args.mode}: success {np.mean([r['success'] for r in rows]):.3f} ({sum(r['success'] for r in rows)}/{len(rows)}), "
      f"re-placed {len(re)}, success|re-placed {np.mean([r['success'] for r in re]) if re else float('nan'):.3f}")
if args.out:
    json.dump(dict(args=vars(args), rows=rows), open(args.out, "w"))
