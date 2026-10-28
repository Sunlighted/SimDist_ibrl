"""Check a cell's needle_com_y / inhand_dr_* with the scripted expert (2026-10-28, real_1027/README.md).

Per episode: success, the drawn in-hand pose vs the one measured right after the re-place (along / depth [mm],
pitch / yaw [deg] in the grip-site frame, same convention as the cell fields), the slip 10 steps later and whether the
needle stayed in the hand.
  --prob 1          every episode re-placed (stress test; the cell's value otherwise)
  --draw real       every draw = the real 6-grasp mean (results/inhand_real_1027.json; the acceptance pose)

  python scripts/test_inhand_dr_cell.py --cell real2sim/zed_capture/calib_20261015/threading_real_cell_20261028.yaml --prob 1
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True)
ap.add_argument("--prob", type=float, default=None)
ap.add_argument("--draw", default="", choices=["", "real"])
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed", type=int, default=900)
ap.add_argument("--out", default="")
args = ap.parse_args()

import torch  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from env.sim2real import load_sim2real  # noqa: E402
from scripts.threading_robotiq_expert import ThreadingExpert  # noqa: E402

cell = load_sim2real(args.cell)
if args.prob is not None:
    cell = dataclasses.replace(cell, inhand_dr_prob=args.prob)
env = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, camera_names=[], rl_cameras=[], use_state=True,
                     state_only_no_render=True, device="cpu", hand_init_perturb_joint_sigma=0.1, end_on_success=True,
                     sim2real=cell, exec_obs=True)
REAL = json.load(open("real2sim/zed_capture/calib_20261015/real_1027/results/inhand_real_1027.json"))["real_summary"]


def inhand(sim):
    m, d = sim.model, sim.data
    gs, nb = m.site_name2id("gripper0_grip_site"), m.body_name2id("needle_obj_root")
    Rg, pg = d.site_xmat[gs].reshape(3, 3), d.site_xpos[gs]
    R = Rg.T @ d.body_xmat[nb].reshape(3, 3)
    s = 1.0 if R[1, 1] >= 0 else -1.0
    h = Rg.T @ (d.body_xpos[nb] - pg) + R @ np.array([0, 0.06, 0])
    tip = -R[:, 1]
    return dict(along=s * h[1] * 1000, depth=h[2] * 1000, pitch=float(np.degrees(np.arcsin(tip[2]))),
                yaw=float(np.degrees(np.arctan2(tip[0], -s * tip[1]))) * s, pos=Rg.T @ (d.body_xpos[nb] - pg))


rows = []
for ep in range(args.episodes):
    np.random.seed(args.seed + ep)
    obs, _ = env.reset()
    if args.draw == "real" and env._inhand_pending:
        env._inhand_draw = dict(along=REAL["handle_center_in_grip_mm"][0][1] / 1000, depth=REAL["handle_center_in_grip_mm"][0][2] / 1000,
                                pitch=np.radians(REAL["pitch_deg"][0]), yaw=np.radians(REAL["yaw_deg"][0]))
    ex = ThreadingExpert(env.env, noise=0.0, rng=np.random.default_rng(args.seed + ep))
    sim = env.env.sim
    term, succ, t, meas, slip, phase = False, False, 0, None, None, ""
    while not term:
        a, info = ex.act()
        obs, _, term, succ, _ = env.step(torch.from_numpy(a.astype(np.float32)))
        t += 1
        phase = info["phase"]
        if env.last_inhand_dr is not None and meas is None:
            meas = inhand(sim)
        if meas is not None and slip is None and t >= env.last_inhand_dr["t"] + 10:
            slip = float(np.linalg.norm(inhand(sim)["pos"] - meas["pos"]) * 1000)
    dr = env.last_inhand_dr
    rows.append(dict(ep=ep, success=bool(succ), steps=t, last_phase=phase, draw=dr,
                     meas={k: round(float(v), 2) for k, v in meas.items() if k != "pos"} if meas else None, slip_mm=slip))
    print(f"ep {ep}: {'OK  ' if succ else 'FAIL'} {t:3d} steps, last {phase:7s} | draw {dr} | meas {rows[-1]['meas']} | slip {slip}",
          flush=True)
sr = np.mean([r["success"] for r in rows])
dr_rows = [r for r in rows if r["draw"]]
print(f"\nsuccess {sr:.3f} ({sum(r['success'] for r in rows)}/{len(rows)}); re-placed {len(dr_rows)}, success|re-placed "
      f"{np.mean([r['success'] for r in dr_rows]) if dr_rows else float('nan'):.3f}, success|not "
      f"{np.mean([r['success'] for r in rows if not r['draw']]) if len(dr_rows) < len(rows) else float('nan'):.3f}")
if dr_rows:
    for k in ("along", "depth", "pitch", "yaw"):
        e = [r["meas"][k] - r["draw"][f"{k}_mm" if k in ("along", "depth") else f"{k}_deg"] for r in dr_rows]
        print(f"  {k}: |meas - draw| median {np.median(np.abs(e)):.2f} max {np.max(np.abs(e)):.2f}")
    print(f"  slip 10 steps later: median {np.median([r['slip_mm'] for r in dr_rows if r['slip_mm'] is not None]):.1f} mm, "
          f"p90 {np.percentile([r['slip_mm'] for r in dr_rows if r['slip_mm'] is not None], 90):.1f} mm")
if args.out:
    json.dump(dict(args=vars(args), rows=rows), open(args.out, "w"))
