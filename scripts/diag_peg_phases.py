"""Where does a PegInsertion policy fail? Per-phase stats over N episodes.

Phases: lifted (peg raised > 2 cm off the table), closest xy approach of the
peg tip to the pocket centre while the tip is above the board, deepest reach
inside the pocket column, final tilt. Same env for every --kind: sigma 0.1 +
board jitter +-10 mm / +-5 deg + the env's current success/contact defaults.

    python scripts/diag_peg_phases.py --kind bc     --weight <bc model0.pt>
    python scripts/diag_peg_phases.py --kind rl     --weight <ibrl snapshot.pt>
    python scripts/diag_peg_phases.py --kind expert
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from scripts.smoke_peg_insertion import scripted_expert_action  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["bc", "rl", "expert"], required=True)
    p.add_argument("--weight", default=None)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=55000)
    args = p.parse_args()

    env = PixelRobosuite(
        env_name="PegInsertion_D0", robots=["Panda"], episode_length=300, use_state=True,
        state_only_no_render=True, device="cuda", hand_init_perturb_joint_sigma=0.1,
        end_on_success=True, env_kwargs=dict(board_pos_noise=0.01, board_yaw_noise=5.0))
    if args.kind == "bc":
        import train_bc
        policy, _, _ = train_bc.load_model(args.weight, "cuda", verbose=False)
        policy.train(False)   # BC act() asserts eval mode
        act = lambda obs, raw, ctl: policy.act(obs)  # noqa: E731
    elif args.kind == "rl":
        import train_rl
        agent, _, _ = train_rl.load_model(args.weight, "cuda")
        agent.train(False)
        act = lambda obs, raw, ctl: agent.act(obs, eval_mode=True)  # noqa: E731
    else:
        act = lambda obs, raw, ctl: torch.from_numpy(scripted_expert_action(raw, ctl))  # noqa: E731

    R = []
    with torch.no_grad():
        for ep in range(args.n):
            np.random.seed(args.seed + ep)
            obs, _ = env.reset()
            raw, sim = env.env, env.env.sim
            pid, bid = raw.obj_body_id["peg"], raw.obj_body_id["board"]
            z0 = float(sim.data.body_xpos[pid][2])
            lifted, xy_min, depth, ctl, succ, t = False, np.inf, -np.inf, {}, False, 0
            terminal = False
            while not terminal:
                obs, _, terminal, success, _ = env.step(act(obs, raw, ctl))
                t += 1
                succ |= bool(success)
                lifted |= float(sim.data.body_xpos[pid][2]) - z0 > 0.02
                bR = np.array(sim.data.body_xmat[bid]).reshape(3, 3)
                d = bR.T @ (raw._peg_tip_pos() - np.array(sim.data.body_xpos[bid]))
                if lifted and d[2] > raw.board.half_z - 0.002:          # tip at/above the board top
                    xy_min = min(xy_min, float(np.hypot(d[0], d[1])))
                if abs(d[0]) < raw.board.hole_half_x and abs(d[1]) < raw.board.hole_half_y:
                    depth = max(depth, raw.board.half_z - d[2])
            Rp = np.array(sim.data.body_xmat[pid]).reshape(3, 3)
            tilt = np.degrees(np.arccos(min(1.0, abs(Rp[2, 2]))))
            R.append((succ, lifted, xy_min * 1000, depth * 1000, tilt, t))
    S, L, XY, D, TL, T = (np.array(x) for x in zip(*R))
    print(f"[{args.kind}] n={len(S)}  success {S.mean():.2f}   lifted {L.mean():.2f}")
    xy = XY[np.isfinite(XY)]
    if len(xy):
        print(f"  closest tip-to-pocket-centre (xy, mm) when above board: median {np.median(xy):.1f}  "
              f"p25 {np.percentile(xy, 25):.1f}  (<1.3 mm/side clearance; within 5 mm: {np.mean(xy < 5):.2f})")
    dd = D[np.isfinite(D)]
    print(f"  entered pocket column {np.mean(np.isfinite(D)):.2f}; deepest reach median "
          f"{np.median(dd) if len(dd) else np.nan:.1f} mm (>=30 needed)")
    print(f"  final tilt median {np.median(TL):.1f} deg; episode length median {int(np.median(T))}")
    print("  per-episode (succ lifted xy_mm depth_mm tilt):")
    for r in R[:10]:
        print(f"    {int(r[0])} {int(r[1])} {r[2]:6.1f} {r[3]:6.1f} {r[4]:5.1f}")


if __name__ == "__main__":
    main()
