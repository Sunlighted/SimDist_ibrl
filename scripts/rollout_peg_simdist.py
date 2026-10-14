"""Roll out a SimDist stage-1 ckpt on PegInsertion (photoreal obs) and say WHERE it fails.

Env = the ckpt's own env params, with hand-init sigma forced to the replay's 0.1
(the stage-1 configs never set it, so their eval env ran robosuite's 0.02).
Per episode: lifted (peg > 2 cm off the table), closest tip-to-pocket xy while the
tip is above the board, deepest reach in the pocket column, success. Records the
first --video episodes (both cameras side by side, as the policy sees them).

    python scripts/rollout_peg_simdist.py --weight <stage1>/latest.pt --n 10 --out <dir>
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import real2sim.splat_obs  # noqa: F401,E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_simdist  # noqa: E402
from env.robosuite_wrapper import PixelRobosuite  # noqa: E402
from common_utils import ibrl_utils as utils  # noqa: E402


def to_hwc(x):
    x = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if x.ndim == 4:
        x = x[-3:] if x.shape[0] % 3 == 0 and x.shape[1] != 3 else x[-1]
    return np.ascontiguousarray(x.transpose(1, 2, 0) if x.shape[0] == 3 else x).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--video", type=int, default=4)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=31000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    agent, _, env_params = train_simdist.load_model(a.weight, "cuda")
    env_params = dict(env_params)
    print("[ckpt env] sigma", env_params.get("hand_init_perturb_joint_sigma"), "-> forcing", a.sigma,
          "| env_kwargs", env_params.get("env_kwargs"), "| splat", env_params.get("splat_obs") is not None,
          "| cams", env_params.get("rl_cameras"))
    env_params["hand_init_perturb_joint_sigma"] = a.sigma
    env = PixelRobosuite(**env_params)
    cams = list(env_params["rl_cameras"])
    rows = []
    with torch.no_grad(), utils.eval_mode(agent):
        for ep in range(a.n):
            np.random.seed(a.seed + ep)
            obs, _ = env.reset()
            raw, sim = env.env, env.env.sim
            pid, bid = raw.obj_body_id["peg"], raw.obj_body_id["board"]
            z0 = float(sim.data.body_xpos[pid][2])
            lifted, xy_min, depth, succ, t, frames = False, np.inf, -np.inf, False, 0, []
            done = False
            while not done:
                if ep < a.video:
                    frames.append(np.concatenate([to_hwc(obs[c]) for c in cams], axis=1))
                action = agent.act(obs, eval_mode=True)
                obs, _, done, success, _ = env.step(action)
                t += 1
                succ |= bool(success)
                lifted |= float(sim.data.body_xpos[pid][2]) - z0 > 0.02
                bR = np.array(sim.data.body_xmat[bid]).reshape(3, 3)
                d = bR.T @ (raw._peg_tip_pos() - np.array(sim.data.body_xpos[bid]))
                if lifted and d[2] > raw.board.half_z - 0.002:
                    xy_min = min(xy_min, float(np.hypot(d[0], d[1])))
                if abs(d[0]) < raw.board.hole_half_x and abs(d[1]) < raw.board.hole_half_y:
                    depth = max(depth, raw.board.half_z - d[2])
            rows.append((succ, lifted, xy_min * 1000, depth * 1000, t))
            print(f"ep {ep}: succ={int(succ)} lifted={int(lifted)} closest_xy={xy_min*1000:.1f}mm "
                  f"depth={depth*1000:.1f}mm steps={t}", flush=True)
            if ep < a.video:
                imageio.mimsave(os.path.join(a.out, f"ep{ep}.mp4"), frames, fps=20, macro_block_size=1)
    S, L = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"\nSUMMARY n={len(rows)} success {S.mean():.2f} lifted {L.mean():.2f} "
          f"entered pocket {np.mean([np.isfinite(r[3]) for r in rows]):.2f}")


if __name__ == "__main__":
    main()
