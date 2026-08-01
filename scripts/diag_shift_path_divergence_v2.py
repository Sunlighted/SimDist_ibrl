"""Single-env diagnostic. Each invocation creates ONE env (matching real
eval), reproduces a fixed action trajectory, and dumps obs at t=0 / t=10 /
t=50 / t=100. Run twice (--mode none vs --mode fixed) and diff offline.

Usage:
  python scripts/diag_shift_path_divergence_v2.py --mode none  --out /tmp/diag_none.npz
  python scripts/diag_shift_path_divergence_v2.py --mode fixed --out /tmp/diag_fixed.npz
  python scripts/diag_shift_path_divergence_v2.py --diff /tmp/diag_none.npz /tmp/diag_fixed.npz
"""
import argparse
import os
import sys

import numpy as np


def run_single_env(mode_str, out_path, n_steps=100, seed=1, dump_raw_obs=True):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    from env.robosuite_wrapper import PixelRobosuite
    from env.domain_shift import DomainShiftConfig

    if mode_str == "none":
        shift = None
    elif mode_str == "fixed":
        shift = DomainShiftConfig(mode="fixed")  # all defaults = 1.0/0
    else:
        raise ValueError(mode_str)

    env = PixelRobosuite(
        env_name="Threading_D0",
        robots=["Panda"],
        episode_length=400,
        image_size=96, rl_image_size=96,
        camera_names=["agentview"], rl_cameras=["agentview"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_state=True, obs_stack=1, state_stack=1, prop_stack=1,
        domain_shift=shift,
    )
    np.random.seed(seed)
    obs, _ = env.reset()

    snapshots = {}
    def snap(t, obs):
        s = obs["state"]
        if hasattr(s, "cpu"):
            s = s.cpu().numpy()
        snapshots[f"obs_t{t}"] = np.asarray(s, dtype=np.float64)
        snapshots[f"qpos_t{t}"] = env.env.sim.data.qpos.copy()
        snapshots[f"qvel_t{t}"] = env.env.sim.data.qvel.copy()
        if dump_raw_obs:
            raw = env.env._get_observations(force_update=False)
            for k in ["robot0_eef_pos", "robot0_eef_quat",
                      "robot0_gripper_qpos", "object-state",
                      "robot0_joint_pos", "robot0_joint_vel"]:
                if k in raw:
                    snapshots[f"raw_{k}_t{t}"] = np.asarray(raw[k], dtype=np.float64)
            # also: the gripper's body_xpos / body_xquat fresh from sim.data
            sim = env.env.sim
            bid = sim.model.body_name2id("gripper0_eef")
            snapshots[f"body_xpos_t{t}"]  = sim.data.body_xpos[bid].copy()
            snapshots[f"body_xquat_t{t}"] = sim.data.body_xquat[bid].copy()

    snap(0, obs)

    # Deterministic action sequence — small periodic perturbations
    rng = np.random.default_rng(seed + 100)
    action_seq = rng.normal(size=(n_steps, env.action_dim)) * 0.1

    targets = {10, 50, 100}
    for t in range(1, n_steps + 1):
        a = torch.from_numpy(action_seq[t - 1])
        obs, _, _, _, _ = env.step(a)
        if t in targets:
            snap(t, obs)

    np.savez(out_path, **snapshots)
    print(f"[{mode_str}] wrote {out_path}  (obs/qpos/qvel at t = 0, 10, 50, 100)")
    # Quick fingerprint for stdout
    for t in [0, 10, 50, 100]:
        s = snapshots[f"obs_t{t}"]
        print(f"  t={t:3d}: obs[:5] = {s[:5]}, ||obs||={np.linalg.norm(s):.6f}")


def diff_npz(path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    print(f"diff: {path_a} vs {path_b}")
    for k in sorted(a.files):
        if k not in b.files:
            print(f"  {k}: missing in B")
            continue
        da, db = a[k], b[k]
        diff = da - db
        print(f"  {k:12s}  max|diff|={np.abs(diff).max():.3e}  L2={np.linalg.norm(diff):.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default=None, choices=["none", "fixed"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--diff", nargs=2, default=None,
                    help="Two npz files to diff (skips env run)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.diff:
        diff_npz(*args.diff)
        return

    assert args.mode is not None and args.out is not None
    run_single_env(args.mode, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
