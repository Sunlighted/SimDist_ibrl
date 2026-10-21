"""Audit the scripted-expert demos of the Threading Robotiq cell (threading_robotiq_expert_n1000) against the STRICT
success that train_rl / eval use (2026-10-21).

collect_threading_robotiq_demos.py built its env WITHOUT strict_success and with end_on_success=1, so every demo ends
on the first RAW success frame. Here each demo is re-run from its stored seed with the same expert, in the same cell,
but with strict success on and end_on_success off, and the expert keeps acting for `--extra` steps past the demo end:
  - reproducible?   initial sim_state and every recorded action match the re-run;
  - strict?         does the trajectory reach strict success (dwell 3, tripod displacement <= 4 cm), and when;
  - per-step trace  raw success, tripod displacement, expert phase -- saved to an npz for the plots.

    python scripts/check_threading_robotiq_demos.py --data <dir>/dataset.hdf5 --out <npz> --workers 20 [--num 1000]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        import env.threading_variants as tv
        from env.robosuite_wrapper import PixelRobosuite
        from env.sim2real import load_sim2real
        from env.strict_success import StrictSuccessConfig
        _ENV = PixelRobosuite("Threading_RealCellRobotiq", "Panda", 400, camera_names=[], rl_cameras=[],
                              use_state=True, state_only_no_render=True, record_sim_state=True, device="cpu",
                              hand_init_perturb_joint_sigma=0.1, end_on_success=False,
                              sim2real=load_sim2real(tv.REAL_CELL_ROBOTIQ_YAML),
                              strict_success=StrictSuccessConfig(enabled=True, dwell_steps=3, max_tripod_disp=0.04))
    return _ENV


def _pose_in_base(sim, body):
    """xy [m] and yaw [deg] in the robot base frame (the cell's layout ranges are base-frame; base DR moves the base)."""
    Rb, pb = sim.data.get_body_xmat("robot0_base"), sim.data.get_body_xpos("robot0_base")
    R = Rb.T @ sim.data.get_body_xmat(body)
    return (Rb.T @ (sim.data.get_body_xpos(body) - pb))[:2].copy(), float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def check(job):
    import h5py
    import torch
    from env.strict_success import SuccessValidator
    from scripts.threading_robotiq_expert import ThreadingExpert
    path, key, extra = job
    with h5py.File(path, "r") as f:
        g = f["data"][key]
        seed, acts, ss0 = int(g.attrs["seed"]), g["actions"][:], g["obs/sim_state"][0]
        rew = g["rewards"][:]
    env = _env()
    np.random.seed(seed)
    obs, _ = env.reset()
    sim = env.env.sim
    ex = ThreadingExpert(env.env, noise=0.0, rng=np.random.default_rng(seed))
    T = len(acts)
    (nxy, nyaw), (txy, tyaw) = _pose_in_base(sim, "needle_obj_root"), _pose_in_base(sim, "tripod_obj_root")
    init = dict(needle_xy=nxy, needle_yaw=nyaw, tripod_xy=txy, tripod_yaw=tyaw)
    ss_err = float(np.max(np.abs(obs["sim_state"].numpy() - ss0)))
    val = env._success_validator
    raw, strict, disp, phase, act_err = [], [], [], [], 0.0
    first_strict = -1
    for t in range(min(T + extra, 400)):
        a, info = ex.act()
        if t < T:
            act_err = max(act_err, float(np.max(np.abs(a.astype(np.float32) - acts[t]))))
        obs, r, term, succ, _ = env.step(torch.from_numpy(a.astype(np.float32)))
        raw.append(bool(env.env._check_success()))
        strict.append(bool(succ))
        c = SuccessValidator._ring_center(env.env, env._bindings)
        disp.append(float(np.linalg.norm(c - val._ring_center0)))
        phase.append(info["phase"])
        if succ and first_strict < 0:
            first_strict = t
        if first_strict >= 0 and t >= max(T, first_strict) + 5:
            break
    return dict(key=key, seed=seed, T=T, demo_reward_sum=float(rew.sum()), demo_last_reward=float(rew[-1]),
                ss_err=ss_err, act_err=act_err, first_strict=first_strict, raw=np.array(raw), strict=np.array(strict),
                disp=np.array(disp), phase=np.array(phase), **init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--extra", type=int, default=40, help="expert steps allowed past the demo end")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    import h5py
    with h5py.File(args.data, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:args.num]
    with Pool(args.workers) as p:
        res = []
        for i, r in enumerate(p.imap(check, [(args.data, k, args.extra) for k in keys], chunksize=4)):
            res.append(r)
            if (i + 1) % 50 == 0:
                print(f"[check] {i + 1}/{len(keys)}", flush=True)
    fs = np.array([r["first_strict"] for r in res])
    T = np.array([r["T"] for r in res])
    print(f"demos {len(res)} | init sim_state max err {max(r['ss_err'] for r in res):.2e} | "
          f"action max err {max(r['act_err'] for r in res):.2e}")
    print(f"raw success at demo end: {np.mean([r['raw'][r['T'] - 1] for r in res]):.3f} | demo reward sum==1: "
          f"{np.mean([r['demo_reward_sum'] == 1 for r in res]):.3f}")
    print(f"strict success within T+{args.extra}: {np.mean(fs >= 0):.3f} | within T: {np.mean((fs >= 0) & (fs < T)):.3f}"
          f" | steps past demo end (median, of the strict ones): {np.median((fs - T + 1)[fs >= 0]):.0f}")
    print(f"max tripod disp over the demo: median {np.median([r['disp'][:r['T']].max() for r in res]) * 1e3:.1f} mm, "
          f">40 mm {np.mean([r['disp'][:r['T']].max() > 0.04 for r in res]):.3f}")
    np.savez_compressed(args.out, res=np.array(res, dtype=object))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
