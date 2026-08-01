"""Confirm the mechanism (un-restored MjData / solver warmstart) and test a fix.

For the anchors that diverged in step18, compare repeat-determinism of the k-step
rollout under three reset strategies before stepping:
  (a) current : set_state_from_flattened + forward            (step18 behavior)
  (b) +warmstart0 : also zero data.qacc_warmstart before forward
  (c) full reset  : sim.reset() (clears all MjData) THEN set_state + forward
If (b) or (c) drives the repeat deviation to ~0, the divergence was MjData state
not captured by the 45-dim flattened (time,qpos,qvel), and (c)/(b) is the fix to
make counterfactual GT gaps deterministic.
"""

import argparse
import os
import numpy as np

from cf_utils import REPO, get_source_env, load_demo_episodes, extract_state
from data_utils import state_gap


def rollout(env, sim_state, actions, mode):
    raw = env.env
    if mode == "full_reset":
        raw.sim.reset()
    raw.done = False; raw.timestep = 0
    raw.sim.set_state_from_flattened(np.asarray(sim_state, np.float64))
    if mode == "warmstart0":
        raw.sim.data.qacc_warmstart[:] = 0.0
    raw.sim.forward()
    out = []
    for a in actions:
        raw.done = False
        obs, _, _, _ = raw.step(np.asarray(a, np.float64))
        out.append(extract_state(obs))
    return np.asarray(out, np.float32)


def repeat_dev(env, sim, acts, mode, n=5):
    rolls = np.stack([rollout(env, sim, acts, mode) for _ in range(n)])
    return max(state_gap(rolls[0, -1:], rolls[r, -1:])[0] for r in range(1, n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_hdf5", default=f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--source_weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n_repeat", type=int, default=5)
    args = p.parse_args()

    eps = load_demo_episodes(args.target_hdf5)
    env = get_source_env(args.source_weight)
    # the divergent anchors from step18 (+ a clean one as control)
    anchors = [(3, 67), (17, 51), (6, 3), (10, 23)]

    # confirm qacc_warmstart exists
    has_ws = hasattr(env.env.sim.data, "qacc_warmstart")
    print(f"sim.data.qacc_warmstart present: {has_ws}")

    print(f"\n{'anchor':12s} {'(a) current':>12s} {'(b) warmstart0':>15s} {'(c) full_reset':>15s}")
    for ei, t in anchors:
        sim = eps[ei]["sim_state"][t]; acts = eps[ei]["action"][t:t + args.k]
        a = repeat_dev(env, sim, acts, "current", args.n_repeat)
        b = repeat_dev(env, sim, acts, "warmstart0", args.n_repeat) if has_ws else float("nan")
        c = repeat_dev(env, sim, acts, "full_reset", args.n_repeat)
        print(f"  ep{ei:2d} t{t:3d}  {a:>12.2e} {b:>15.2e} {c:>15.2e}")

    print(f"\n  (terminal k={args.k} deviation across {args.n_repeat} repeats; ~0 = deterministic)")


if __name__ == "__main__":
    main()
