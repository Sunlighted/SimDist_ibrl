"""Sanity check: is the counterfactual rollout reproducible?

Given the SAME demo sim_state + SAME action sequence, does the source sim produce
the SAME trajectory every time? If not, the "GT gap" ||s'_tgt - s'_src|| is
contaminated by sim nondeterminism / set_state reconstruction error rather than
being a pure dynamics gap.

Three tests:
  (1) reconstruction repeatability : set_state(sim_state) -> read state, N times.
  (2) rollout repeatability        : set_state + apply k actions, N times, back to back.
  (3) controller contamination     : same, but scramble the env with a DIFFERENT
                                      anchor between repeats (exposes any OSC /
                                      interpolator internal state not in sim_state).
Reports per-step max deviation across repeats (quaternion-aware state_gap), and
compares it to the set_state reconstruction floor and the typical GT gap (~0.005).
"""

import argparse
import os

import numpy as np

from cf_utils import (REPO, get_source_env, load_demo_episodes, counterfactual_step,
                      reconstructed_state)
from data_utils import state_gap
from step15_build_multistep_pairs import counterfactual_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_hdf5", default=f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--source_weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--n_repeat", type=int, default=5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n_anchor", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    eps = load_demo_episodes(args.target_hdf5)
    env = get_source_env(args.source_weight)
    rng = np.random.default_rng(args.seed)

    # pick anchors (episode, t) with room for k steps
    anchors = []
    for _ in range(args.n_anchor):
        ei = int(rng.integers(0, len(eps)))
        T = eps[ei]["state"].shape[0] - 1
        t = int(rng.integers(0, max(1, T - args.k)))
        anchors.append((ei, t))

    # ---- (1) reconstruction repeatability ----
    print("=== (1) set_state reconstruction repeatability (N repeats, no action) ===")
    recon_dev = []
    for ei, t in anchors:
        sim = eps[ei]["sim_state"][t]
        recs = np.stack([reconstructed_state(env, sim) for _ in range(args.n_repeat)])
        dev = max(state_gap(recs[0:1], recs[j:j+1])[0] for j in range(1, args.n_repeat))
        # also vs the demo's recorded state at t (set_state fidelity)
        fidelity = state_gap(recs[0:1], eps[ei]["state"][t:t+1])[0]
        recon_dev.append(dev)
        print(f"  ep{ei:2d} t{t:3d}: max dev across repeats={dev:.2e} | "
              f"recon vs demo state={fidelity:.4f}")

    # ---- (2) rollout repeatability (back to back) ----
    print(f"\n=== (2) k={args.k}-step rollout repeatability (N={args.n_repeat} back-to-back) ===")
    roll_dev_terminal = []
    for ei, t in anchors:
        sim = eps[ei]["sim_state"][t]; acts = eps[ei]["action"][t:t+args.k]
        rolls = np.stack([counterfactual_rollout(env, sim, acts) for _ in range(args.n_repeat)])  # (N,k,37)
        # per-step max deviation across repeats
        per_step = [max(state_gap(rolls[0, j:j+1], rolls[r, j:j+1])[0]
                        for r in range(1, args.n_repeat)) for j in range(args.k)]
        roll_dev_terminal.append(per_step[-1])
        print(f"  ep{ei:2d} t{t:3d}: per-step max dev = "
              f"[{', '.join(f'{d:.1e}' for d in per_step)}]")

    # ---- (3) controller / internal-state contamination ----
    print(f"\n=== (3) contamination: scramble env with a DIFFERENT anchor between repeats ===")
    other = eps[anchors[0][0]]
    for ei, t in anchors[:3]:
        sim = eps[ei]["sim_state"][t]; acts = eps[ei]["action"][t:t+args.k]
        clean = counterfactual_rollout(env, sim, acts)
        # scramble: run a different rollout, THEN redo the same one
        ot = anchors[-1]
        _ = counterfactual_rollout(env, eps[ot[0]]["sim_state"][ot[1]],
                                   eps[ot[0]]["action"][ot[1]:ot[1]+args.k])
        after = counterfactual_rollout(env, sim, acts)
        dev = max(state_gap(clean[j:j+1], after[j:j+1])[0] for j in range(args.k))
        print(f"  ep{ei:2d} t{t:3d}: max dev (clean vs after-scramble) = {dev:.2e}")

    print(f"\n=== summary ===")
    print(f"  reconstruction repeat dev : max {max(recon_dev):.2e} (should be ~0 = deterministic)")
    print(f"  rollout terminal repeat dev: max {max(roll_dev_terminal):.2e}")
    print(f"  reference: typical GT gap median ~0.005-0.007, set_state fidelity ~0.003-0.006")
    print(f"  -> if repeat dev << GT gap, the counterfactual is reproducible and the gap is real.")


if __name__ == "__main__":
    main()
