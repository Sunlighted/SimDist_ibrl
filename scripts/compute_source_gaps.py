"""Compute the GROUND-TRUTH per-transition dynamics gap for SOURCE demos.

For each source-demo transition (sim_state_t, a_t), the recorded next state
s'_src IS the source-dynamics outcome. We counterfactually replay the SAME
(sim_state_t, a_t) in the TARGET env (cold-start, qacc_warmstart=0) to get
s'_tgt, and the quaternion-aware ||s'_src - s'_tgt|| is the pure dynamics gap
between source and target for that transition.

This is exactly classifier_test/step1_build_pairs.py with the roles swapped:
there the RECORDED demo is the target and the COUNTERFACTUAL is the source;
here the recorded demo is the SOURCE and the counterfactual is the target. The
recorded/cold-start asymmetry is identical, so the gap distribution (and hence
the gap_tau=0.024 calibration in CLAUDE.md §16) is directly comparable.

The result is cached as an npz keyed by (episode_idx, step_idx) so
`train_weighted_bc.py` can map gaps onto RobomimicDataset entries (which are
built in the same demo_0, demo_1, ... order, capped at num_data) and turn them
into per-sample BC weights `c = exp(-gap / gap_tau)`.

Usage:
  python scripts/compute_source_gaps.py --num_demos 500
  python scripts/compute_source_gaps.py --num_demos -1   # all 2000 (slow)
"""

import argparse
import os
import sys
import time

import numpy as np

# classifier_test has no __init__; add it to the path and import its helpers
# (cf_utils sets REPO sys.path / cwd / MUJOCO_GL on import).
REPO = "/home/Chenyang/ibrl"
sys.path.insert(0, os.path.join(REPO, "classifier_test"))
from cf_utils import get_env, counterfactual_step, load_demo_episodes  # noqa: E402
from data_utils import state_gap, needle_ring_dist  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source_hdf5",
        default=f"{REPO}/release/data/mimicgen/threading_d0_source_sigma0.1/dataset.hdf5",
    )
    p.add_argument(
        "--source_weight",
        default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt",
        help="any ckpt whose env_params describe the Threading obs space "
        "(used only to BUILD the target env; weights themselves are unused)",
    )
    p.add_argument("--num_demos", type=int, default=500,
                   help="cap on source episodes (-1 = all). Entries are taken "
                        "in demo_0, demo_1, ... order to match RobomimicDataset.")
    p.add_argument("--out", default=None,
                   help="output npz (default: gt_gaps_n{num_demos}.npz next to the hdf5)")
    args = p.parse_args()

    max_ep = None if args.num_demos < 0 else args.num_demos
    eps = load_demo_episodes(args.source_hdf5, max_ep)
    print(f"loaded {len(eps)} source demo episodes from {args.source_hdf5}")

    # Build the TARGET env (dynamics-only shift, matches threading_d0_target_dynamics).
    env = get_env(args.source_weight, domain="target")

    EP, TS, GAP, DIST = [], [], [], []
    t0 = time.time()
    n_trans = 0
    for ei, ep in enumerate(eps):
        state = ep["state"]            # (T, 37)  recorded SOURCE-domain states
        action = ep["action"]          # (T, 7)
        sim_state = ep["sim_state"]    # (T, 45)
        T = state.shape[0]
        # Transitions t -> t+1 exist for t in [0, T-1). The counterfactual gives
        # s'_tgt; the recorded state[t+1] is s'_src.
        ep_gaps = np.zeros(T, dtype=np.float32)
        for t in range(T - 1):
            s_tgt, _, _ = counterfactual_step(env, sim_state[t], action[t])
            s_src = state[t + 1]
            ep_gaps[t] = state_gap(s_src[None], s_tgt[None])[0]
            n_trans += 1
        # The final entry (i=T-1) has no successor transition; carry the last
        # computed gap forward so every RobomimicDataset entry gets a weight.
        if T >= 2:
            ep_gaps[T - 1] = ep_gaps[T - 2]
        dist = needle_ring_dist(state)  # (T,) contact-phase proxy
        for t in range(T):
            EP.append(ei); TS.append(t); GAP.append(ep_gaps[t]); DIST.append(dist[t])
        if (ei + 1) % 20 == 0 or ei == len(eps) - 1:
            rate = n_trans / max(time.time() - t0, 1e-6)
            print(f"  ep {ei+1:4d}/{len(eps)}  ({n_trans} trans, {rate:.0f} trans/s)")

    EP = np.asarray(EP, np.int64)
    TS = np.asarray(TS, np.int64)
    GAP = np.asarray(GAP, np.float32)
    DIST = np.asarray(DIST, np.float32)

    contact = DIST < 0.15
    print(f"\ntotal entries: {len(GAP)}  ({n_trans} real transitions)")
    print(f"gap: mean {GAP.mean():.4f} median {np.median(GAP):.4f} "
          f"p90 {np.percentile(GAP,90):.4f} max {GAP.max():.4f}")
    print(f"  contact (dist<0.15): mean {GAP[contact].mean():.4f} "
          f"(n={contact.sum()})")
    print(f"  free-space        : mean {GAP[~contact].mean():.4f} "
          f"(n={(~contact).sum()})")
    for tau in (0.024, 0.05, 0.1):
        c = np.exp(-GAP / tau)
        print(f"  tau={tau:.3f}: mean c={c.mean():.3f}  "
              f"contact c={np.exp(-GAP[contact]/tau).mean():.3f}  "
              f"free c={np.exp(-GAP[~contact]/tau).mean():.3f}")

    out = args.out or os.path.join(
        os.path.dirname(args.source_hdf5), f"gt_gaps_n{args.num_demos}.npz")
    np.savez(out, ep_id=EP, t_in_ep=TS, gap=GAP, dist=DIST, num_demos=args.num_demos)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
