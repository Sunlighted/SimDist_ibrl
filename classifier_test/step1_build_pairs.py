"""Step 1: build counterfactual (s, a, s'_src, s'_tgt) pairs from target demos.

For every target-demo transition (sim_state_t, a_t), reset the SOURCE (nominal)
sim to sim_state_t, apply a_t, read s'_src. s'_tgt is the demo's own next state
(recorded under target dynamics). Both start from the SAME physical state, so
||s'_tgt - s'_src|| is the ground-truth per-transition dynamics gap.
"""

import argparse
import os

import numpy as np

# cf_utils sets sys.path/cwd/MUJOCO_GL on import, so import it first.
from cf_utils import REPO, get_source_env, counterfactual_step, load_demo_episodes
from data_utils import needle_ring_dist, state_gap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_hdf5", default=f"{REPO}/release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5")
    p.add_argument("--source_weight", default=f"{REPO}/exps/mimicgen/threading_ibrl_source_seed1/model0.pt")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--out", default=f"{os.path.dirname(__file__)}/pairs_demo.npz")
    args = p.parse_args()

    eps = load_demo_episodes(args.target_hdf5, args.max_episodes)
    print(f"loaded {len(eps)} target demo episodes")
    env = get_source_env(args.source_weight)

    S, A, S_SRC, S_TGT, EP, TS = [], [], [], [], [], []
    for ei, ep in enumerate(eps):
        T = ep["state"].shape[0] - 1
        for t in range(T):
            s_src, _, _ = counterfactual_step(env, ep["sim_state"][t], ep["action"][t])
            S.append(ep["state"][t])
            A.append(ep["action"][t])
            S_SRC.append(s_src)
            S_TGT.append(ep["state"][t + 1])
            EP.append(ei)
            TS.append(t)
        print(f"  ep {ei:2d}: {T} pairs")

    S, A, S_SRC, S_TGT = map(lambda x: np.asarray(x, np.float32), (S, A, S_SRC, S_TGT))
    EP, TS = np.asarray(EP, np.int64), np.asarray(TS, np.int64)
    gap = state_gap(S_SRC, S_TGT)  # quaternion-aware
    dist = needle_ring_dist(S)
    print(f"total pairs: {len(S)}")
    print(f"gap: mean {gap.mean():.4f} median {np.median(gap):.4f} max {gap.max():.4f}")
    print(f"  contact (dist<0.15) gap mean {gap[dist<0.15].mean():.4f} | "
          f"free-space gap mean {gap[dist>=0.15].mean():.4f}")

    np.savez(args.out, s=S, a=A, s_src=S_SRC, s_tgt=S_TGT, gap=gap,
             dist=dist, ep_id=EP, t_in_ep=TS)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
