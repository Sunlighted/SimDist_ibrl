"""Convert a demo/rollout HDF5 into the `episodes` npz that the IQL tooling reads.

`scripts/pretrain_iql_qagent.py` / `scripts/verify_iql_critic.py` consume
    npz["episodes"] = [ {states[T,obs], actions[T,act], rewards[T], success: bool}, ... ]
while everything produced by `collect_target_demos.py` / `generate_simdist_data.py`
is HDF5 (`data/demo_N/{obs/state, actions, rewards}` + a `success` attr). This
bridges the two so an offline-RL baseline can be trained on, e.g.,
`threading_d0_source_diverse_v2` (100k SOURCE trajectories, 63.7% success).

Subsampling is on by default: the full diverse set is 4.3 GB / 100k episodes,
which is both slower than it needs to be for an offline critic and larger than
the npz format wants to hold in RAM. `--num_episodes` draws a uniform random
subset (seeded, so the same call reproduces the same subset).

Usage:
    python scripts/hdf5_to_rollouts_npz.py \
        --hdf5 release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
        --out  release/data/mimicgen/source_diverse_v2_5k.npz --num_episodes 5000
"""
import argparse
import os
import sys

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_episodes", type=int, default=5000,
                   help="uniform random subset; <=0 keeps all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--success_only", type=int, default=0,
                   help="1 = keep only successful episodes")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    with h5py.File(args.hdf5, "r") as f:
        g = f["data"]
        keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))
        if args.success_only:
            keys = [k for k in keys if bool(g[k].attrs.get("success", False))]
        if 0 < args.num_episodes < len(keys):
            keys = [keys[i] for i in sorted(rng.choice(len(keys), args.num_episodes,
                                                       replace=False))]
        eps, n_tr, n_succ = [], 0, 0
        for k in keys:
            d = g[k]
            a = np.asarray(d["actions"], dtype=np.float32)
            s = np.asarray(d["obs"]["state"], dtype=np.float32)
            r = np.asarray(d["rewards"], dtype=np.float32)
            ok = bool(d.attrs.get("success", r.sum() > 0))
            # `build_flat` indexes states[t+1] with a clamp, so states may be
            # either T or T+1 long; both layouts exist in this repo.
            eps.append({"states": s, "actions": a, "rewards": r, "success": ok})
            n_tr += len(a)
            n_succ += int(ok)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, episodes=np.array(eps, dtype=object))
    print(f"[convert] {len(eps)} episodes / {n_tr} transitions / "
          f"{n_succ} success ({n_succ / max(1, len(eps)):.1%}) -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
