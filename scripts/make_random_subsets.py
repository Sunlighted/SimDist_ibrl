"""Draw K DISJOINT random subsets of n demos from a pool.

Motivation (CLAUDE.md 35): the original 20-demo set adapts to 0.93-0.95 across
3 replicates, while ~8 other 20-demo sets -- matched on every raw-state,
set-level, latent and closed-loop statistic we could measure -- land at
0.36-0.90. Off-manifold policy disagreement between such sets is ~80% of the
action's own range while 100% of stage-3 collection happens off-manifold, so
"which 20 trajectories" may simply be a high-variance draw.

This builds the control that separates the two remaining explanations:
  * if K disjoint random 20-demo draws from ONE pool (one network, one training
    stage) spread widely in adapt, then 20 demos is intrinsically unreliable and
    the original set was a lucky draw;
  * if they cluster tightly, the difference is a property of the collecting
    network, not of the draw.

Usage:
    python scripts/make_random_subsets.py --pool threading_d0_repro952_pool500 \
        --n 20 --k 5 --seed 0 --prefix threading_d0_rand20
"""
import argparse
import json
import os

import h5py
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "release/data/mimicgen")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()

    src_path = os.path.join(_DATA, args.pool, "dataset.hdf5")
    with h5py.File(src_path, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    assert args.k * args.n <= len(keys), f"pool has {len(keys)} demos, need {args.k*args.n}"
    perm = np.random.RandomState(args.seed).permutation(len(keys))

    for g in range(args.k):
        idx = sorted(perm[g * args.n:(g + 1) * args.n].tolist())
        out = os.path.join(_DATA, f"{args.prefix}_g{g}")
        os.makedirs(out, exist_ok=True)
        with h5py.File(src_path, "r") as src, h5py.File(os.path.join(out, "dataset.hdf5"), "w") as hf:
            gg = hf.create_group("data")
            for i, j in enumerate(idx):
                gs = src[f"data/{keys[j]}"]
                gd = gg.create_group(f"demo_{i}")
                for ok in gs["obs"].keys():
                    gd.create_dataset(f"obs/{ok}", data=gs[f"obs/{ok}"][:])
                for ok in gs.keys():
                    if ok != "obs":
                        gd.create_dataset(ok, data=gs[ok][:])
                for ak, av in gs.attrs.items():
                    gd.attrs[ak] = av
                gd.attrs["src_demo"] = keys[j]
            gg.attrs["num_demos"] = len(idx)
        cfg = json.load(open(os.path.join(_DATA, args.pool, "env_cfg.json")))
        cfg["composition"] = {"num_demos_total": len(idx), "selected_from": args.pool,
                              "selection": f"DISJOINT random draw {g+1}/{args.k}, seed={args.seed}",
                              "src_demos": [keys[j] for j in idx]}
        json.dump(cfg, open(os.path.join(out, "env_cfg.json"), "w"), indent=2)
        # quick profile so the draws can be compared without re-reading
        with h5py.File(os.path.join(out, "dataset.hdf5"), "r") as f:
            L = np.array([f[f"data/demo_{i}"].attrs["num_samples"] for i in range(len(idx))])
        print(f"  {args.prefix}_g{g}: n={len(idx)}  len {L.mean():5.1f}±{L.std():4.1f} "
              f"[{L.min()},{L.max()}]  Σ={L.sum()}")


if __name__ == "__main__":
    main()
