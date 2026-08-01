"""Make a transition-count-matched subset of one OR MORE recovery datasets.

For the recovery-transfer experiment (CLAUDE.md §19) we compare weighted-BC /
weighted-ReBRAC trained on the recovery datasets against the ds4 baseline
(50,554 transitions). To control for DATA VOLUME (BC/RL sample per-transition),
each training set must use the SAME transition budget.

We can't just take leading demos: ds3 was written A-block-then-B-block, so a
leading prefix collapses its fail fraction. Instead SHUFFLE all demos (across
ALL --src dirs, for DAgger aggregation) with a fixed seed and take them in
shuffled order until the cumulative transition count reaches the budget.
Shuffling preserves the per-type transition mix.

Writes a fresh HDF5 with re-indexed demo_0.. (so a gap cache computed on it lines
up by (ep_id, t)) + the env_cfg.json sidecar (composition updated).

Usage:
    # single source
    python scripts/make_tmatch_subset.py --src <dir> --budget 50554 --seed 0
    # DAgger aggregation: pack across round-0..i raw datasets
    python scripts/make_tmatch_subset.py --src <dir0> <dir1> ... --out <dir> --budget 50554
"""

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True,
                   help="one or more dataset dirs (each with dataset.hdf5 + env_cfg.json); "
                        "demos are pooled across all of them.")
    p.add_argument("--out", default=None, help="output dir (default: <src[0]>_tmatch)")
    p.add_argument("--budget", type=int, default=50554)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = args.out or (args.src[0].rstrip("/") + "_tmatch")
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, "dataset.hdf5")

    # gather (src_idx, demo_idx, len) across all sources
    pool = []
    handles = []
    for si, sdir in enumerate(args.src):
        f = h5py.File(os.path.join(sdir, "dataset.hdf5"), "r")
        handles.append(f)
        d = f["data"]
        for i in range(len(d.keys())):
            pool.append((si, i, int(d[f"demo_{i}"].attrs["num_samples"])))

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(pool))
    cum = 0
    chosen = []
    for oi in order:
        chosen.append(pool[oi])
        cum += pool[oi][2]
        if cum >= args.budget:
            break

    types = defaultdict(int)
    src_counts = defaultdict(int)
    with h5py.File(out_h5, "w") as hf:
        g_out = hf.create_group("data")
        for new_i, (si, old_i, _ln) in enumerate(chosen):
            g_src = handles[si][f"data/demo_{old_i}"]
            g_dst = g_out.create_group(f"demo_{new_i}")
            for k in g_src["obs"].keys():
                g_dst.create_dataset(f"obs/{k}", data=g_src[f"obs/{k}"][:])
            g_dst.create_dataset("actions", data=g_src["actions"][:])
            g_dst.create_dataset("rewards", data=g_src["rewards"][:])
            for ak, av in g_src.attrs.items():
                g_dst.attrs[ak] = av
            types[str(g_src.attrs.get("ep_type", "?"))] += 1
            src_counts[args.src[si]] += 1
        g_out.attrs["num_demos"] = len(chosen)
    for f in handles:
        f.close()

    cfg = json.load(open(os.path.join(args.src[0], "env_cfg.json")))
    cfg["tmatch"] = {
        "sources": args.src, "budget": args.budget, "seed": args.seed,
        "num_demos": len(chosen), "num_transitions": int(cum),
        "composition": dict(types), "per_source": dict(src_counts),
    }
    json.dump(cfg, open(os.path.join(out_dir, "env_cfg.json"), "w"), indent=2)

    print(f"{args.src} -> {out_dir}")
    print(f"  {len(chosen)} demos, {cum} transitions (budget {args.budget}), "
          f"types={dict(types)}, per_source={dict(src_counts)}")


if __name__ == "__main__":
    main()
