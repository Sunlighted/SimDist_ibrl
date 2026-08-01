"""Merge several TARGET-domain demo HDF5 pools into one cotraining pool.

Used for the recovery-augmented cotraining experiment (chat 2026-06-13):
combine the existing 20 hand-collected TARGET demos with N freshly-collected
V-gated recovery (type-C) trajectories into a single dataset.hdf5 +
env_cfg.json sidecar that train_weighted_bc's `cotrain_path` channel reads.

All inputs must be the collect_target_demos / collect_target_recovery format
(per-demo obs/<key> + actions + rewards, + an env_cfg.json sidecar). Demos are
renumbered demo_0 .. demo_{M-1} in the order the inputs are given. The output
sidecar is copied from --sidecar_from (default: the first input) so the merged
pool advertises the canonical TARGET shift.

Example
-------
    python scripts/merge_target_demos.py \
        --inputs release/data/mimicgen/threading_d0_target_dynamics/dataset.hdf5 \
                 release/data/mimicgen/threading_d0_target_recovery40/dataset.hdf5 \
        --out release/data/mimicgen/threading_d0_target_dynamics_plus_recovery40
"""

import argparse
import json
import os
import shutil

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="input dataset.hdf5 files, in merge order.")
    p.add_argument("--out", required=True, help="output dir (gets dataset.hdf5 + env_cfg.json).")
    p.add_argument("--sidecar_from", default=None,
                   help="dir or hdf5 whose env_cfg.json to copy (default: first input's dir).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_h5 = os.path.join(args.out, "dataset.hdf5")

    n_out = 0
    max_len = 0
    per_input = []
    with h5py.File(out_h5, "w") as fo:
        data_out = fo.create_group("data")
        for inp in args.inputs:
            assert os.path.exists(inp), f"missing input: {inp}"
            with h5py.File(inp, "r") as fi:
                din = fi["data"]
                keys = sorted(din.keys(), key=lambda k: int(k.split("_")[1]))
                for k in keys:
                    gi = din[k]
                    go = data_out.create_group(f"demo_{n_out}")
                    # copy everything verbatim (obs group, actions, rewards,
                    # takeover_mask if present) + per-demo attrs.
                    for sub in gi.keys():
                        gi.copy(sub, go)
                    for ak, av in gi.attrs.items():
                        go.attrs[ak] = av
                    nl = int(gi["actions"].shape[0])
                    max_len = max(max_len, nl)
                    n_out += 1
                per_input.append((inp, len(keys)))
        data_out.attrs["num_demos"] = n_out

    # sidecar
    src_dir = args.sidecar_from or os.path.dirname(args.inputs[0])
    if src_dir.endswith(".hdf5"):
        src_dir = os.path.dirname(src_dir)
    src_sidecar = os.path.join(src_dir, "env_cfg.json")
    dst_sidecar = os.path.join(args.out, "env_cfg.json")
    if os.path.exists(src_sidecar):
        shutil.copy(src_sidecar, dst_sidecar)
        with open(dst_sidecar) as f:
            meta = json.load(f)
        meta["merged_from"] = {os.path.dirname(i): n for i, n in per_input}
        meta["num_demos"] = n_out
        with open(dst_sidecar, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        print(f"[WARN] no env_cfg.json at {src_dir}; merged pool has NO sidecar.")

    print("merged:")
    for inp, n in per_input:
        print(f"  {n:3d} demos  <- {inp}")
    print(f"  ----")
    print(f"  {n_out:3d} demos  -> {out_h5}")
    print(f"  longest episode = {max_len} steps  "
          f"(set --dataset.eval_episode_len > {max_len} when training)")


if __name__ == "__main__":
    main()
