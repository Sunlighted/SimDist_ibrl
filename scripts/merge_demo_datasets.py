"""Merge several collect_target_demos.py output dirs into one dataset.

Concatenates all demo_* groups (renumbered demo_0..demo_{N-1}) from each
`--parts` dir's dataset.hdf5 into `--out/dataset.hdf5`, and copies the
env_cfg.json sidecar from the first part. Used to stitch parallel collection
shards back into a single dataset.
"""

import argparse
import os
import shutil

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", required=True, help="part dirs (each has dataset.hdf5)")
    p.add_argument("--out", required=True, help="output dir")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_h5 = os.path.join(args.out, "dataset.hdf5")
    n = 0
    with h5py.File(out_h5, "w") as hf:
        out_data = hf.create_group("data")
        for part in args.parts:
            src = os.path.join(part, "dataset.hdf5")
            if not os.path.exists(src):
                print(f"  skip (missing): {src}")
                continue
            with h5py.File(src, "r") as sf:
                keys = sorted(sf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                for k in keys:
                    sf.copy(sf[f"data/{k}"], out_data, name=f"demo_{n}")
                    n += 1
                print(f"  {part}: +{len(keys)} demos (total {n})")
        out_data.attrs["num_demos"] = n

    # env_cfg.json from the first part that has one
    for part in args.parts:
        cfg = os.path.join(part, "env_cfg.json")
        if os.path.exists(cfg):
            shutil.copy(cfg, os.path.join(args.out, "env_cfg.json"))
            break
    print(f"merged {n} demos -> {out_h5}")


if __name__ == "__main__":
    main()
