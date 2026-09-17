"""Merge sharded MimicGen outputs (<root>/shard_XX/demo.hdf5) into one demo.hdf5.

Demos are renumbered demo_0..demo_{N-1} in shard order. The `data` group attrs
(env_args -- which names the env, e.g. Threading_RealCell -- and total) come
from the first shard, and every other shard's env_args must MATCH it: merging
demos generated against two different geometries would produce a dataset that
converts without error and trains a prior on a mixture of tables.

Usage:
  python scripts/merge_mimicgen_shards.py \
      --root release/data/mimicgen/generated/threading_realcell \
      --out  release/data/mimicgen/generated/threading_realcell/demo.hdf5
"""
import argparse
import glob
import json
import os

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="shard_*/demo.hdf5")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    assert shards, f"no shards match {os.path.join(args.root, args.pattern)}"
    assert os.path.abspath(args.out) not in map(os.path.abspath, shards)

    env_args0 = None
    n_out, total = 0, 0
    tmp = args.out + ".partial"
    with h5py.File(tmp, "w") as fo:
        go = fo.create_group("data")
        for sp in shards:
            with h5py.File(sp, "r") as fi:
                gi = fi["data"]
                ea = gi.attrs["env_args"]
                ea_obj = json.loads(ea)
                if env_args0 is None:
                    env_args0 = ea_obj
                    for k, v in gi.attrs.items():
                        go.attrs[k] = v
                else:
                    assert ea_obj == env_args0, f"{sp}: env_args differ from first shard"
                keys = sorted(gi.keys(), key=lambda k: int(k.split("_")[-1]))
                for k in keys:
                    fi.copy(gi[k], go, name=f"demo_{n_out}")
                    total += int(gi[k].attrs.get("num_samples", 0))
                    n_out += 1
                print(f"  {sp}: {len(keys)} demos")
        go.attrs["total"] = total
        if "mask" in fo:
            del fo["mask"]
    os.replace(tmp, args.out)       # never leave a half-written demo.hdf5 behind
    print(f"merged {n_out} demos ({total} samples) from {len(shards)} shards -> {args.out}")
    print(f"env: {env_args0.get('env_name')}")


if __name__ == "__main__":
    main()
