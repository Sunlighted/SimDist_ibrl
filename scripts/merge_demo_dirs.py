"""Concatenate several demo dataset dirs into one, re-indexing demo_0..N-1.

Used to assemble a large pool from parallel `collect_target_demos.py` shards:
that script is single-process, so running K instances with DISJOINT --seed
ranges and merging is ~K times faster. The sidecar is taken from the first
shard, with `composition.shards` recording how many demos each contributed and
`demo_*.attrs["src_shard"]` recording per-demo provenance.

Usage:
    python scripts/merge_demo_dirs.py --src poolA poolB poolC --out big_pool
"""
import argparse
import json
import os

import h5py

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "release/data/mimicgen")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = os.path.join(_DATA, args.out)
    os.makedirs(out, exist_ok=True)
    n, shards = 0, []
    with h5py.File(os.path.join(out, "dataset.hdf5"), "w") as hf:
        gg = hf.create_group("data")
        for s in args.src:
            path = os.path.join(_DATA, s, "dataset.hdf5")
            if not os.path.exists(path):
                print(f"  [skip] {s}: no dataset.hdf5")
                continue
            with h5py.File(path, "r") as f:
                keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                for k in keys:
                    gs = f[f"data/{k}"]
                    gd = gg.create_group(f"demo_{n}")
                    for ok in gs["obs"].keys():
                        gd.create_dataset(f"obs/{ok}", data=gs[f"obs/{ok}"][:])
                    for ok in gs.keys():
                        if ok != "obs":
                            gd.create_dataset(ok, data=gs[ok][:])
                    for ak, av in gs.attrs.items():
                        gd.attrs[ak] = av
                    gd.attrs["src_shard"] = s
                    n += 1
                shards.append((s, len(keys)))
        gg.attrs["num_demos"] = n

    cfg = json.load(open(os.path.join(_DATA, shards[0][0], "env_cfg.json")))
    cfg["composition"] = {"num_demos_total": n, "shards": dict(shards)}
    json.dump(cfg, open(os.path.join(out, "env_cfg.json"), "w"), indent=2)
    print(f"merged {n} demos from {len(shards)} shards -> {out}")
    for s, c in shards:
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
