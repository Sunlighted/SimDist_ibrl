"""Extract a compact bc_anchor state pool from a huge source-diverse dataset.

`AdaptWorkspace._build_bc_anchor` opens `bc_anchor_path`, draws
`bc_anchor_episodes` episodes at random, concatenates their `obs/state`, and
subsamples to `bc_anchor_states`. Nothing else in the file is ever read -- not
actions, not rewards, not values. But the source sets are 5.5 GB (Square) and
7.7 GB (Coffee), which is a painful thing to ship to a cluster for the sake of
40000 states.

This writes a file with the same layout (`data/demo_<i>/obs/state`, keys sortable
by the integer after the underscore, `num_demos` attr) holding only the states of
a random subset of episodes. Square: 4000 episodes ~= 62 MB. Coffee ~= 266 MB.

Keep the extracted count ABOVE `bc_anchor_episodes` (default 2000) so each seed
still draws its own subset -- `_build_bc_anchor` seeds its RNG with `cfg.seed`,
so an exactly-2000 file would hand every seed the same episodes (it would still
subsample 40000 states per seed, so the effect is small, but there is no reason
to give it up for a few tens of MB).

Usage:
    python scripts/extract_anchor_pool.py \
        release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5 \
        release/data/mimicgen/square_anchor_pool_4k/dataset.hdf5 --episodes 4000
"""
import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    # _build_bc_anchor reads only obs/state by default, but three knobs need more:
    #   bc_anchor_data_actions=1  -> `actions` (the regression target)
    #   bc_anchor_expert_only=1   -> `expert_flags` (asserts the key exists)
    #   bc_anchor_success_only=1  -> the per-episode `success` attribute
    # Off by default so the existing state-only extracts stay byte-identical.
    ap.add_argument("--with_actions", type=int, default=0)
    ap.add_argument("--with_expert_flags", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    rng = np.random.RandomState(args.seed)

    with h5py.File(args.src, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        n = min(args.episodes, len(keys))
        pick = sorted(rng.choice(len(keys), n, replace=False))
        print(f"[extract] {len(keys)} episodes in source -> keeping {n}")
        with h5py.File(args.dst, "w") as g:
            d = g.create_group("data")
            total = 0
            for j, i in enumerate(pick):
                src_ep = f[f"data/{keys[i]}"]
                st = np.asarray(src_ep["obs"]["state"], dtype=np.float32)
                # Renumber to demo_0..demo_{n-1}: the loader only needs the keys to
                # sort by their integer suffix, and dense numbering keeps it obvious
                # that this file is a derived subset, not the original indices.
                g_ep = d.create_group(f"demo_{j}")
                g_ep.create_group("obs").create_dataset("state", data=st,
                                                        compression="gzip",
                                                        compression_opts=4)
                if args.with_actions:
                    g_ep.create_dataset(
                        "actions",
                        data=np.asarray(src_ep["actions"], dtype=np.float32),
                        compression="gzip", compression_opts=4)
                if args.with_expert_flags:
                    assert "expert_flags" in src_ep, \
                        f"{args.src} has no expert_flags -- cannot build an " \
                        f"expert-subset pool from it"
                    g_ep.create_dataset(
                        "expert_flags", data=np.asarray(src_ep["expert_flags"]),
                        compression="gzip", compression_opts=4)
                # success is an ATTRIBUTE, and bc_anchor_success_only reads it with
                # a default of False -- so it must be carried or every episode is
                # silently dropped.
                g_ep.attrs["success"] = bool(src_ep.attrs.get("success", False))
                g_ep.attrs["num_samples"] = int(len(st))
                total += len(st)
                if (j + 1) % 500 == 0:
                    print(f"  {j + 1}/{n} episodes, {total} states", flush=True)
            d.attrs["num_demos"] = n
            d.attrs["source_file"] = os.path.abspath(args.src)
            d.attrs["extract_seed"] = args.seed
            d.attrs["with_actions"] = int(args.with_actions)
            d.attrs["with_expert_flags"] = int(args.with_expert_flags)
    size = os.path.getsize(args.dst) / 1e6
    print(f"[extract] wrote {args.dst}: {n} episodes, {total} states, {size:.0f} MB")


if __name__ == "__main__":
    main()
