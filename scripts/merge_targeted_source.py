"""Merge scripts/collect_source_targeted.py shards into one loadable dataset.

TWO things happen here beyond concatenation, both deliberate:

1. expert_flags are ZEROED. In tpa_d0_source_diverse_v1 `expert_flag==1` means
   "step came from the SOURCE EXPERT and carried no injected noise" (measured
   16% of steps). In a targeted top-up every rollout comes from the ADAPTED
   SimDist policy, so its clean steps would be flagged at 75% -- and stage-1 runs
   with `bc_expert_only=1`, i.e. the policy head clones exactly the flagged steps.
   Merging as-is would therefore redefine pi_src into a blend of the source expert
   and the adapted policy, which in turn moves the stage-2 anchor target and the
   distilled V = V^{pi_src} (docs/simdist.md 44). The top-up exists to fix the
   DYNAMICS; zeroing the flags keeps it out of the policy-BC mask while the
   dynamics/value/reward heads still train on every frame
   (rl/sequence_replay.py:158). The raw flags stay in the shard files, so an arm
   that deliberately re-enables them is still one merge away.

2. Segments shorter than 6 steps are dropped (H=5 needs horizon+1) -- the
   collector already enforces this, so the count should be 0; it is re-checked
   here because a silent length-5 episode would fail the loader's can_sample.

Usage:
    python scripts/merge_targeted_source.py \
        --in_dir release/data/mimicgen/tpa_d0_source_targeted_v1 \
        --out release/data/mimicgen/tpa_d0_source_targeted_v1/dataset.hdf5
"""
import argparse
import glob
import os

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--zero_expert_flags", type=int, default=1)
    a = p.parse_args()

    shards = sorted(glob.glob(os.path.join(a.in_dir, "**", "_shard*.hdf5"),
                              recursive=True))
    assert shards, f"no _shard*.hdf5 under {a.in_dir}"
    print(f"merging {len(shards)} shards -> {a.out}")

    n = n_succ = n_skip = 0
    tot_steps = 0
    flag_before = []
    with h5py.File(a.out, "w") as fo:
        go = fo.create_group("data")
        for sp in shards:
            with h5py.File(sp, "r") as fi:
                keys = sorted(fi["data"].keys(), key=lambda k: int(k.split("_")[1]))
                for k in keys:
                    gi = fi["data"][k]
                    T = gi["actions"].shape[0]
                    if T < 6:
                        n_skip += 1
                        continue
                    ef = np.asarray(gi["expert_flags"], dtype=np.float32)
                    flag_before.append(ef.mean())
                    d = go.create_group(f"demo_{n}")
                    d.create_dataset("obs/state", data=np.asarray(gi["obs"]["state"]),
                                     compression="gzip", compression_opts=1)
                    d.create_dataset("actions", data=np.asarray(gi["actions"]),
                                     compression="gzip", compression_opts=1)
                    d.create_dataset("rewards", data=np.asarray(gi["rewards"]))
                    d.create_dataset(
                        "expert_flags",
                        data=(np.zeros_like(ef) if a.zero_expert_flags else ef))
                    succ = bool(gi.attrs.get("success", False))
                    d.attrs["success"] = succ
                    n += 1
                    n_succ += int(succ)
                    tot_steps += T
            print(f"  {os.path.relpath(sp, a.in_dir)} -> running n={n}", flush=True)
        go.attrs["num_demos"] = n
        go.attrs["num_success"] = n_succ

    print(f"\ndone: {n} episodes / {tot_steps} transitions / {n_succ} success "
          f"({n_succ / max(n, 1):.3f})")
    print(f"  skipped (T<6): {n_skip}")
    print(f"  expert_flags: mean {np.mean(flag_before):.3f} in the shards -> "
          f"{'0.000 (zeroed)' if a.zero_expert_flags else 'kept'} in the merge")
    print(f"  size: {os.path.getsize(a.out) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
