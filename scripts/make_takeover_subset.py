"""Build a TAKEOVER-ONLY recovery subset from a masked recovery dataset.

The recovery datasets (collect_recovery_data.py with the per-step
`takeover_mask`) store WHOLE C/D episodes: a BC-driven approach, one or more
V-gated RL-takeover (recovery) segments, and possibly BC-driven tails after a
hand-back. Only the takeover transitions are the recovery signal we want to
clone (off-manifold state -> source-RL corrective action — the DAgger
correction set). The pre-/post-takeover BC steps are redundant with the success
pool (or are the drift-into-trouble steps we do NOT want to imitate).

This script slices out each MAXIMAL CONTIGUOUS takeover run and writes it as its
own `demo_*` (so runs stay coherent trajectories — chunk-safe for flow/diffusion
and clean transitions for single-step BC). Output keeps the same HDF5 +
`env_cfg.json` sidecar layout, so bc/dataset.py reads it directly (point the
weighted-BC recovery channel at it; no ep_type_filter needed — every demo is a
takeover run, tagged ep_type="C").

Usage:
  python scripts/make_takeover_subset.py \
      --src release/data/mimicgen/threading_d0_source_recovery_mask_ds1 \
      --out release/data/mimicgen/threading_d0_source_recovery_mask_ds1_takeover
  # options:
  #   --ep_types C        # only extract runs from these ep_types (default: all)
  #   --min_run_len 1     # drop takeover runs shorter than this (default 1)
"""
import argparse
import os
import shutil

import h5py
import numpy as np


def contiguous_runs(mask: np.ndarray):
    """Yield (start, end) [end exclusive] for each maximal True run in mask."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="masked recovery dataset dir")
    p.add_argument("--out", required=True, help="output dir for the takeover subset")
    p.add_argument("--ep_types", type=str, default="",
                   help="comma-separated ep_types to extract runs from (empty = all)")
    p.add_argument("--min_run_len", type=int, default=1,
                   help="drop takeover runs shorter than this many steps")
    args = p.parse_args()

    src_h5 = os.path.join(args.src, "dataset.hdf5")
    assert os.path.exists(src_h5), f"not found: {src_h5}"
    allowed = {t for t in args.ep_types.split(",") if t}
    os.makedirs(args.out, exist_ok=True)

    n_runs = 0
    n_trans = 0
    skipped_short = 0
    src_eps_with_takeover = 0
    with h5py.File(src_h5, "r") as fin, \
            h5py.File(os.path.join(args.out, "dataset.hdf5"), "w") as fout:
        gin = fin["data"]
        gout = fout.create_group("data")
        demo_keys = sorted([k for k in gin.keys() if k.startswith("demo_")],
                           key=lambda x: int(x.split("_")[1]))
        for dk in demo_keys:
            d = gin[dk]
            ep_type = d.attrs.get("ep_type", "?")
            if allowed and ep_type not in allowed:
                continue
            if "takeover_mask" not in d:
                raise KeyError(
                    f"{dk} has no takeover_mask — re-collect with the patched "
                    "collect_recovery_data.py (it stores the per-step mask).")
            mask = np.array(d["takeover_mask"]).astype(bool)
            if not mask.any():
                continue
            src_eps_with_takeover += 1
            for (i, j) in contiguous_runs(mask):
                if j - i < args.min_run_len:
                    skipped_short += 1
                    continue
                g = gout.create_group(f"demo_{n_runs}")
                # slice every obs channel + actions/rewards over [i, j)
                og = g.create_group("obs")
                for k in d["obs"].keys():
                    og.create_dataset(k, data=np.array(d["obs"][k])[i:j])
                g.create_dataset("actions", data=np.array(d["actions"])[i:j])
                if "rewards" in d:
                    g.create_dataset("rewards", data=np.array(d["rewards"])[i:j])
                g.attrs["num_samples"] = int(j - i)
                g.attrs["ep_type"] = "C"          # every run is a recovery segment
                g.attrs["took_over"] = True
                g.attrs["n_takeover_steps"] = int(j - i)
                g.attrs["success"] = bool(d.attrs.get("success", True))
                g.attrs["src_demo"] = dk
                g.attrs["src_run"] = [int(i), int(j)]
                n_runs += 1
                n_trans += int(j - i)
        gout.attrs["num_demos"] = n_runs

    # carry over the sidecar so RobomimicDataset detects the collect format.
    sidecar = os.path.join(args.src, "env_cfg.json")
    if os.path.exists(sidecar):
        shutil.copy(sidecar, os.path.join(args.out, "env_cfg.json"))

    print(f"source episodes with takeover: {src_eps_with_takeover}")
    print(f"takeover runs written: {n_runs}  (dropped {skipped_short} runs "
          f"< min_run_len={args.min_run_len})")
    print(f"total takeover transitions: {n_trans}")
    print(f"-> {os.path.join(args.out, 'dataset.hdf5')}")


if __name__ == "__main__":
    main()
