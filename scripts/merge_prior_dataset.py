"""Merge a SOURCE rollout set + TARGET demos into one preload dataset (for RLPD).

`train_rl.py` has a single `--preload_datapath`, but RLPD's prior buffer is meant
to hold BOTH the large offline set and the few in-domain demos. This concatenates
them into one HDF5 in the layout `rl.replay.add_demos_to_replay` expects.

Two format fixes happen here:

1. **Per-key obs.** `add_demos_to_replay` reads `STATE_KEYS[task]` (and `prop`)
   as separate datasets, but `generate_simdist_data.py` writes only a packed
   `obs/state`. The packing is verified exact for Threading_D0:
       state = eef_pos[0:3] | eef_quat[3:7] | gripper_qpos[7:9] | object[9:37]
       prop  = state[0:9]
   so the split is lossless (asserted at runtime against any file that already
   carries both).
2. **Sidecar.** `add_demos_to_replay` reads `env_cfg.json` next to the HDF5 for
   `env_name`; it is copied from the TARGET demo dir (the deployment domain,
   which is what the online phase runs in).

Usage:
    python scripts/merge_prior_dataset.py \
        --source release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
        --source_episodes 2000 \
        --target release/data/mimicgen/threading_d0_sat_400000_s1 \
        --out release/data/mimicgen/rlpd_prior_s1
"""
import argparse
import json
import os
import shutil
import sys

import h5py
import numpy as np

# Threading_D0 packing; see the module docstring (verified exact).
SPLIT = [("robot0_eef_pos", 0, 3), ("robot0_eef_quat", 3, 7),
         ("robot0_gripper_qpos", 7, 9), ("object", 9, 37)]


def _write_episode(dst_grp, i, obs: dict, actions, rewards, success):
    g = dst_grp.create_group(f"demo_{i}")
    for k, v in obs.items():
        g.create_dataset(f"obs/{k}", data=v)
    g.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
    g.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
    g.attrs["num_samples"] = len(actions)
    g.attrs["success"] = bool(success)


def _unpack(state: np.ndarray) -> dict:
    out = {k: state[:, a:b].astype(np.float32) for k, a, b in SPLIT}
    out["prop"] = state[:, 0:9].astype(np.float32)
    out["state"] = state.astype(np.float32)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="source rollout hdf5")
    p.add_argument("--source_episodes", type=int, default=2000)
    p.add_argument("--source_success_only", type=int, default=0)
    p.add_argument("--target", required=True, help="target demo DIR (needs env_cfg.json)")
    p.add_argument("--target_episodes", type=int, default=20)
    p.add_argument("--out", required=True, help="output DIR")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    n = 0
    with h5py.File(os.path.join(args.out, "dataset.hdf5"), "w") as hf:
        dst = hf.create_group("data")

        # ---- SOURCE (packed obs/state -> per-key) ----
        with h5py.File(args.source, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if args.source_success_only:
                keys = [k for k in keys if bool(f[f"data/{k}"].attrs.get("success", False))]
            if 0 < args.source_episodes < len(keys):
                keys = [keys[i] for i in sorted(
                    rng.choice(len(keys), args.source_episodes, replace=False))]
            for k in keys:
                d = f[f"data/{k}"]
                st = np.asarray(d["obs"]["state"])
                obs = _unpack(st)
                for name, a, b in SPLIT:          # if the file already has them, verify
                    if name in d["obs"]:
                        assert np.allclose(np.asarray(d["obs"][name]), obs[name]), \
                            f"{k}: packed state disagrees with stored {name}"
                _write_episode(dst, n, obs, np.asarray(d["actions"]),
                               np.asarray(d["rewards"]),
                               d.attrs.get("success", np.asarray(d["rewards"]).sum() > 0))
                n += 1
        n_src = n

        # ---- TARGET demos (already per-key) ----
        with h5py.File(os.path.join(args.target, "dataset.hdf5"), "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            keys = keys[:args.target_episodes] if args.target_episodes > 0 else keys
            for k in keys:
                d = f[f"data/{k}"]
                obs = {kk: np.asarray(d["obs"][kk]) for kk in d["obs"].keys()
                       if kk != "sim_state"}
                _write_episode(dst, n, obs, np.asarray(d["actions"]),
                               np.asarray(d["rewards"]), d.attrs.get("success", True))
                n += 1
        dst.attrs["num_demos"] = n

    shutil.copy(os.path.join(args.target, "env_cfg.json"),
                os.path.join(args.out, "env_cfg.json"))
    cfg = json.load(open(os.path.join(args.out, "env_cfg.json")))
    cfg["composition"] = {"num_demos_total": n, "source": args.source,
                          "num_source": n_src, "target": args.target,
                          "num_target": n - n_src}
    json.dump(cfg, open(os.path.join(args.out, "env_cfg.json"), "w"), indent=2)
    print(f"[merge] {n_src} source + {n - n_src} target = {n} episodes -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
