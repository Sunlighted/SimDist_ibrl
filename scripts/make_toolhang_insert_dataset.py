"""Truncate robomimic ToolHang PH demos at the stage-1 ("frame inserted") success.

The robomimic `tool_hang/ph` demos solve the full two-stage task. `ToolHangInsert`
(env/tool_hang_variants.py) redefines success as stage 1 only, so the demos have
to be cut at the first state where the NEW success predicate fires -- otherwise
BC would be trained on ~50% of steps that belong to the (now irrelevant) second
stage, and the RL replay would contain post-success transitions.

Method: replay each demo's recorded sim states in the ToolHangInsert env (no
rendering), evaluate `_check_success()` per state, cut at the first state index
`k` where it holds for `--dwell` consecutive states. Transitions 0..k-1 are kept
(the reward convention in this repo is r(s') AFTER the transition, so transition
k-1 is the one that earns reward 1 and gets done=1).

Observations are copied verbatim from the source file: `ToolHangInsert` differs
from `ToolHang` only in `_check_success`, so every obs/state/action is identical.

Usage:
    python scripts/make_toolhang_insert_dataset.py \
        --dataset release/data/robomimic/tool_hang/ph/low_dim_v141.hdf5 \
        --output  release/data/robomimic/tool_hang_insert/ph/insert_low_dim.hdf5
"""

import argparse
import hashlib
import json
import os

import h5py
import numpy as np

import env.tool_hang_variants  # noqa: F401  # registers ToolHangInsert
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def first_success_index(env, raw, states, model_file, dwell):
    """Replay `states` and return (k, succ) where succ[t] = success at state t.

    k = first index such that succ[k:k+dwell] are all True, else -1.
    """
    env.reset()
    env.reset_to({"states": states[0], "model": model_file})
    succ = np.zeros(len(states), dtype=bool)
    for t in range(len(states)):
        env.reset_to({"states": states[t]})
        succ[t] = raw._check_success()
    k = -1
    for t in range(len(states) - dwell + 1):
        if succ[t : t + dwell].all():
            k = t
            break
    return k, succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--env_name", default="ToolHangInsert")
    ap.add_argument("--n", type=int, default=None, help="only process the first n demos")
    ap.add_argument(
        "--dwell",
        type=int,
        default=5,
        help="require this many consecutive success states before cutting "
        "(guards against a one-frame contact fluke)",
    )
    ap.add_argument(
        "--pad_after",
        type=int,
        default=0,
        help="extra transitions to keep past the success state (0 = cut exactly at success)",
    )
    args = ap.parse_args()

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    src_env_name = env_meta["env_name"]
    env_meta["env_name"] = args.env_name
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    raw = env.env  # underlying robosuite env

    f = h5py.File(args.dataset, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))
    if args.n is not None:
        demos = demos[: args.n]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    f_out = h5py.File(args.output, "w")
    grp = f_out.create_group("data")

    total, kept, dropped = 0, [], []
    src_lens, new_lens = [], []
    for i, ep in enumerate(demos):
        src = f["data/{}".format(ep)]
        states = src["states"][()]
        model_file = src.attrs["model_file"]
        k, succ = first_success_index(env, raw, states, model_file, args.dwell)
        if k <= 0:
            dropped.append((ep, k, int(succ.sum())))
            print(f"[{i}] {ep}: NO stage-1 success (succ steps={int(succ.sum())}) -> dropped")
            continue

        end = min(k + args.pad_after, len(states) - 1)  # transitions 0..end-1
        n = end
        actions = src["actions"][()][:n]
        rewards = np.zeros(n, dtype=np.float64)
        dones = np.zeros(n, dtype=np.int64)
        # r(s') convention: transition n-1 lands in the success state
        rewards[-1] = 1.0
        dones[-1] = 1
        if args.pad_after > 0:
            # every state from k on is a success state; mark them all
            rewards[k - 1 :] = 1.0

        g = grp.create_group(ep)
        g.create_dataset("actions", data=actions)
        g.create_dataset("states", data=states[:n])
        g.create_dataset("rewards", data=rewards)
        g.create_dataset("dones", data=dones)
        for key in src["obs"].keys():
            g.create_dataset(f"obs/{key}", data=src[f"obs/{key}"][()][:n])
        g.attrs["model_file"] = model_file
        g.attrs["num_samples"] = n
        g.attrs["first_success_state"] = k
        total += n
        kept.append(ep)
        src_lens.append(len(states))
        new_lens.append(n)
        print(
            f"[{i}] {ep}: T={len(states)} -> {n} (first success state {k}, "
            f"{100.0 * n / len(states):.0f}% of the demo)"
        )

    # filter keys, restricted to demos we kept
    if "mask" in f:
        mgrp = f_out.create_group("mask")
        keptset = set(kept)
        for mk in f["mask"].keys():
            names = [x.decode() if isinstance(x, bytes) else x for x in f["mask"][mk][()]]
            names = [x for x in names if x in keptset]
            mgrp.create_dataset(mk, data=np.array(names, dtype="S"))

    env_args = json.loads(f["data"].attrs["env_args"])
    env_args["env_name"] = args.env_name
    grp.attrs["env_args"] = json.dumps(env_args, indent=4)
    grp.attrs["total"] = total
    f_out.close()

    # `rl/replay.py:add_demos_to_replay` reads the task name from an
    # `env_cfg.json` SIDECAR next to the data file (not from the hdf5 attrs),
    # so write one or the IBRL preload will KeyError on STATE_KEYS.
    with open(os.path.join(os.path.dirname(os.path.abspath(args.output)), "env_cfg.json"), "w") as jf:
        json.dump(env_args, jf, indent=4)

    src_lens, new_lens = np.array(src_lens), np.array(new_lens)
    meta = dict(
        source_dataset=os.path.abspath(args.dataset),
        source_sha256=sha256(args.dataset),
        source_env_name=src_env_name,
        env_name=args.env_name,
        dwell=args.dwell,
        pad_after=args.pad_after,
        num_demos_in=len(demos),
        num_demos_kept=len(kept),
        dropped=[d[0] for d in dropped],
        total_transitions=int(total),
        len_mean=float(new_lens.mean()),
        len_min=int(new_lens.min()),
        len_max=int(new_lens.max()),
        frac_of_original=float(new_lens.sum() / src_lens.sum()),
    )
    with open(os.path.splitext(args.output)[0] + "_meta.json", "w") as jf:
        json.dump(meta, jf, indent=2)

    print("\n==== summary ====")
    print(json.dumps({k: v for k, v in meta.items() if k != "dropped"}, indent=2))
    print(f"dropped {len(dropped)} demos: {[d[0] for d in dropped]}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
