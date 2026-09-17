"""Generate MimicGen demos for Threading on the REAL-CELL geometry.

The env is `Threading_RealCell` (env/threading_variants.py), which reads the
table height and workspace offset from release/cfgs/sim2real/threading_real_cell.yaml
-- the same file train_rl.py's `sim2real_cell` uses -- so demos, BC prior and RL
env share one geometry.

Why new demos at all: the existing source expert scores 0.880 on the old
geometry and 0.000 on the real cell (table +132.8 mm and sign-flipped relative
to the base; parts shifted +0.10 m in y), so the old flow-BC prior and the
preloaded core D0 demos are unusable there.

Pipeline:
  1. prepare source (done once):
       prepare_src_dataset.py --dataset release/data/mimicgen/source/threading.hdf5
                              --env_interface MG_Threading --env_interface_type robosuite
  2. THIS SCRIPT (shardable: run N copies with distinct --shard)
  3. scripts/merge_mimicgen_shards.py -> one demo.hdf5
  4. tools/dataset_states_to_obs.py   -> processed state dataset
  5. train_bc.py (flow)  6. train_rl.py with sim2real_cell

Usage:
  python scripts/gen_realcell_demos.py --measure 50            # success rate
  python scripts/gen_realcell_demos.py --num 125 --shard 0     # one shard
"""

import argparse
import json
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mimicgen  # noqa: F401,E402  (registers D0/D1/D2 + env interfaces)
import env.threading_variants  # noqa: F401,E402  (registers Threading_RealCell)

TEMPLATE = os.path.join(REPO, "deps/mimicgen/mimicgen/exps/templates/robosuite/threading.json")
SRC = os.path.join(REPO, "release/data/mimicgen/source/threading.hdf5")
OUT_ROOT = os.path.join(REPO, "release/data/mimicgen/generated/threading_realcell")


def build_config(name: str, num: int, guarantee: bool, seed: int) -> str:
    with open(TEMPLATE) as f:
        cfg = json.load(f)
    exp = cfg["experiment"]
    exp["name"] = name
    exp["source"]["dataset_path"] = SRC
    exp["generation"]["path"] = OUT_ROOT
    exp["generation"]["guarantee"] = guarantee
    exp["generation"]["num_trials"] = num
    exp["generation"]["keep_failed"] = False
    exp["task"]["name"] = "Threading_RealCell"
    # With guarantee on, generation aborts after max_num_failures FAILURES. The
    # template's default is sized for tiny runs; at a ~50% success rate a
    # 125-success shard legitimately sees >100 failures, so cap it generously.
    exp["max_num_failures"] = max(int(exp.get("max_num_failures", 50)), 20 * num)
    exp["seed"] = seed
    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = []          # state-only: no rendering
    exp["render_video"] = False
    path = os.path.join(OUT_ROOT, "_configs", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=1000, help="SUCCESSES to guarantee")
    p.add_argument("--measure", type=int, default=0,
                   help=">0: run this many ATTEMPTS with guarantee off -> success rate")
    p.add_argument("--shard", type=int, default=0,
                   help="shard index; sets the experiment name and the seed")
    args = p.parse_args()

    if args.measure > 0:
        name, num, guarantee = "measure", args.measure, False
    else:
        name, num, guarantee = f"shard_{args.shard:02d}", args.num, True
    seed = 1000 + args.shard
    cfg_path = build_config(name, num, guarantee, seed)
    print(f"[gen] {name}: {'attempts' if not guarantee else 'successes'}={num} "
          f"seed={seed} config={cfg_path}", flush=True)

    sys.argv = ["generate_dataset", "--config", cfg_path, "--auto-remove-exp"]
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")
