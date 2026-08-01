"""Generate MimicGen demos on the Threading_Enlarged reset distribution.

Wraps mimicgen's generate_dataset.py so that our custom env (env/threading_variants.py
:: Threading_Enlarged, registered on import) is available in-process, and builds
the generation config from the stock threading.json template.

Pipeline context (recovery training):
  1. download source human demos  (release/data/mimicgen/source/threading.hdf5)
  2. prepare_src_dataset.py        (adds subtask signals)            [done]
  3. THIS SCRIPT -> generate N demos on Threading_Enlarged           (slow)
  4. tools/dataset_states_to_obs.py -> state format
  5. train_bc.py -> a BC prior that covers the enlarged needle distribution

Usage:
  python scripts/gen_enlarged_demos.py                 # 1000 demos (guarantee)
  python scripts/gen_enlarged_demos.py --num 400
  python scripts/gen_enlarged_demos.py --debug         # tiny run to validate

NOTE: MimicGen replays object-centric source segments from the env's DEFAULT
robot init (no hand_init perturbation) — by design here ("先不加 hand initial").
It randomizes the NEEDLE/tripod placement (the enlarged box), which is the
coverage gap we want to fill; arm-init variation is left to RL exploration.
"""

import argparse
import json
import os
import runpy
import sys

import mimicgen  # noqa: F401  (registers D0/D1/D2 + env interfaces)
import env.threading_variants  # noqa: F401  (registers Threading_Enlarged)

REPO = "/home/Chenyang/ibrl"
TEMPLATE = os.path.join(REPO, "deps/mimicgen/mimicgen/exps/templates/robosuite/threading.json")
SRC = os.path.join(REPO, "release/data/mimicgen/source/threading.hdf5")
OUT_DIR = os.path.join(REPO, "release/data/mimicgen/generated/threading_enlarged")
CONFIG_OUT = os.path.join(REPO, "release/cfgs/mimicgen/gen_threading_enlarged.json")


def build_config(num_traj: int, guarantee: bool = True) -> str:
    with open(TEMPLATE) as f:
        cfg = json.load(f)
    exp = cfg["experiment"]
    exp["name"] = "threading_enlarged"
    exp["source"]["dataset_path"] = SRC
    exp["generation"]["path"] = OUT_DIR
    exp["generation"]["guarantee"] = guarantee
    exp["generation"]["num_trials"] = num_traj
    exp["generation"]["keep_failed"] = False
    exp["task"]["name"] = "Threading_Enlarged"
    # interface / robot / gripper left null -> inherited from the prepared source
    # (MG_Threading + Panda). State-only: no camera rendering (much faster).
    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = []
    exp["render_video"] = False
    os.makedirs(os.path.dirname(CONFIG_OUT), exist_ok=True)
    with open(CONFIG_OUT, "w") as f:
        json.dump(cfg, f, indent=4)
    return CONFIG_OUT


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=1000, help="number of SUCCESSES to guarantee")
    p.add_argument("--debug", action="store_true", help="tiny validation run")
    p.add_argument("--measure", type=int, default=0,
                   help="if >0, run this many ATTEMPTS (guarantee off) to estimate success rate")
    args = p.parse_args()

    if args.measure > 0:
        cfg_path = build_config(args.measure, guarantee=False)
        print(f"[gen] MEASURE mode: {args.measure} attempts (guarantee off) -> success rate")
    else:
        cfg_path = build_config(args.num)
        print(f"[gen] wrote config -> {cfg_path}  (Threading_Enlarged, guarantee {args.num})")

    argv = ["generate_dataset", "--config", cfg_path, "--auto-remove-exp"]
    if args.debug:
        argv.append("--debug")
    sys.argv = argv
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")
