"""Generate MimicGen demos for a task on the REAL-CELL geometry.

The env is a `*_RealCell` class (env/threading_variants.py, env/square_variants.py),
which reads the table height and workspace offset from the task's cell YAML in
release/cfgs/sim2real/ -- the same file train_rl.py's `sim2real_cell` uses -- so
demos, BC prior and RL env share one geometry.

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
  python scripts/gen_realcell_demos.py --task square --measure 50
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
import env.square_variants  # noqa: F401,E402  (registers Square_RealCell)

# Per task: MimicGen's config template, the PREPARED source dataset (10 human
# demos carrying `datagen_info` -- run mimicgen's prepare_src_dataset.py once if
# a fresh download lacks it), the registered env class, and the output dir.
TASKS = {
    "threading": dict(template="threading.json", src="threading.hdf5",
                      env="Threading_RealCell", out="threading_realcell"),
    # rotw1 (120-180 deg nut window) is the task definition of this project's
    # Square line, and the class -- not a wrapper knob -- is what carries it,
    # so generation and train_rl.py cannot disagree (env/square_variants.py).
    "square": dict(template="square.json", src="square.hdf5",
                   env="Square_RealCell_RotW1", out="square_realcell_rotw1"),
    "square_full": dict(template="square.json", src="square.hdf5",
                        env="Square_RealCell", out="square_realcell"),
}
TASK = TASKS["threading"]   # set from --task in __main__


def build_config(name: str, num: int, guarantee: bool, seed: int, video: bool = False) -> str:
    template = os.path.join(
        REPO, "deps/mimicgen/mimicgen/exps/templates/robosuite", TASK["template"])
    src = os.path.join(REPO, "release/data/mimicgen/source", TASK["src"])
    out_root = os.path.join(REPO, "release/data/mimicgen/generated", TASK["out"])
    with open(template) as f:
        cfg = json.load(f)
    exp = cfg["experiment"]
    exp["name"] = name
    exp["source"]["dataset_path"] = src
    exp["generation"]["path"] = out_root
    exp["generation"]["guarantee"] = guarantee
    exp["generation"]["num_trials"] = num
    exp["generation"]["keep_failed"] = bool(video)
    exp["task"]["name"] = TASK["env"]
    # With guarantee on, generation aborts after max_num_failures FAILURES. The
    # template's default is sized for tiny runs; at a ~50% success rate a
    # 125-success shard legitimately sees >100 failures, so cap it generously.
    exp["max_num_failures"] = max(int(exp.get("max_num_failures", 50)), 20 * num)
    exp["seed"] = seed
    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = []          # state-only: no rendering
    # --video: keep the failures and play them back, so a broken LAYOUT can be
    # seen (which subtask dies) instead of inferred from a success rate.
    exp["render_video"] = bool(video)
    exp["num_demo_to_render"] = 4 if video else 0
    exp["num_fail_demo_to_render"] = 6 if video else 0
    path = os.path.join(out_root, "_configs", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="threading", choices=sorted(TASKS))
    p.add_argument("--num", type=int, default=1000, help="SUCCESSES to guarantee")
    p.add_argument("--measure", type=int, default=0,
                   help=">0: run this many ATTEMPTS with guarantee off -> success rate")
    p.add_argument("--video", action="store_true",
                   help="keep failures and write playback mp4s (layout debugging)")
    p.add_argument("--shard", type=int, default=0,
                   help="shard index; sets the experiment name and the seed")
    args = p.parse_args()
    TASK = TASKS[args.task]

    if args.measure > 0:
        # shard-tagged so several candidate cells can be measured in parallel
        # (each run needs its own experiment dir under generated/<task>/).
        name = "measure" if args.shard == 0 else f"measure_{args.shard:02d}"
        num, guarantee = args.measure, False
    else:
        name, num, guarantee = f"shard_{args.shard:02d}", args.num, True
    seed = 1000 + args.shard
    cfg_path = build_config(name, num, guarantee, seed, video=args.video)
    print(f"[gen] {name}: {'attempts' if not guarantee else 'successes'}={num} "
          f"seed={seed} config={cfg_path}", flush=True)

    sys.argv = ["generate_dataset", "--config", cfg_path, "--auto-remove-exp"]
    runpy.run_module("mimicgen.scripts.generate_dataset", run_name="__main__")
