"""Replay finished runs' train.log into wandb, without re-running anything.

Use when wandb.init was broken during the run (`USE_WB=0`) or when offline runs
could not be synced. Every metric a run logs also lands in train.log as
    <step>: <key>   : <value>
so the curves are fully recoverable from disk. cfg.yaml goes in as the run
config, and project/run/group follow the same save_dir convention wandb would
have used (common_utils/py/run_cfg.py derives them as read-only properties):
    exps/<project>/<run>   and group = run with any "seed" token removed.

Run this on a LOGIN node (compute nodes have no internet), after `wandb login`.

Usage:
    python scripts/wandb_import_logs.py 'exps/square_rotw1_baselines/*'
    python scripts/wandb_import_logs.py 'exps/.../*' --dry_run 1
"""
import argparse
import glob
import os
import re
import sys

import wandb
import yaml

LINE = re.compile(r"^\s*(\d+):\s+([A-Za-z0-9_/()\[\]. -]+?)\s*:\s*(-?[0-9.eE+-]+)\s*$")


def parse(log_path):
    """-> {step: {key: value}}, in file order (steps may repeat across keys)."""
    steps = {}
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            step, key, val = int(m.group(1)), m.group(2).strip(), m.group(3)
            # `avg: ... min: ...` summary rows share the "[N]" suffix; skip them,
            # they are per-update stats, not the metric series.
            if key.endswith("]") or "[" in key:
                continue
            try:
                steps.setdefault(step, {})[key] = float(val)
            except ValueError:
                pass
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob of run dirs, e.g. 'exps/proj/*'")
    ap.add_argument("--dry_run", type=int, default=0)
    ap.add_argument("--rename", default="",
                    help="comma-separated old=new metric renames, e.g. "
                         "'score/score=score/score_mppi'")
    ap.add_argument("--rename_if", default="",
                    help="only rename in runs whose NAME contains this substring")
    ap.add_argument("--min_steps", type=int, default=0,
                    help="skip runs with fewer logged steps than this (killed runs)")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no run dirs matched {args.pattern!r}")

    for d in dirs:
        log = os.path.join(d, "train.log")
        if not os.path.exists(log):
            print(f"[skip] {d}: no train.log", file=sys.stderr)
            continue
        steps = parse(log)
        if not steps:
            print(f"[skip] {d}: no metric lines", file=sys.stderr)
            continue

        run_name = os.path.basename(d.rstrip("/"))
        if len(steps) < args.min_steps:
            print(f"[skip] {d}: only {len(steps)} steps < min_steps", file=sys.stderr)
            continue
        # Pre-2026-08-31 `dyna_*` runs logged the MPPI score as `score/score`:
        # train_simdist_dyna.py set use_planner=0 in __init__ but inherited the
        # parent's _eval_mppi, which forces use_planner=1 for the duration of every
        # eval. Importing that under the same key as a post-fix run would put two
        # different quantities on one curve -- they differ by 0.4 at the same point,
        # and at 100k it is 0.000 (policy, 3/3 seeds) against 0.20-0.30 (MPPI).
        if args.rename and (not args.rename_if or args.rename_if in run_name):
            pairs = [kv.split("=", 1) for kv in args.rename.split(",") if "=" in kv]
            for step_d in steps.values():
                for old_k, new_k in pairs:
                    if old_k in step_d:
                        step_d[new_k] = step_d.pop(old_k)
        project = os.path.basename(os.path.dirname(d.rstrip("/")))
        group = "_".join(w for w in run_name.split("_") if "seed" not in w)
        cfg_path = os.path.join(d, "cfg.yaml")
        cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}

        n_keys = len({k for v in steps.values() for k in v})
        print(f"[{'dry' if args.dry_run else 'push'}] {project}/{run_name} "
              f"(group={group}) {len(steps)} steps, {n_keys} keys")
        if args.dry_run:
            continue

        run = wandb.init(project=project, name=run_name, group=group, config=cfg,
                         reinit=True)
        for step in sorted(steps):
            run.log(steps[step], step=step)
        run.finish()


if __name__ == "__main__":
    main()
