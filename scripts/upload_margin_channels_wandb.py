"""Replay the margin-channel seed-1 runs into W&B from their local log.pkl.

These four runs were launched by cluster/job_threading_margin_channels.sh with
`--use_wb 0`, so no W&B run was ever created and there is nothing to
`wandb sync` -- the only record is <save_dir>/log.pkl. This script creates one
fresh run per directory and replays the history into it.

Every uploaded run carries `replayed_from_logpkl: true` in its config, plus the
source path and the log's mtime, so a post-hoc upload can never be mistaken for
a live-logged one.

Project/group/name follow the repo convention (save_dir's parent is the project,
the directory name minus the seed suffix is the group), so these land beside the
live runs in cao1233/simdist_threading_3arm.

    python scripts/upload_margin_channels_wandb.py --dry_run   # show, upload nothing
    python scripts/upload_margin_channels_wandb.py
"""
import argparse
import datetime
import os
import pickle

import yaml

EXP = "exps/simdist_threading_3arm"
RUNS = [
    "dynaibrl_m001_seed1",
    "dynaibrl_m001dataonly_seed1",
    "dynaibrl_m001bootonly_seed1",
    "dynaibrl_m0demo_seed1",
]
PROJECT = os.path.basename(EXP)
ENTITY = "cao1233"


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "_"))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    for name in RUNS:
        d = os.path.join(EXP, name)
        log_path, cfg_path = os.path.join(d, "log.pkl"), os.path.join(d, "cfg.yaml")
        if not os.path.exists(log_path):
            raise SystemExit(f"missing {log_path}")
        with open(log_path, "rb") as f:
            hist = pickle.load(f)
        cfg = flatten(yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {})
        cfg.update({
            "replayed_from_logpkl": True,
            "replay_source": d,
            "replay_logpkl_mtime": datetime.datetime.utcfromtimestamp(
                os.path.getmtime(log_path)).isoformat() + "Z",
            "use_wb_original": 0,
        })
        group = name.rsplit("_seed", 1)[0]
        n_score = sum(1 for e in hist if e.get("score/score") is not None)
        print(f"{name:32s} -> {args.entity}/{args.project} group={group:26s} "
              f"rows={len(hist)} evals={n_score}")
        if args.dry_run:
            continue

        import wandb
        run = wandb.init(entity=args.entity, project=args.project, group=group,
                         name=name, config=cfg, mode="online", reinit=True)
        # Log by row index so `_step` matches the live runs (0..100), and let
        # adapt/transitions ride along as a metric exactly as they do.
        for i, row in enumerate(hist):
            clean = {k: v for k, v in row.items() if v is not None}
            if clean:
                wandb.log(clean, step=i)
        run.finish()
        print(f"{'':32s}    uploaded: {run.url}")


if __name__ == "__main__":
    main()
