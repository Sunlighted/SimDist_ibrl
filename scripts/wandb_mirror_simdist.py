"""Mirror a RUNNING train_simdist job's metrics to wandb, from its train.log.

Why a side-car instead of `--use_wb 1`: the run is already going. Restarting it
to turn wandb on would resume from the last resume.pt (written every
log_per_step), throwing away every step since -- up to two hours of a 30-hour
job -- for a dashboard.

It also keeps the known failure mode out of the trainer: online wandb has
deadlocked long runs here (memory: gotcha-wandb-online-deadlock, the thread
parks in __skb_wait_for_more_packets and the whole process freezes). In a
separate process a hang costs the dashboard, not the training.

Parses the two line shapes train_simdist prints per log point:
    <step>: <key>            : <float>
    <step>: <key>   [<n>]: avg: <float>, min: ..., max: ...
and backfills everything already in the file before following new points.

  python scripts/wandb_mirror_simdist.py \
      --save_dir exps/mimicgen/threading_realcell_pixel_stage1_seed1 \
      --project realcell_pixel_stage1 --name base
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINE = re.compile(r"^(\d+):\s+(\S+)\s*(?:\[\s*\d+\s*\])?:\s+(.*)$")
AVG = re.compile(r"avg:\s*(-?[\d.eE+]+)")
NUM = re.compile(r"^(-?[\d.eE+]+)$")


def parse(path):
    """{step: {key: value}} for every log point currently in the file."""
    out = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            step, key, rest = int(m.group(1)), m.group(2), m.group(3).strip()
            a = AVG.search(rest)
            n = NUM.match(rest)
            if a:
                val = float(a.group(1))
            elif n:
                try:
                    val = float(n.group(1))
                except ValueError:
                    continue
            else:
                continue
            out.setdefault(step, {})[key] = val
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--project", default="realcell_pixel_stage1")
    p.add_argument("--name", required=True)
    p.add_argument("--poll", type=int, default=300)
    p.add_argument("--until_step", type=int, default=0,
                   help="exit once this step has been mirrored (0 = run forever)")
    p.add_argument("--idle_exit_min", type=int, default=0,
                   help="exit after this many minutes with no new log point (0 = off)")
    args = p.parse_args()

    log = os.path.join(args.save_dir, "train.log")
    assert os.path.exists(log), log
    import wandb

    run = wandb.init(project=args.project, name=args.name,
                     id=f"{args.project}-{args.name}", resume="allow",
                     config={"save_dir": args.save_dir, "mirrored_from": log})
    print(f"[mirror] {args.name}: {run.url}", flush=True)

    sent = set()
    last_new = time.time()
    while True:
        points = parse(log)
        for step in sorted(points):
            if step in sent:
                continue
            wandb.log(points[step], step=step)
            sent.add(step)
            last_new = time.time()
            print(f"[mirror] {args.name}: step {step} "
                  f"({len(points[step])} metrics)", flush=True)
        if args.until_step and max(sent, default=-1) >= args.until_step:
            print(f"[mirror] {args.name}: reached {args.until_step}, done", flush=True)
            break
        if args.idle_exit_min and (time.time() - last_new) > args.idle_exit_min * 60:
            print(f"[mirror] {args.name}: idle {args.idle_exit_min} min, done", flush=True)
            break
        time.sleep(args.poll)
    run.finish()


if __name__ == "__main__":
    main()
