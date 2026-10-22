"""Watch the contact health of running IBRL runs on the Robotiq cell (2026-10-22).

Reads the safety/* stats train_rl.py logs every log_per_step (per training episode: table impact, max gripper-table
force, max needle-ring / tripod-table / needle-table / gripper-needle penetration) plus the eval score, prints one line
per new log block, and an ALERT line when a threshold is crossed:
  - needle-ring penetration: block MAX > --pen_max_mm (default 3 mm) in more than --pen_frac of the last blocks, or
    the block AVERAGE of the per-episode max > --pen_avg_mm (default 1 mm);
  - table impacts: impact rate in the last --window blocks not below the first --window blocks after warm-up and
    > --impact_min (default 0.05).

    python scripts/monitor_ibrl_contact_health.py exps/sim2real/ibrl_robotiq_v5_fr_chunk4_seed1 [...] --every 600
"""
import argparse
import os
import re
import time

KEYS = ("score/score", "score/train_score", "safety/table_impact", "safety/table_force_max_N",
        "safety/pen_needle_tripod_mm", "safety/pen_tripod_table_mm", "safety/pen_needle_table_mm",
        "safety/pen_grip_needle_mm")


def parse(path):
    """-> {step: {key: (avg, max)}} from a train.log"""
    out = {}
    pat = re.compile(r"^(\d+): (\S+)\s+(?:\[\s*\d+\]: avg:\s+([-\d.e]+), min:\s+[-\d.e]+\[\s*\d+\], max:\s+([-\d.e]+)|: ([-\d.e]+))")
    for line in open(path, errors="ignore"):
        m = pat.match(line)
        if not m or m.group(2) not in KEYS:
            continue
        step, key = int(m.group(1)), m.group(2)
        avg = float(m.group(3) if m.group(3) is not None else m.group(5))
        mx = float(m.group(4)) if m.group(4) is not None else avg
        out.setdefault(step, {})[key] = (avg, mx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=600, help="seconds between checks")
    ap.add_argument("--pen_max_mm", type=float, default=3.0)
    ap.add_argument("--pen_frac", type=float, default=0.5)
    ap.add_argument("--pen_avg_mm", type=float, default=1.0)
    ap.add_argument("--impact_min", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    seen = {r: set() for r in args.runs}
    while True:
        for r in args.runs:
            log = os.path.join(r, "train.log")
            if not os.path.exists(log):
                continue
            blocks = parse(log)
            name = os.path.basename(r.rstrip("/"))
            for step in sorted(blocks):
                if step in seen[r] or "safety/table_impact" not in blocks[step]:
                    continue
                seen[r].add(step)
                b = blocks[step]
                g = lambda k, i=0: b.get(k, (float("nan"), float("nan")))[i]
                print(f"[health] {name} {step:>7d} | eval {g('score/score'):.2f} train {g('score/train_score'):.2f} | "
                      f"impact {g('safety/table_impact'):.2f} (force max {g('safety/table_force_max_N', 1):.0f} N) | "
                      f"needle-ring avg {g('safety/pen_needle_tripod_mm'):.2f} max {g('safety/pen_needle_tripod_mm', 1):.2f} mm | "
                      f"tripod-table max {g('safety/pen_tripod_table_mm', 1):.2f} | needle-table max "
                      f"{g('safety/pen_needle_table_mm', 1):.2f} | grip-needle max {g('safety/pen_grip_needle_mm', 1):.2f}",
                      flush=True)
            st = [s for s in sorted(blocks) if "safety/table_impact" in blocks[s]]
            if len(st) >= args.window:
                last = st[-args.window:]
                pens = [blocks[s]["safety/pen_needle_tripod_mm"] for s in last]
                if (sum(p[1] > args.pen_max_mm for p in pens) > args.pen_frac * len(pens)
                        or max(p[0] for p in pens) > args.pen_avg_mm):
                    print(f"[ALERT] {name} needle-ring penetration over the last {len(last)} blocks: "
                          f"avg {[round(p[0], 2) for p in pens]} max {[round(p[1], 2) for p in pens]} mm", flush=True)
            if len(st) >= 2 * args.window + 2:
                first = [blocks[s]["safety/table_impact"][0] for s in st[2:2 + args.window]]
                last = [blocks[s]["safety/table_impact"][0] for s in st[-args.window:]]
                if sum(last) / len(last) > args.impact_min and sum(last) / len(last) >= sum(first) / len(first):
                    print(f"[ALERT] {name} table-impact rate not decreasing: early {sum(first)/len(first):.2f} "
                          f"-> recent {sum(last)/len(last):.2f}", flush=True)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
