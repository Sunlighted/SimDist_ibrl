#!/usr/bin/env python3
"""Re-score an adapt run's periodic snapshots over a grid of EVAL-TIME knobs.

Ensemble size (`--dyn_heads`) and the pessimism coef (`dynamics_disagree_coef`)
only re-score a fixed checkpoint, so sweeping them inside training multiplies the
wall clock of every run: a 100-episode MPPI eval is 221-282 s and a 100k run has
~20 eval points, i.e. ~1.5 h per extra coef or head size. Train ONE run with
`--snapshot_every_rounds 10`, ship the snapshots, and sweep here instead.

⚠️ evaluate/eval.py CANNOT read an adapt save_dir directly: train_simdist's
load_model reads cfg.yaml next to the weight, and an adapt dir's cfg.yaml is an
AdaptConfig dump whose `target_num_data: null` makes MainConfig parsing blow up.
This builds a shim dir holding the STAGE-1 cfg.yaml plus symlinks to the
snapshots -- which is also the correct env, since the target is defined by the
stage-1 cfg (CLAUDE.md rule 19b).

    python scripts/eval_snapshots.py --run exps/optim_local/thr_optim_0.5_seed1 \
        --stage1 exps/mimicgen/threading_simdist_dynens10_diversev2_h5_seed1 \
        --heads 1,3,5,10 --coefs 0,0.01,0.05,0.1,0.2 --num_game 100 --gpus 0,1,2,3

Writes <run>/snapshot_sweep.json:  [{snap, transitions, heads, coef, success}]
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import queue

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("EVAL_PY", "/home/Chenyang/.conda/envs/ibrl/bin/python")
RE_SR = re.compile(r"success rate:\s*([0-9.]+)%")


def build_shim(run: str, stage1: str, out: str) -> list:
    """Shim dir = stage-1 cfg.yaml + symlinks to the run's snapshots."""
    cfg = os.path.join(stage1, "cfg.yaml")
    if not os.path.isfile(cfg):
        sys.exit(f"MISSING stage-1 cfg: {cfg}")
    os.makedirs(out, exist_ok=True)
    dst_cfg = os.path.join(out, "cfg.yaml")
    if not os.path.exists(dst_cfg):
        os.symlink(os.path.abspath(cfg), dst_cfg)

    snaps = []
    for f in sorted(os.listdir(run)):
        m = re.fullmatch(r"snap_(\d+)\.pt", f)
        if not m:
            continue
        link = os.path.join(out, f)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(run, f)), link)
        snaps.append((int(m.group(1)), link))
    if not snaps:
        sys.exit(f"no snap_*.pt in {run} -- was it trained with "
                 f"--snapshot_every_rounds ?")
    return sorted(snaps)


def run_one(gpu, weight, heads, coef, args, extra):
    cmd = [PY, "evaluate/eval.py", "--weight", weight, "--mode", "simdist",
           "--num_game", str(args.num_game), "--mp", str(args.mp),
           "--shift_from_cfg", "1",
           "--hand_init_perturb_joint_sigma", str(args.sigma)]
    if args.strict_success:
        cmd += ["--strict_success", "1"]
    if args.original_mppi:
        cmd += ["--original_mppi", "1"]
    # --planner_set is applied AFTER --original_mppi, so the coef survives the
    # recipe reset. Any user planner_set is appended to the same flag, since
    # eval.py takes one comma-separated string.
    pset = f"dynamics_disagree_coef={coef}"
    if args.planner_set:
        pset += "," + args.planner_set
    cmd += ["--planner_set", pset]
    if heads is not None:
        cmd += ["--dyn_heads", str(heads)]
    cmd += extra

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=REPO, MUJOCO_GL=os.environ.get("MUJOCO_GL", "osmesa"))
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    hits = RE_SR.findall(p.stdout)
    if not hits:
        tail = (p.stdout or p.stderr)[-400:]
        return None, tail
    return float(hits[-1]) / 100.0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="adapt save_dir holding snap_*.pt")
    ap.add_argument("--stage1", required=True, help="stage-1 dir (for cfg.yaml)")
    ap.add_argument("--heads", default="", help="e.g. 1,3,5,10; empty = all heads")
    ap.add_argument("--coefs", default="0", help="dynamics_disagree_coef grid")
    ap.add_argument("--num_game", type=int, default=100)
    ap.add_argument("--mp", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--strict_success", type=int, default=1)
    ap.add_argument("--original_mppi", type=int, default=1)
    ap.add_argument("--planner_set", default="", help="extra planner overrides")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--snaps", default="", help="only these transition counts")
    args, extra = ap.parse_known_args()

    if args.num_game % args.mp:
        sys.exit(f"--mp {args.mp} must divide --num_game {args.num_game} "
                 f"(mp eval asserts this)")

    shim = os.path.join(args.run, "_eval_shim")
    snaps = build_shim(args.run, args.stage1, shim)
    if args.snaps:
        want = {int(x) for x in args.snaps.split(",")}
        snaps = [s for s in snaps if s[0] in want]
    heads = [int(h) for h in args.heads.split(",") if h.strip()] or [None]
    coefs = [float(c) for c in args.coefs.split(",") if c.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    jobs = [(t, w, h, c) for (t, w), h, c in itertools.product(snaps, heads, coefs)]
    print(f"[sweep] {len(snaps)} snapshots x {len(heads)} head sizes x "
          f"{len(coefs)} coefs = {len(jobs)} evals of {args.num_game} games, "
          f"on GPUs {gpus}")

    out_path = args.out or os.path.join(args.run, "snapshot_sweep.json")
    done = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path)):
            done[(r["transitions"], r["heads"], r["coef"])] = r
        print(f"[sweep] resuming: {len(done)} cells already in {out_path}")

    q = queue.Queue()
    for j in jobs:
        if (j[0], j[2], j[3]) not in done:
            q.put(j)
    lock = threading.Lock()
    results = list(done.values())

    def worker(gpu):
        while True:
            try:
                t, w, h, c = q.get_nowait()
            except queue.Empty:
                return
            sr, err = run_one(gpu, w, h, c, args, extra)
            with lock:
                if sr is None:
                    print(f"  FAIL  t={t:>7} heads={h} coef={c}\n    {err}")
                else:
                    print(f"  ok    t={t:>7} heads={str(h):>4} coef={c:<5} "
                          f"success={sr:.3f}")
                    results.append(dict(snap=os.path.basename(w), transitions=t,
                                        heads=h, coef=c, success=sr,
                                        num_game=args.num_game))
                    with open(out_path, "w") as f:
                        json.dump(sorted(results, key=lambda r: (
                            r["transitions"], r["heads"] or 0, r["coef"])), f,
                            indent=1)

    ts = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t_ in ts:
        t_.start()
    for t_ in ts:
        t_.join()
    print(f"[sweep] wrote {out_path} ({len(results)} cells)")


if __name__ == "__main__":
    main()
