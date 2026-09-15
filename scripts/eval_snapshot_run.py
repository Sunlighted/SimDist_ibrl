#!/usr/bin/env python3
"""ONE wandb run = one (sweep, value, seed): re-score every snapshot of an adapt
run at a single eval-time setting, logging to wandb after each snapshot.

    python scripts/eval_snapshot_run.py --run exps/optim_local/thr_snap_o0.5_seed1 \
        --stage1 exps/mimicgen/threading_simdist_dynens10_diversev2_h5_seed1 \
        --sweep penalty_planning --value 0.01 --seed 1

Sweeps (one axis at a time, the other held at its default):
    penalty_planning <c>   dynamics_disagree_coef=c, all heads of the checkpoint
    ens_heads <h>          --dyn_heads h, dynamics_disagree_coef=0

wandb: project --project (optim_threading), group "<sweep><value>" and name
"<sweep><value>_seed<seed>", e.g. penalty_planning0.01_seed1. Keys match the
adapt runs so plots/wandb_fetch.py reads them unchanged:
    adapt/transitions   x axis (the snapshot's transition count)
    score/score         planner success rate at that setting
config carries `seed` (fetch_group keys runs by it).

Why a separate job per run: the head / pessimism knobs only re-score a fixed
checkpoint, so they do not belong inside training (a 100-episode MPPI eval is
221-282 s; a sweep inside every training run doubled its wall clock).

⚠️ evaluate/eval.py cannot read an adapt save_dir (its cfg.yaml is an AdaptConfig
dump; `target_num_data: null` breaks MainConfig). The shim dir pairs the STAGE-1
cfg.yaml with the snapshot weight -- which is also the correct env, since the
target is defined by the stage-1 cfg (CLAUDE.md rule 19b).

Resumable: finished (transitions, success) rows are kept in
<run>/sweeps/<name>.tsv and the wandb run id is derived from the name, so a
requeued job skips done snapshots and appends to the same run.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_SR = re.compile(r"success rate:\s*([0-9.]+)%")


def snapshots(run):
    out = []
    for f in os.listdir(run):
        m = re.fullmatch(r"snap_(\d+)\.pt", f)
        if m:
            out.append((int(m.group(1)), f))
    if not out:
        sys.exit(f"no snap_*.pt in {run} -- trained with --snapshot_every_rounds ?")
    return sorted(out)


def shim(run, stage1, snap_file):
    d = os.path.join(run, "_eval_shim")
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(stage1, "cfg.yaml")
    if not os.path.isfile(cfg):
        sys.exit(f"MISSING stage-1 cfg: {cfg}")
    for src, dst in ((cfg, "cfg.yaml"), (os.path.join(run, snap_file), snap_file)):
        p = os.path.join(d, dst)
        if not os.path.lexists(p):
            try:
                os.symlink(os.path.abspath(src), p)
            except FileExistsError:      # another array task made it first
                pass
    return os.path.join(d, snap_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--sweep", required=True, choices=["penalty_planning", "ens_heads"])
    ap.add_argument("--value", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--project", default="optim_threading")
    ap.add_argument("--num_game", type=int, default=100)
    ap.add_argument("--mp", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--strict_success", type=int, default=1)
    ap.add_argument("--planner_set", default="", help="extra planner overrides")
    ap.add_argument("--use_wb", type=int, default=1)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if args.num_game % args.mp:
        sys.exit(f"--mp {args.mp} must divide --num_game {args.num_game}")

    group = f"{args.sweep}{args.value}"
    name = f"{group}_seed{args.seed}"
    if args.sweep == "penalty_planning":
        coef, heads = float(args.value), None
    else:
        coef, heads = 0.0, int(args.value)

    os.makedirs(os.path.join(args.run, "sweeps"), exist_ok=True)
    tsv = os.path.join(args.run, "sweeps", f"{name}.tsv")
    done = {}
    if os.path.exists(tsv):
        for line in open(tsv):
            t, sr = line.split()
            done[int(t)] = float(sr)

    wb = None
    if args.use_wb:
        import wandb
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")
        wb = wandb.init(
            project=args.project, group=group, name=name,
            id=hashlib.md5(f"{args.project}/{name}".encode()).hexdigest()[:16],
            resume="allow",
            config=dict(seed=args.seed, sweep=args.sweep, value=args.value,
                        dynamics_disagree_coef=coef, dyn_heads=heads,
                        source_run=os.path.abspath(args.run),
                        stage1=os.path.abspath(args.stage1),
                        num_game=args.num_game, sigma=args.sigma,
                        strict_success=args.strict_success,
                        planner_set=args.planner_set))

    snaps = snapshots(args.run)
    print(f"[{name}] {len(snaps)} snapshots, {len(done)} already done")
    for t, f in snaps:
        if t in done:
            sr = done[t]
            print(f"  skip t={t:>6}  success={sr:.3f} (from {tsv})")
        else:
            w = shim(args.run, args.stage1, f)
            # --planner_set is applied AFTER --original_mppi, so the coef survives
            # the recipe reset.
            pset = f"dynamics_disagree_coef={coef}"
            if args.planner_set:
                pset += "," + args.planner_set
            cmd = [args.python, "evaluate/eval.py", "--weight", w, "--mode", "simdist",
                   "--num_game", str(args.num_game), "--mp", str(args.mp),
                   "--shift_from_cfg", "1",
                   "--strict_success", str(args.strict_success),
                   "--hand_init_perturb_joint_sigma", str(args.sigma),
                   "--original_mppi", "1", "--planner_set", pset]
            if heads is not None:
                cmd += ["--dyn_heads", str(heads)]
            log = os.path.join(args.run, "sweeps", f"{name}_t{t}.log")
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            txt = open(log, errors="replace").read()
            hits = RE_SR.findall(txt)
            # guard against a silently-ignored knob: eval.py prints both overrides
            ok_h = heads is None or f"dynamics heads:" in txt
            ok_c = coef == 0.0 or "dynamics_disagree_coef" in txt
            if not hits or not ok_h or not ok_c:
                print(f"  FAIL t={t} rc={rc} (success line={bool(hits)} heads={ok_h} "
                      f"coef={ok_c}) -- see {log}")
                print("  " + txt[-600:].replace("\n", "\n  "))
                continue
            sr = float(hits[-1]) / 100.0
            with open(tsv, "a") as fh:
                fh.write(f"{t}\t{sr}\n")
            print(f"  ok   t={t:>6}  success={sr:.3f}")
        if wb is not None and t not in done:
            wb.log({"adapt/transitions": t, "score/score": sr})
    if wb is not None:
        # A resumed run already has the skipped rows; a FRESH run whose tsv was
        # copied in from elsewhere would be missing them, so backfill in that case.
        if wb.resumed is False and done:
            for t in sorted(done):
                wb.log({"adapt/transitions": t, "score/score": done[t]})
        wb.finish()
    print(f"[{name}] done -> {tsv}")


if __name__ == "__main__":
    main()
