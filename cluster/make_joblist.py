#!/usr/bin/env python3
"""Generate a SEED-MAJOR job list for the Dyna-IBRL batches.

Why this exists: picking a subset of arms by --array range only works if the
list happens to be laid out that way, and hand-editing a list is how
cluster/jobs_sqcof_dynaibrl.txt ended up 40 lines long against a `--array=1-20`
header (every cof_* row silently never ran). Name the arms you want instead.

Seed-major output means --array width IS the budget: with N arms, --array=1-N is
one seed of everything, 1-5N is five seeds, and no arm is starved by a truncated
range.

    python3 cluster/make_joblist.py --arms sq_dynaibrl_m001,cof_dynaibrl_m001 \
        --seeds 10 --out cluster/jobs_x.txt

--skip-done drops rows whose run already logged step 100000 (the same test the
job scripts use to declare a run complete). Run it ON THE CLUSTER for that to
mean anything -- the exps/ tree here is the workstation's.
"""
import argparse, os, sys

# Same mapping the job scripts use to pick EXP_DIR from the arm prefix.
def exp_dir(arm: str) -> str:
    if arm.startswith("tpa_"):
        return "exps/tpa_kp025_6arm"
    if arm.startswith("sq_"):
        return "exps/square_rotw1_6arm"
    if arm.startswith("cof_"):
        return "exps/coffee_g5x_3arm"
    return "exps/simdist_threading_3arm"

# Arms each job script knows. A typo here costs a queue wait and then
# "unknown arm", so it is checked up front.
LAM_A = [f"lamA_{v}" for v in ("0.1", "0.3", "3", "10")]
LAM_B = [f"dsrcB_{d}" for d in ("0.02", "0.035", "0.05", "0.08")]
LAM_B_SQCOF = [f"{t}_dsrcB_{d}" for t in ("sq", "cof")
               for d in ("0.02", "0.035", "0.05", "0.08")]
SQCOF = [f"{t}_dynaibrl{s}" for t in ("sq", "cof")
         for s in ("", "_m0", "_m001", "_sgft", "_plain", "_plain_m0",
                   "_plain_m001", "_plain_sgft")]
SQCOF += [f"{t}_valabl_{a}" for t in ("sq", "cof")
          for a in ("dyn", "joint", "v2d", "d2v", "marg")]
SQCOF += LAM_B_SQCOF
SQCOF += [f"{t}_anch_{f}" for t in ("sq", "cof") for f in ("f1", "f5")]
SQCOF += ["sq_plain_d600000", "sq_plain_d400000",      # SimDist MPPI reference
          "cof_plain",
          "sq_ibrl_m001", "cof_ibrl_m001",              # model-free, no world model
          "sq_ibrl_m001_qsrc", "cof_ibrl_m001_qsrc",   # ... + source-expert critic warm-start
          "sq_ibrl_m0_qsrc", "cof_ibrl_m0_qsrc",       # ... same, margin 0
          "sq_ibrl_m0", "cof_ibrl_m0"]                 # margin 0, NO warm-start
ANCHOR_ABL = ["anch_f2wd", "anch_f3l2", "anch_f4a", "anch_f4b", "anch_f5",
              "anch_dat", "anch_es_src", "anch_es_dat"]
THREADING = ["plain_h5", "plain_nobc", "plain_anch", "dynens_nobc",
             *ANCHOR_ABL, *LAM_A, *LAM_B,
             "valabl_dyn", "valabl_joint", "valabl_v2d", "valabl_d2v", "valabl_marg", "ibrl_m005anneal0", "ibrl_m001", "ibrl_m001_qsrc",
             "ibrl_m0_qsrc", "ibrl_m0",
             "dynaibrl_plain", "dynaibrl_plain_m0", "dynaibrl_plain_m001",
             "dynaibrl_plain_m005", "dynaibrl_plain_m01",
             "dynaibrl_plain_sgft",
             "dynaibrl_dynens", "dynaibrl_dynens_m0", "dynaibrl_dynens_m001",
             "dynaibrl_dynens_sgft"]
TPA = ["tpa_ibrl", "tpa_ibrl_qsrc", "tpa_ibrl_qsrc_m001",
       "tpa_dynaibrl", "tpa_dynaibrl_m001", "tpa_dynaibrl_dynens_m001",
       "tpa_iql", "tpa_rlpd"]
KNOWN = set(SQCOF) | set(THREADING) | set(TPA)

def done(arm: str, seed: int, override: str | None) -> bool:
    log = os.path.join(override or exp_dir(arm), f"{arm}_seed{seed}", "train.log")
    try:
        with open(log, errors="replace") as f:
            return any(l.startswith("100000: score/score") for l in f)
    except OSError:
        return False

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--seeds", type=int, default=10, help="how many seeds (1, 101, 201, ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-done", action="store_true",
                    help="drop rows whose train.log already reached step 100000")
    ap.add_argument("--exp-dir", default=None, help="override the arm->exps/ mapping")
    ap.add_argument("--tail-arms", default="",
                    help="comma-separated subset of --arms to emit as a SECOND "
                         "seed-major block after all the others, so they occupy a "
                         "contiguous --array range at the end of the file. Use when "
                         "only the new arms still need to run.")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    unknown = [x for x in arms if x not in KNOWN]
    if unknown:
        print(f"unknown arm(s): {', '.join(unknown)}", file=sys.stderr)
        print("known:\n  " + "\n  ".join(sorted(KNOWN)), file=sys.stderr)
        return 1
    if len({exp_dir(x) for x in arms}) > 1 and not a.exp_dir:
        # sq_ and cof_ share a job script, so mixing them is fine. Mixing either
        # with a Threading arm is not: they are different job scripts.
        fams = [f for f in (THREADING, SQCOF, TPA) if any(x in f for x in arms)]
        if len(fams) > 1:
            print("arms from different job scripts (Threading / sq_+cof_ / tpa_) "
                  "cannot share one list -- generate one per script.",
                  file=sys.stderr)
            return 1

    tail = [x.strip() for x in a.tail_arms.split(",") if x.strip()]
    unknown_tail = [x for x in tail if x not in arms]
    if unknown_tail:
        print(f"--tail-arms not in --arms: {', '.join(unknown_tail)}", file=sys.stderr)
        return 1
    head = [x for x in arms if x not in tail]

    seeds = [1 + 100 * i for i in range(a.seeds)]
    rows, skipped = [], 0
    boundary = None
    for block in ([head, tail] if tail else [arms]):
        if not block:
            continue
        if block is tail:
            boundary = len(rows) + 1
        for s in seeds:
            for arm in block:       # seed-major within each block
                if a.skip_done and done(arm, s, a.exp_dir):
                    skipped += 1
                    continue
                rows.append(f"{arm} {s}")
    with open(a.out, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{a.out}: {len(rows)} rows, {len(arms)} arms x {a.seeds} seeds"
          + (f" ({skipped} already complete, skipped)" if a.skip_done else ""))
    if rows:
        print(f"  sbatch --array=1-{len(rows)} ...")
        if boundary is not None:
            print(f"  tail block ({', '.join(tail)}) only:  "
                  f"--array={boundary}-{len(rows)}")
        if a.skip_done and skipped:
            print("  ⚠️ --skip-done makes the list RAGGED: it is no longer "
                  "'N arms per seed block'. Read it, do not assume the layout.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
