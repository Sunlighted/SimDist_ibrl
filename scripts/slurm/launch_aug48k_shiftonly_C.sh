#!/bin/bash
# §48k: rung C (anchor lambda=1) at 5 disjoint demo draws on §47's EXISTING
# shift-only stage-1. No stage-1 retrain: rungs B and C share a stage-1, and
# `threading_pixel224_stage1_seed1` IS the shift-only dynens checkpoint §47 used
# (verified: 150000 steps, 5 heads, H=5, aug_shift_pad=8, all other aug knobs 0).
#
# PURPOSE (see docs/simdist.md §48k for the pre-registered predictions):
#   seed 1 uses draw [0,20) on that exact checkpoint with the byte-identical
#   adapt protocol => it is a PURE REPLICATION TEST of §47's 0.766. The other
#   four draws give the anchor arm's draw-variance on a shift-only checkpoint,
#   and pair by draw against §48's full-aug rung C for the augmentation contrast.
#
# Protocol is IDENTICAL to §48 or the comparison is void (cmp_adapt_runs.py
# refuses across budgets): NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100
# DYN_ANCHOR_COEF=0.1, ORIGINAL MPPI, ANCHOR_COEF=1.0.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/yifanr/simdist/repo}"

STAGE1=exps/mimicgen/threading_pixel224_stage1_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5
ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5
# All on gigor: it beat rhinehar by 13 h and then 17 h on measured start times
# (CLAUDE.md 16b). Unlike the §48 fan-out there is NO within-arm pairing
# constraint here -- these 5 runs pair against the ALREADY-COMPLETE full-aug
# rung C by draw -- so concentrating on the fast queue costs nothing, and a
# straggler can be moved with `scontrol update Account=` (dependency-safe).
ACCTS=(aip-gigor aip-gigor)
# RUNG selects which rung of the SHIFT-ONLY ladder to run on §47's checkpoint.
# Both share that stage-1, so neither needs a retrain:
#   RUNG=C (default) -> ANCHOR_COEF=1.0, dirs aug48k_C_s*, jobs a48Ks*
#   RUNG=B           -> ANCHOR_COEF=0,   dirs aug48k_B_s*, jobs a48Ls*
# B exists so the shift-only ANCHOR DELTA is paired within seed instead of
# resting on §47's single B run (window 0.425, n=1).
RUNG="${RUNG:-C}"
case "$RUNG" in
  C) COEF=1.0; TAGCH=K ;;
  B) COEF=0;   TAGCH=L ;;
  *) echo "FATAL: RUNG must be B or C (got '$RUNG')"; exit 1 ;;
esac
DRYRUN="${DRYRUN:-0}"
sb () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi; }

for f in "$STAGE1" "$DEMOS" "$ANCHOR"; do
    [ -s "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "=== §48k shift-only rung $RUNG (anchor=$COEF) | stage1 = $STAGE1 ==="

n=0
for OFF in 0 20 40 60 80; do
    SEED=$(( OFF / 20 + 1 ))
    # account chosen by seed, not submission order (§48 lesson: per-job
    # alternation split the seeds so the fast queue produced no usable pairs).
    ACC="${ACCTS[$((SEED % 2))]}"
    DIR="exps/mimicgen/aug48k_${RUNG}_s${SEED}"
    AID=$(ARM=pix224 STAGE1="$STAGE1" DEMOS="$DEMOS" ANCHOR="$ANCHOR" SAVE="$DIR" \
      NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 \
      DYN_ANCHOR_COEF=0.1 ANCHOR_COEF="$COEF" DEMO_OFFSET="$OFF" \
      sb --parsable --job-name="a48${TAGCH}s${SEED}" --time=23:59:00 --mem=96G \
         --cpus-per-task=12 --account="$ACC" scripts/slurm/adapt.slrm)
    [ -z "$AID" ] && { echo "!! adapt submit FAILED for seed $SEED"; continue; }
    EID=$(ADAPT="$DIR" N=500 MP=10 TAG="aug48k_${RUNG}_s${SEED}" \
      sb --parsable --job-name="e48${TAGCH}s${SEED}" --time=2:59:00 --account="$ACC" \
         --dependency=afterany:"$AID" scripts/slurm/eval_simdist.slrm)
    echo "seed $SEED  offset=$OFF  acct=$ACC  adapt=$AID eval=${EID:-FAILED}"
    n=$((n+1))
done
echo "=== submitted $n shift-only rung-$RUNG pairs ==="
