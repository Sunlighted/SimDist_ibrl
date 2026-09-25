#!/bin/bash
# §48p: bc_anchor_coef (= the KL coefficient) sweep on §47's SHIFT-ONLY stage-1.
# lambda=0 and lambda=1 already exist (aug48k_B_s*/aug48k_C_s*); this adds the rest.
# Protocol identical to every §48 arm or the paired deltas are void.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/yifanr/simdist/repo}"
STAGE1=exps/mimicgen/threading_pixel224_stage1_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5
ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5
LAMBDAS="${LAMBDAS:-0.1 0.3 3}"; SEEDS="${SEEDS:-1 2 3}"; ACC="${ACC:-aip-gigor}"
DRYRUN="${DRYRUN:-0}"
sb () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi; }
for f in "$STAGE1" "$DEMOS" "$ANCHOR"; do [ -s "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done
n=0
for LAM in $LAMBDAS; do
  TAG=$(echo "$LAM" | tr . p)          # 0.1 -> 0p1 (job names / dirs)
  for SEED in $SEEDS; do
    OFF=$(( (SEED - 1) * 20 ))
    DIR="exps/mimicgen/aug48sw_lam${TAG}_s${SEED}"
    AID=$(ARM=pix224 STAGE1="$STAGE1" DEMOS="$DEMOS" ANCHOR="$ANCHOR" SAVE="$DIR" \
      NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 \
      DYN_ANCHOR_COEF=0.1 ANCHOR_COEF="$LAM" DEMO_OFFSET="$OFF" \
      sb --parsable --job-name="sw${TAG}s${SEED}" --time=23:59:00 --mem=96G \
         --cpus-per-task=12 --account="$ACC" scripts/slurm/adapt.slrm)
    [ -z "$AID" ] && { echo "!! adapt submit FAILED lam=$LAM seed=$SEED"; continue; }
    EID=$(ADAPT="$DIR" N=500 MP=10 TAG="aug48sw_lam${TAG}_s${SEED}" \
      sb --parsable --job-name="ew${TAG}s${SEED}" --time=2:59:00 --account="$ACC" \
         --dependency=afterany:"$AID" scripts/slurm/eval_simdist.slrm)
    echo "lambda=$LAM seed=$SEED offset=$OFF adapt=$AID eval=${EID:-FAILED}"; n=$((n+1))
  done
done
echo "=== submitted $n lambda-sweep pairs ==="
