#!/usr/bin/env bash
# Summarize the flow BC data-efficiency sweep: for each demo-count, evaluate the
# best checkpoint on SOURCE (mode=none) with a large game count for a low-noise
# success number, and print a success-vs-#demos table.
#
# Usage:  GPU=1 SEED=1 NGAME=200 bash scripts/summarize_datasize_sweep.sh
set -eo pipefail
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

GPU=${GPU:-0}
SEED=${SEED:-1}
NGAME=${NGAME:-200}
POINTS=${POINTS:-"10 20 50 100 200 500"}

echo "=== flow BC data-efficiency sweep | SOURCE eval | ${NGAME} games ==="
printf "%-8s %-12s\n" "#demos" "source_succ"
for N in $POINTS; do
  CKPT=exps/mimicgen/threading_flow_bc_n${N}_seed${SEED}/model0.pt
  if [[ ! -f $CKPT ]]; then printf "%-8s %-12s\n" "$N" "(no ckpt)"; continue; fi
  SCORE=$(CUDA_VISIBLE_DEVICES=${GPU} python evaluate/eval.py \
            --weight $CKPT --mode bc --num_game ${NGAME} --shift_mode none 2>/dev/null \
          | grep -iE "score" | tail -1)
  printf "%-8s %-12s\n" "$N" "$SCORE"
done
