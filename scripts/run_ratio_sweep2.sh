#!/usr/bin/env bash
# Fine sweep around the peak: target_batch_ratio 0.40/0.45/0.55/0.60.
# Same setup as run_ratio_sweep.sh (n1000 src + 20 tgt, unweighted gap_tau=1e6,
# weight_mode=sample, MLP-BC, seed 1). 4 runs, one per GPU, all parallel.
cd /home/Chenyang/ibrl

run() {
  local gpu=$1 ratio=$2 tag=$3
  echo "[start] GPU${gpu} ratio=${ratio} tag=${tag} $(date +%T)"
  GPU="$gpu" N_SRC=1000 N_TGT=20 GAP_SOURCE=gt GAP_TAU=1000000 \
    WEIGHT_MODE=sample TARGET_RATIO="$ratio" \
    METHOD="cotrain_unweighted_n1000_${tag}" \
    bash scripts/train_weighted_bc_threading.sh \
    > "exps_ratiosweep_${tag}.log" 2>&1
  echo "[done ] GPU${gpu} ratio=${ratio} tag=${tag} $(date +%T)"
}

( run 0 0.40 ratio040 ) &
( run 1 0.60 ratio060 ) &
( run 2 0.45 ratio045 ) &
( run 3 0.55 ratio055 ) &
wait
echo "[ALL DONE] $(date +%T)"
