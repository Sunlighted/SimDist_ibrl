#!/usr/bin/env bash
# Ad-hoc sweep: target_batch_ratio for cotrain-unweighted BC (n1000 src + 20 tgt).
# Everything except TARGET_RATIO matches threading_cotrain_unweighted_n1000_seed1
# (unweighted gap_tau=1e6, weight_mode=sample, MLP-BC, seed 1). ratio=0.5 reuses
# the existing baseline, so we only run 0.05/0.1/0.2/0.3/0.7 here.
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

# 3 free GPUs (1,2,3); 5 runs. Pack 2 onto GPU1 & GPU2, 1 on GPU3.
( run 1 0.05 ratio005; run 1 0.30 ratio030 ) &
( run 2 0.10 ratio010; run 2 0.70 ratio070 ) &
( run 3 0.20 ratio020 ) &
wait
echo "[ALL DONE] $(date +%T)"
