#!/bin/bash
# Seed replication for Sec. "What Makes Dyna Work", panel (b): WHICH CHANNEL the
# prior-preference floor acts through.  Four arms, one variable each, all at
# m=0.01 with the 20 target demos preloaded.  seed 1 already exists in
# exps/simdist_threading_3arm/ ; this script produces the replications.
#
#   both      margin in every channel                (= dynaibrl_m001)
#   dataonly  margin NOT in the TD target            (--dyna.margin_on_boot 0)
#   bootonly  margin ONLY in the TD target           (--dyna.margin_on_act 0 + imagine 0)
#   neither   no margin at all                       (= dynaibrl_m0demo)
#
# The evaluated policy is identical in all four (margin_on_eval stays 1), so the
# arms differ only in what happened during TRAINING.
#
# Usage:   ./cluster/job_threading_margin_channels.sh <arm> <seed> <gpu>
# Batch:   see cluster/jobs_threading_margin_channels.txt
#
# NOTES / TRAPS
# - The demo set follows the seed (threading_d0_sat_160000_s${SEED}), so a
#   replication changes BOTH the algorithm seed and the demonstrations. Within
#   one seed the four arms share a demo set, so the arm-vs-arm comparison stays
#   paired; across seeds the spread includes demo-collection variance (~0.09).
# - stage-1 is FIXED across seeds (one distillation, reused). The reported seed
#   spread therefore does NOT include stage-1 variance. Say so in the paper.
# - preload_demos is passed EXPLICITLY even though it currently defaults to 1:
#   the seed-1 runs of panel (a) predate the flag and behaved as 0, and relying
#   on a default is how that pair became incomparable in the first place.
set -euo pipefail
cd /home/Chenyang/ibrl

ARM=${1:?arm: both|dataonly|bootonly|neither}
SEED=${2:?seed}
GPU=${3:-0}

PY=/home/Chenyang/.conda/envs/ibrl/bin/python
CK=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5
EXP=exps/simdist_threading_3arm

[ -f "$CK" ]     || { echo "MISSING stage-1 ckpt $CK"; exit 1; }
[ -f "$DEMOS" ]  || { echo "MISSING demos $DEMOS"; exit 1; }

case $ARM in
  both)     NAME=dynaibrl_m001;          MARGIN=0.01; EXTRA="" ;;
  dataonly) NAME=dynaibrl_m001dataonly;  MARGIN=0.01; EXTRA="--dyna.margin_on_boot 0" ;;
  bootonly) NAME=dynaibrl_m001bootonly;  MARGIN=0.01; EXTRA="--dyna.margin_on_act 0 --dyna.margin_on_imagine 0" ;;
  neither)  NAME=dynaibrl_m0demo;        MARGIN=0.0;  EXTRA="" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=$EXP/${NAME}_seed${SEED}
[ -d "$OUT" ] && { echo "REFUSING: $OUT already exists"; exit 1; }

# GPU guard: this arm peaks at ~2 GB; refuse to start if the card is nearly full
# (a driver-level OOM here takes the whole machine down, not just this job).
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
[ "$FREE" -gt 8000 ] || { echo "GPU $GPU has only ${FREE} MiB free; refusing"; exit 1; }

export PYTHONPATH=/home/Chenyang/ibrl OMP_NUM_THREADS=1 WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=$GPU $PY train_simdist_dyna_ibrl.py \
  --stage1_ckpt $CK --save_dir $OUT --seed $SEED \
  --target_demo_path $DEMOS --target_num_data 20 \
  --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
  --num_eval_episode 100 --num_eval_proc 10 \
  --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
  --dyna.prior_margin $MARGIN --dyna.preload_demos 1 \
  $EXTRA \
  --use_wb 0 --state_only_no_render 1 \
  > $OUT.stdout 2>&1
