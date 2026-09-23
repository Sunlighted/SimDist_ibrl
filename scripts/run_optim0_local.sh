#!/bin/bash
# Local queue for the optim_0 arm (collect_disagree_coef 0) of the Threading
# exploration-optimism sweep, W&B project optim_threading / group optim_0.
#
# Command is copied field for field from the optim_* branch of
# cluster/job_threading_margin.sh (which produced optim_0.05 ... optim_2.0 on
# the cluster); the only change is V=0. With coef 0 the collect branch in
# train_simdist_adapt.py leaves the planner at its eval coef (0) => pure exploit.
#
# Usage (one worker per call, seeds run sequentially):
#   setsid bash scripts/run_optim0_local.sh <gpu> <seed> [<seed> ...] \
#       > exps/optim_threading/queue_gpu<gpu>.log 2>&1 < /dev/null & disown
#
# W&B: offline while training (online mode deadlocks long runs), synced from
# <save_dir>/wandb right after each seed finishes. A seed whose save_dir already
# has a finished marker is skipped, so the queue can be re-launched safely.
set -u
cd "$(dirname "$0")/.."
source set_env.sh
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
WANDB="/home/Chenyang/.conda/envs/ibrl/bin/python -m wandb"   # the console script has a broken shebang
GPU=$1; shift

export CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_MODE=offline

STAGE1_DYNENS=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
EXP_DIR=exps/optim_threading
V=0

for SEED in "$@"; do
  OUT=$EXP_DIR/optim_${V}_seed${SEED}
  if [ -f "$OUT/.done" ]; then echo "[$(date)] skip seed $SEED (done)"; continue; fi
  mkdir -p "$OUT"
  DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5
  echo "[$(date)] gpu=$GPU start seed $SEED -> $OUT"
  WANDB_DIR=$OUT $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_DYNENS \
    --save_dir $OUT --seed $SEED \
    --target_demo_path $DEMOS --target_num_data 20 \
    --bc_anchor_coef 1.0 \
    --bc_anchor_path release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
    --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
    --num_eval_episode 100 --num_eval_proc 10 \
    --eval_disagree_coefs 0 \
    --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
    --collect_disagree_coef $V --original_mppi 1 \
    --use_wb 1 --state_only_no_render 1 > "$OUT.stdout" 2>&1
  rc=$?
  echo "[$(date)] seed $SEED rc=$rc"
  for d in "$OUT"/wandb/offline-run-*; do
    [ -d "$d" ] && $WANDB sync "$d" && echo "[$(date)] synced $d"
  done
  [ $rc -eq 0 ] && touch "$OUT/.done"
done
echo "[$(date)] queue on gpu $GPU finished"
