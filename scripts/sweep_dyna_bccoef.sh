#!/bin/bash
# Dyna stage-3: bc_coef scan (the ORIGINAL Dyna knob -- TD3+BC-style regularisation
# on the 20 target demos, already in DynaConfig; default 0.1).
#
# WHY: in the trust-region sweep bc_coef was still climbing at the top of its range
# and never turned over, so its optimum is unbracketed:
#     0.1 -> 0.171   1.0 -> 0.208   10 -> 0.386      (75-100k steady, paired seeds)
# The stage-3 anchor peaked at coef=10 with an inverted U (1: 0.471, 10: 0.619,
# 100: 0.476), so bc_coef is expected to turn over somewhere too. This scan brackets
# it: 30 / 100 / 300. It also answers "how far do the ORIGINAL Dyna knobs go on
# their own", since bc_coef is the only native knob that beat doing nothing.
#
# Cells bc1/bc10 are listed only to top up their missing seed 201 -- completed runs
# are skipped, so they cost 1 run each rather than 3.
#
# Usage:  bash scripts/sweep_dyna_bccoef.sh <worker_id> <gpu> [nworker]
set -u
WORKER=${1:?worker id}
GPU=${2:?gpu id}
ROOT=/home/Chenyang/ibrl
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
STAGE1=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_sat_400000_s1/dataset.hdf5
NWORKER=${3:-6}

cd $ROOT
export PYTHONPATH=$ROOT OMP_NUM_THREADS=1
export MUJOCO_PY_MUJOCO_PATH=/home/Chenyang/.mujoco/mujoco210
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=$GPU

declare -A CELL=(
  [bc30]="--dyna.bc_coef 30.0"
  [bc100]="--dyna.bc_coef 100.0"
  [bc300]="--dyna.bc_coef 300.0"
  [bc1]="--dyna.bc_coef 1.0"      # seed 201 top-up only
  [bc10]="--dyna.bc_coef 10.0"    # seed 201 top-up only
)
ORDER=(bc30 bc100 bc300 bc1 bc10)

i=0
for SEED in 1 101 201; do
  for CELL_NAME in "${ORDER[@]}"; do
    if (( i % NWORKER == WORKER )); then
      RUN=dyna_${CELL_NAME}_seed${SEED}
      DIR=exps/simdist_threading_3arm/$RUN
      if [[ -f $DIR/train.log ]] && grep -q "^100000: score/score" $DIR/train.log 2>/dev/null; then
        echo "[w$WORKER] SKIP $RUN (already complete)"
      else
        rm -rf $DIR    # a killed partial run must not be resumed into
        echo "[w$WORKER] $(date '+%m-%d %H:%M') START $RUN  ${CELL[$CELL_NAME]}"
        $PY train_simdist_dyna.py \
          --stage1_ckpt $STAGE1 --target_demo_path $DEMOS \
          --save_dir $DIR --seed $SEED \
          --max_transitions 100000 --eval_every_rounds 5 \
          --num_eval_episode 100 --num_eval_proc 10 \
          --eval_hand_init_perturb_joint_sigma 0.1 \
          ${CELL[$CELL_NAME]} > $DIR.launch.log 2>&1
        echo "[w$WORKER] $(date '+%m-%d %H:%M') DONE  $RUN rc=$?"
      fi
    fi
    i=$((i+1))
  done
done
echo "[w$WORKER] all tasks finished"
