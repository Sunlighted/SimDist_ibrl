#!/usr/bin/env bash
# SimDist TARGET adaptation (CLAUDE.md §17): load a stage-1-distilled world model,
# (stage 2) BC the POLICY head on 20 TARGET demos with the encoder frozen, then
# (stage 3) adapt the DYNAMICS head online via MPPI rollouts in the TARGET env —
# every 1000 collected transitions train dynamics only + eval 500 episodes, up to
# 20000 transitions. Everything except the named head stays frozen (incl. encoder).
#
#   bash scripts/train_simdist_adapt_threading.sh
#   SEED=2 STAGE1_CKPT=exps/.../latest.pt bash scripts/train_simdist_adapt_threading.sh
#   GPU=1 bash scripts/train_simdist_adapt_threading.sh
set -euo pipefail

SEED=${SEED:-1}
STAGE1_CKPT=${STAGE1_CKPT:-exps/mimicgen/threading_simdist_tdmpc2_stage1_seed1/latest.pt}
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_simdist_adapt_seed${SEED}}
GPU=${GPU:-0}

BC_STEPS=${BC_STEPS:-1000}
ADAPT_INTERVAL=${ADAPT_INTERVAL:-1000}
MAX_TRANSITIONS=${MAX_TRANSITIONS:-20000}
DYN_UPDATE_STEPS=${DYN_UPDATE_STEPS:-1000}
NUM_EVAL_EPISODE=${NUM_EVAL_EPISODE:-500}
NUM_EVAL_PROC=${NUM_EVAL_PROC:-10}
USE_WB=${USE_WB:-1}
# SIGMA: TARGET eval hand-init joint sigma. empty => inherit stage-1 cfg
# (robosuite default 0.02). Set e.g. SIGMA=0.1 for the project's final-eval口径.
SIGMA=${SIGMA:-}

EXTRA=()
if [ -n "${SIGMA}" ]; then
    EXTRA+=(--eval_hand_init_perturb_joint_sigma "${SIGMA}")
fi

CUDA_VISIBLE_DEVICES=${GPU} python train_simdist_adapt.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --save_dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --bc_steps "${BC_STEPS}" \
    --adapt_interval "${ADAPT_INTERVAL}" \
    --max_transitions "${MAX_TRANSITIONS}" \
    --dyn_update_steps "${DYN_UPDATE_STEPS}" \
    --num_eval_episode "${NUM_EVAL_EPISODE}" \
    --num_eval_proc "${NUM_EVAL_PROC}" \
    --use_wb "${USE_WB}" \
    "${EXTRA[@]}" \
    "$@"
