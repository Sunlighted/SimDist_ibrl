#!/bin/bash
# IBRL with the frozen ADAPTED SimDist base (MPPI) as the BC prior — head-to-head
# vs the residual run (scripts/train_residual_simdist_threading.sh, §25). Same
# base (threading_simdist_adapt_seed1, ~0.75) + TARGET domain; an RL actor from
# scratch competes with the base in IBRL max-Q (RLPD: demo-mixed replay, RED-Q,
# BC-loss reg). The base's MPPI action is stored in the buffer (queried live only
# at eval). See train_ibrl_simdist.py + CLAUDE.md §25.
#
# Env vars:
#   SEED             random seed (default 1)
#   BASE_ADAPT_CKPT  adapted SimDist ckpt (default threading_simdist_adapt_seed1/model0.pt)
#   GPU              CUDA_VISIBLE_DEVICES (default 0)
#   STEPS            num_train_step (default 100000)
#   Any extra args after the script are forwarded to train_ibrl_simdist.py.
#
# Examples:
#   bash scripts/train_ibrl_simdist_threading.sh
#   GPU=1 SEED=1 bash scripts/train_ibrl_simdist_threading.sh
#
# Standalone re-eval of a saved ckpt on the TARGET:
#   python evaluate/eval.py --weight exps/mimicgen/threading_ibrl_simdist_seed1/latest.pt \
#       --mode ibrl_simdist --num_game 100 --strict_success 1
set -e

SEED=${SEED:-1}
BASE_ADAPT_CKPT=${BASE_ADAPT_CKPT:-exps/mimicgen/threading_simdist_adapt_seed1/model0.pt}
GPU=${GPU:-0}
STEPS=${STEPS:-100000}
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_ibrl_simdist_seed${SEED}}

CUDA_VISIBLE_DEVICES="${GPU}" python train_ibrl_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_ibrl_simdist.yaml \
    --seed "${SEED}" \
    --base_adapt_ckpt "${BASE_ADAPT_CKPT}" \
    --num_train_step "${STEPS}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
