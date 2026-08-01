#!/bin/bash
# Residual off-policy RL (ResFiT-style) on a frozen ADAPTED SimDist base, for
# Threading_D0. The adapted SimDist world model (threading_simdist_adapt_seed1,
# §24) plateaued at TARGET strict-success ≈ 0.75 (MPPI); a small residual RL
# policy continues to improve it. base a_base(s) via MPPI (frozen) + bounded
# residual, executed clip(base+delta,-1,1), TD3 off-policy RL. RLPD recipe: 50/50
# online+offline mix, LayerNorm REDQ critic, base+noise warmup, critic-only
# warmup, TD3 smoothing. Train + eval in the TARGET domain. The MPPI base action
# is computed once/step and stored in the buffer (n-step via rela).
# See rl/residual_simdist_agent.py + train_residual_simdist.py + CLAUDE.md §25.
#
# Env vars:
#   SEED             random seed (default 1)
#   BASE_ADAPT_CKPT  adapted SimDist ckpt (default threading_simdist_adapt_seed1/model0.pt)
#   GPU              CUDA_VISIBLE_DEVICES (default 0)
#   STEPS            num_train_step (default 100000)
#   UTD              num_critic_update per env step (default 4)
#   Any extra args after the script are forwarded to train_residual_simdist.py.
#
# Examples:
#   bash scripts/train_residual_simdist_threading.sh
#   GPU=1 SEED=1 bash scripts/train_residual_simdist_threading.sh
#
# Standalone re-eval of a saved ckpt on the TARGET:
#   python evaluate/eval.py --weight exps/mimicgen/threading_residual_simdist_seed1/latest.pt \
#       --mode residual_simdist --num_game 100 --strict_success 1
set -e

SEED=${SEED:-1}
BASE_ADAPT_CKPT=${BASE_ADAPT_CKPT:-exps/mimicgen/threading_simdist_adapt_seed1/model0.pt}
GPU=${GPU:-0}
STEPS=${STEPS:-100000}
UTD=${UTD:-4}
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_residual_simdist_seed${SEED}}

CUDA_VISIBLE_DEVICES="${GPU}" python train_residual_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_residual_simdist.yaml \
    --seed "${SEED}" \
    --base_adapt_ckpt "${BASE_ADAPT_CKPT}" \
    --num_critic_update "${UTD}" \
    --num_train_step "${STEPS}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
