#!/usr/bin/env bash
# Target-only BC-RNN+GMM on Threading_D0: train on ONLY the 20 demos collected
# under the new dynamics-only target (release/data/mimicgen/
# threading_d0_target_dynamics), eval on the same target. The BC-RNN counterpart
# of scripts/train_flow_bc_threading_target.sh (same data, different head) — for
# the small-data target-only comparison (RNN+GMM vs flow vs the MLP warm-up).
#
# In-training eval is on target (`eval_target_during_training=1`), so the topk
# saver picks model0/1.pt by target success. After training, train_bc auto-runs
# dual eval (source + target) and reports both + the gap.
#
# Usage:
#   bash scripts/train_bc_rnn_threading_target.sh
#   GPU=1 SEED=2 bash scripts/train_bc_rnn_threading_target.sh
#   bash scripts/train_bc_rnn_threading_target.sh --num_epoch 1000   # CLI overrides
#
# Env vars:
#   GPU     CUDA device id              (default 0)
#   TASK    task tag for run name       (default threading)
#   METHOD  method tag for run name     (default target_only_bc_rnn)
#   SEED    training seed               (default 1)
#
# Re-eval a saved ckpt later:
#   python evaluate/eval.py --weight exps/mimicgen/threading_target_only_bc_rnn_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_from_cfg 1   # target (new dynamics-only)
#   python evaluate/eval.py --weight exps/mimicgen/threading_target_only_bc_rnn_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_mode none     # source

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-target_only_bc_rnn}"
SEED="${SEED:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_bc_rnn_target.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "target-only BC-RNN+GMM: 20 target demos (new dynamics-only)"
echo "save_dir: ${SAVE_DIR}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    "$@"
