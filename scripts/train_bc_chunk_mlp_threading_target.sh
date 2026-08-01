#!/usr/bin/env bash
# Target-only CHUNKED deterministic-MLP BC on Threading_D0 — same simple MLP as
# train_bc_mlp_threading_target.sh, but predicting a Tp=16 action chunk and
# executing Ta=8 open-loop. Isolates the effect of action chunking (vs the
# single-step closed-loop MLP), model class held fixed. Eval: target + σ=0.1 +
# strict, lr=1e-4 — all identical to the single-step MLP run.
#
# Env vars:
#   GPU     CUDA device id              (default 0)
#   N_DATA  number of target demos (20 or 10; default 20)
#   SEED    training seed               (default 1)
#   TA      action_horizon Ta executed open-loop (default 8; set 1 for closed-loop)
#   METHOD  method tag (default target_chunk_mlp_n${N_DATA})
#
# Usage:
#   bash scripts/train_bc_chunk_mlp_threading_target.sh
#   N_DATA=10 bash scripts/train_bc_chunk_mlp_threading_target.sh
#   GPU=1 TA=4 bash scripts/train_bc_chunk_mlp_threading_target.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
N_DATA="${N_DATA:-20}"
SEED="${SEED:-1}"
TA="${TA:-8}"
METHOD="${METHOD:-target_chunk_mlp_n${N_DATA}}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_chunk_mlp_bc_target.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "target-only CHUNKED MLP BC: ${N_DATA} demos, Tp=16 Ta=${TA} | eval target σ=0.1 strict"
echo "save_dir: ${SAVE_DIR}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --dataset.num_data "$N_DATA" \
    --chunk_mlp_policy.action_horizon "$TA" \
    --save_dir "$SAVE_DIR" \
    "$@"
