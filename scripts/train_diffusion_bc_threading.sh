#!/usr/bin/env bash
# Train state-based Diffusion Policy on MimicGen Threading_D0 (DDIM scheduler,
# ConditionalUnet1D borrowed from robomimic, Tp=16/Ta=8 chunks, EMA).
#
# wandb naming follows the repo convention (common_utils/py/run_cfg.py):
#   save_dir = exps/mimicgen/{task}_{method}_seed{seed}
#       → wandb project = "mimicgen"
#       → wandb run     = "{task}_{method}_seed{seed}"
#       → wandb group   = "{task}_{method}"   (the "seed*" segment is stripped)
#
# Usage:
#   bash scripts/train_diffusion_bc_threading.sh
#   GPU=1 SEED=2 bash scripts/train_diffusion_bc_threading.sh
#   bash scripts/train_diffusion_bc_threading.sh --use_wb 0     # extra overrides
#
# Env vars:
#   GPU     CUDA device id (default 0)
#   TASK    task tag for run name (default threading)
#   METHOD  method tag for run name (default diffusion_bc)
#   SEED    training seed (default 1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-diffusion_bc}"
SEED="${SEED:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_diffusion_bc.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    "$@"
