#!/usr/bin/env bash
# IL co-training flow BC on Threading_D0: a LOT of source data + a LITTLE
# target data, mixed at a fixed per-batch ratio and fed to the flow policy.
# Default 95% source / 5% target per batch, drawn from 2000 source + 20 target
# episodes (2026-05-28 re-collection). See CLAUDE.md §11 / §12.
#
# Data (collect_target_demos.py format):
#   source = release/data/mimicgen/threading_d0_source_sigma0.1   (no shift, 2000)
#   target = release/data/mimicgen/threading_d0_target_dynamics   (new dynamics-only, 20)
# Both are baked into the config; override paths/ratio via env vars or CLI.
#
# Usage:
#   bash scripts/train_cotrain_flow_bc_threading.sh
#   GPU=1 SEED=2 bash scripts/train_cotrain_flow_bc_threading.sh
#   RATIO=0.90 N_TGT=40 bash scripts/train_cotrain_flow_bc_threading.sh   # 90/10 mix
#   RATIO=1.00 METHOD=source_only bash scripts/train_cotrain_flow_bc_threading.sh  # source-only ablation
#   bash scripts/train_cotrain_flow_bc_threading.sh --num_epoch 1000      # extra CLI overrides
#
# Env vars:
#   GPU     CUDA device id              (default 0)
#   TASK    task tag for run name       (default threading)
#   METHOD  method tag for run name     (default cotrain_flow)
#   SEED    training seed               (default 1)
#   RATIO   per-batch source fraction   (default 0.95)
#   N_SRC   source episodes             (default 2000)
#   N_TGT   target episodes             (default 20)
#
# After training, train_bc auto-runs dual eval (source + target) and reports
# the gap. The target eval is the new dynamics-only target the 20 target demos
# were collected on. To re-eval a saved ckpt later:
#   python evaluate/eval.py --weight exps/mimicgen/threading_cotrain_flow_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_from_cfg 1   # target (YAML's eval shift)
#   python evaluate/eval.py --weight exps/mimicgen/threading_cotrain_flow_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_mode none     # source (no shift)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-cotrain_flow}"
SEED="${SEED:-1}"
RATIO="${RATIO:-0.95}"
N_SRC="${N_SRC:-2000}"
N_TGT="${N_TGT:-20}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "mix: ${RATIO} source / $(python -c "print(round(1-${RATIO},4))") target"
echo "episodes: ${N_SRC} source + ${N_TGT} target"
echo "save_dir: ${SAVE_DIR}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --dataset.num_data "$N_SRC" \
    --dataset.cotrain_num_data "$N_TGT" \
    --dataset.cotrain_ratio "$RATIO" \
    "$@"
