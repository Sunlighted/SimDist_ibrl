#!/usr/bin/env bash
# CFG co-training flow BC on Threading_D0: same source+target mix as
# scripts/train_cotrain_flow_bc_threading.sh, but with the CFG arm of CFG-ADDA
# turned on (Lei et al. 2026, "Science of Co-Training"). A one-hot source/
# target label is concatenated to the obs encoder output, dropouted to ∅
# during training, and at inference the velocity is interpolated as
# `(1+λ)·v(c) − λ·v(∅)`. λ<0 transfers "averaged across domains" knowledge
# (paper default −0.5). See CLAUDE.md §13.
#
# ADDA is intentionally skipped for state-based obs (no visual nuisance
# variation; source/target share the same 37-dim obs space). Revisit when
# we move to pixels.
#
# Data: identical to the non-CFG cotrain script.
#   source = release/data/mimicgen/threading_d0_source_sigma0.1   (no shift, 2000)
#   target = release/data/mimicgen/threading_d0_target_dynamics   (new dynamics-only, 20)
#
# Usage:
#   bash scripts/train_cotrain_cfg_flow_bc_threading.sh
#   GPU=1 SEED=2 bash scripts/train_cotrain_cfg_flow_bc_threading.sh
#   LAMBDA=-1.0 bash scripts/train_cotrain_cfg_flow_bc_threading.sh   # sweep guidance
#   INFER_DOMAIN=0 bash scripts/train_cotrain_cfg_flow_bc_threading.sh # eval as source
#   RATIO=0.90 N_TGT=40 bash scripts/train_cotrain_cfg_flow_bc_threading.sh  # 90/10 mix
#   bash scripts/train_cotrain_cfg_flow_bc_threading.sh --num_epoch 1000     # extra CLI overrides
#
# Env vars:
#   GPU           CUDA device id                          (default 0)
#   TASK          task tag for run name                   (default threading)
#   METHOD        method tag for run name                 (default cotrain_cfg_flow)
#   SEED          training seed                           (default 1)
#   RATIO         per-batch source fraction               (default 0.95)
#   N_SRC         source episodes                         (default 2000)
#   N_TGT         target episodes                         (default 20)
#   DROPOUT       p(c → ∅) during training                (default 0.2)
#   LAMBDA        CFG guidance scale at inference         (default -0.5)
#   INFER_DOMAIN  which one-hot to condition on at eval   (default 1 = target)
#
# After training, train_bc auto-runs dual eval (source + target) and reports
# the gap. To re-eval a saved ckpt later (the same path as non-CFG cotrain):
#   python evaluate/eval.py --weight exps/mimicgen/threading_cotrain_cfg_flow_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_from_cfg 1
#
# Ablation against plain cotrain (no CFG):
#   bash scripts/train_cotrain_flow_bc_threading.sh                 # CFG off
#   bash scripts/train_cotrain_cfg_flow_bc_threading.sh             # CFG on (this)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-cotrain_cfg_flow}"
SEED="${SEED:-1}"
RATIO="${RATIO:-0.95}"
N_SRC="${N_SRC:-2000}"
N_TGT="${N_TGT:-20}"
DROPOUT="${DROPOUT:-0.2}"
LAMBDA="${LAMBDA:--0.5}"
INFER_DOMAIN="${INFER_DOMAIN:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_cotrain_flow_bc.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "mix: ${RATIO} source / $(python -c "print(round(1-${RATIO},4))") target"
echo "episodes: ${N_SRC} source + ${N_TGT} target"
echo "CFG: dropout_p=${DROPOUT}  lambda=${LAMBDA}  infer_domain_id=${INFER_DOMAIN}"
echo "save_dir: ${SAVE_DIR}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --dataset.num_data "$N_SRC" \
    --dataset.cotrain_num_data "$N_TGT" \
    --dataset.cotrain_ratio "$RATIO" \
    --flow_policy.cfg_enabled 1 \
    --flow_policy.cfg_dropout_p "$DROPOUT" \
    --flow_policy.cfg_guidance_lambda "$LAMBDA" \
    --flow_policy.cfg_inference_domain_id "$INFER_DOMAIN" \
    "$@"
