#!/usr/bin/env bash
# Target-only single-step MLP BC on Threading_D0 — the SAME simple network the
# H2O+ warm-up uses (StateBcPolicy / FcActor: 3×1024, single action per step,
# MSE, fully closed-loop). Train on ONLY the target demos collected under the
# new dynamics-only target (release/data/mimicgen/threading_d0_target_dynamics),
# eval on the same target with a WIDER arm-init noise (joint σ=0.1, vs the demos'
# σ=0.02) to stress robustness to start-pose variation.
#
# Why this exists: on 20 demos the simple deterministic MLP regression hit ~40%
# target (pure actor), far above flow/RNN+GMM (~6-8%) — this isolates that head
# as a clean standalone BC baseline and checks the 20- vs 10-demo data scaling.
#
# Env vars:
#   GPU     CUDA device id              (default 0)
#   N_DATA  number of target demos (20 or 10; default 20)
#   SEED    training seed               (default 1)
#   TASK    task tag for run name       (default threading)
#   METHOD  method tag (default target_mlp_bc_n${N_DATA})
#
# Usage:
#   bash scripts/train_bc_mlp_threading_target.sh                 # 20 demos
#   N_DATA=10 bash scripts/train_bc_mlp_threading_target.sh       # 10 demos
#   GPU=1 SEED=2 N_DATA=10 bash scripts/train_bc_mlp_threading_target.sh
#   bash scripts/train_bc_mlp_threading_target.sh --state_policy.dropout 0.5  # overrides
#
# Re-eval a saved ckpt later (target, σ=0.1):
#   python evaluate/eval.py --weight exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt \
#       --mode bc --num_game 100 --shift_from_cfg 1 --hand_init_perturb_joint_sigma 0.1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
N_DATA="${N_DATA:-20}"
SEED="${SEED:-1}"
METHOD="${METHOD:-target_mlp_bc_n${N_DATA}}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_mlp_bc_target.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "target-only single-step MLP BC: ${N_DATA} target demos (new dynamics-only)"
echo "eval: target + joint-init σ=0.1 | save_dir: ${SAVE_DIR}"

python train_bc.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --dataset.num_data "$N_DATA" \
    --save_dir "$SAVE_DIR" \
    "$@"
