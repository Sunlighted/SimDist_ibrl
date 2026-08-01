#!/usr/bin/env bash
# Train RL on Threading_D0 from source-domain data only (no DR, no shift).
# In-training eval uses the same source env. To also get target eval, run
# evaluate/eval.py with --shift_from_cfg 1 on the saved ckpt afterwards.
#
# Uses the "naive sim2real" config (threading_d0_state_naive.yaml): train
# on nominal env, eval-time domain_shift is overridden to "none" here so
# eval is also on source. That makes this the "pure source RL" baseline
# you'd compare DR / DR-oracle / IBRL against.
#
# Usage:
#   bash scripts/train_rl_source_threading.sh
#   GPU=1 SEED=2 bash scripts/train_rl_source_threading.sh
#   bash scripts/train_rl_source_threading.sh --num_train_step 400000
#   METHOD=ibrl_source bash scripts/train_rl_source_threading.sh --bc_policy /home/Chenyang/ibrl/exps/mimicgen/threading_flow_bc_seed1/model0.pt --q_agent.act_method ibrl --state_stack 1 --train_hand_init_perturb_joint_sigma 0.1 --eval_hand_init_perturb_joint_sigma 0.1 --num_train_step 400000
#
# Env vars:
#   GPU     CUDA device id            (default 0)
#   TASK    task tag for run name     (default threading)
#   METHOD  method tag for run name   (default rl_source)
#   SEED    training seed             (default 1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-rl_source}"
SEED="${SEED:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_naive.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

# Override eval_domain_shift.mode = "none" so eval also runs on source.
# (The YAML has eval_domain_shift.mode="fixed" for the naive sim2real
# baseline; we want source-source here.)
python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --eval_domain_shift.mode none \
    "$@"
