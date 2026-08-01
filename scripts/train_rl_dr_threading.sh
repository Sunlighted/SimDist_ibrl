#!/usr/bin/env bash
# Train RL on Threading_D0 with Domain Randomization (sgft style — narrow
# ranges centered on 1.0, target values intentionally OUTSIDE the ranges).
# Eval is on the fixed target shift, so the metric is "DR sim-to-real gap"
# vs the source baseline.
#
# This is the BLIND DR baseline: friction/mass/solref/solimp are sampled per
# reset but gripper_kp and hole_offset stay at nominal (sgft's "unknown
# gap" dims). For the oracle variant that includes those, see
# scripts/train_rl_dr_oracle_threading.sh.
#
# Usage:
#   bash scripts/train_rl_dr_threading.sh
#   GPU=1 SEED=2 bash scripts/train_rl_dr_threading.sh
#   bash scripts/train_rl_dr_threading.sh --num_train_step 400000
#   METHOD=ibrl_dr bash scripts/train_rl_dr_threading.sh --bc_policy /home/Chenyang/ibrl/exps/mimicgen/threading_flow_bc_seed1/model0.pt --q_agent.act_method ibrl --state_stack 1 --num_train_step 400000
#
# Env vars:
#   GPU     CUDA device id            (default 0)
#   TASK    task tag for run name     (default threading)
#   METHOD  method tag for run name   (default rl_dr)
#   SEED    training seed             (default 1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-rl_dr}"
SEED="${SEED:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_dr.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    "$@"
