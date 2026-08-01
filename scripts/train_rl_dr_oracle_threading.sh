#!/usr/bin/env bash
# Train RL on Threading_D0 with ORACLE Domain Randomization — the DR ranges
# are widened to bracket the target shift values, so the target lies *inside*
# the training distribution. Acts as an empirical ceiling on what DR can
# achieve when there is no sim-to-real gap, since the policy has already
# seen target-like states during training.
#
# Compared to the blind DR baseline (scripts/train_rl_dr_threading.sh):
#   blind DR : friction∈[0.7, 1.3]  mass∈[0.7, 1.5]  solref∈[0.7, 1.5]
#              solimp∈[0.85, 1.15] gripper_kp=1.0   hole_offset=0
#   oracle   : friction∈[0.2, 1.3]  mass∈[0.7, 3.5]  solref∈[0.7, 3.5]
#              solimp∈[0.6, 1.2]   gripper_kp∈[0.4, 1.2]  hole_offset∈[0, 0.01]
#
# Eval is on the fixed target shift (same as blind DR), so scores are
# directly comparable. Expectation: oracle ≥ blind DR ≥ source baseline.
#
# Usage:
#   bash scripts/train_rl_dr_oracle_threading.sh
#   GPU=1 SEED=2 bash scripts/train_rl_dr_oracle_threading.sh
#   METHOD=ibrl_dr_oracle bash scripts/train_rl_dr_oracle_threading.sh --bc_policy /home/Chenyang/ibrl/exps/mimicgen/threading_flow_bc_seed1/model0.pt --q_agent.act_method ibrl --state_stack 1 --num_train_step 400000
#
# Env vars:
#   GPU     CUDA device id            (default 0)
#   TASK    task tag for run name     (default threading)
#   METHOD  method tag for run name   (default rl_dr_oracle)
#   SEED    training seed             (default 1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-rl_dr_oracle}"
SEED="${SEED:-1}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_dr_oracle.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    "$@"
