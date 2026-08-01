#!/usr/bin/env bash
# Train IBRL on Threading_D0 from SCRATCH on the SOURCE (nominal) domain and
# dump the ENTIRE replay buffer (all transitions collected over the whole
# 200k-step run) to a single hdf5 file.
#
# Recipe (the "source IBRL" baseline):
#   - train + eval on the source/nominal env (no domain shift)
#   - IBRL hybrid action selection (q_agent.act_method=ibrl) with the flow-BC
#     checkpoint as the frozen BC prior
#   - robot init noise = joint-space gaussian sigma 0.1 (both train & eval)
#   - 200000 env steps
#
# WHY the big --replay_buffer_size: the rela replay is sized in EPISODES and
# evicts the OLDEST episode (FIFO) once full. To keep ALL 200k steps we set the
# capacity (20000 episodes) far above the number of episodes a 200k-step
# Threading run can produce (avg episode length is ~tens-to-hundreds of steps,
# so a few thousand episodes at most). train_rl.py warns if the buffer ever
# filled (i.e. some early episodes were evicted) so you'd know to raise it.
# The C++ buffer preallocates ~1.1*capacity*episode_length of CPU RAM; 20000
# episodes is a few GB here (system has ~295 GB free).
#
# The dump is written by train_rl.py's new --save_replay_buffer flag at the end
# of training, in the same per-episode hdf5 layout rl/replay.py uses elsewhere
# (data/demo_i/{obs/state, obs/prop, actions, rewards}).
#
# Usage:
#   bash scripts/collect_rl_source_replay_buffer_threading.sh
#   GPU=1 SEED=2 bash scripts/collect_rl_source_replay_buffer_threading.sh
#   bash scripts/collect_rl_source_replay_buffer_threading.sh --num_train_step 400000
#
# Env vars:
#   GPU        CUDA device id              (default 0)
#   TASK       task tag for run name       (default threading)
#   METHOD     method tag for run name     (default ibrl_source_replay)
#   SEED       training seed               (default 1)
#   BC_POLICY  flow-BC ckpt used as prior  (default threading_flow_bc_seed1/model0.pt)
#   SIGMA      joint-space init noise std  (default 0.1)
#   N_STEP     num env steps               (default 200000)
#   REPLAY_CAP replay capacity (episodes)  (default 20000)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-ibrl_source_replay}"
SEED="${SEED:-1}"
BC_POLICY="${BC_POLICY:-/home/Chenyang/ibrl/exps/mimicgen/threading_flow_bc_seed1/model0.pt}"
SIGMA="${SIGMA:-0.1}"
N_STEP="${N_STEP:-200000}"
REPLAY_CAP="${REPLAY_CAP:-20000}"

CONFIG="release/cfgs/mimicgen/threading_d0_state_naive.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"
REPLAY_OUT="${SAVE_DIR}/replay_buffer.h5"

# Notes on overrides:
#   --eval_domain_shift.mode none      eval on SOURCE (the naive yaml evals on
#                                      the target shift; we want source-source).
#   --state_stack 1                    match the flow-BC prior's obs space
#                                      (threading_flow_bc_seed1 uses state_stack=1;
#                                      the naive yaml defaults to 3).
#   --q_agent.act_method ibrl          IBRL hybrid (bootstrap_method follows it).
python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --num_train_step "$N_STEP" \
    --eval_domain_shift.mode none \
    --train_domain_shift.mode none \
    --state_stack 1 \
    --bc_policy "$BC_POLICY" \
    --q_agent.act_method ibrl \
    --train_hand_init_perturb_joint_sigma "$SIGMA" \
    --eval_hand_init_perturb_joint_sigma "$SIGMA" \
    --replay_buffer_size "$REPLAY_CAP" \
    --save_replay_buffer "$REPLAY_OUT" \
    "$@"

echo "[done] full replay buffer written to ${REPLAY_OUT}"
