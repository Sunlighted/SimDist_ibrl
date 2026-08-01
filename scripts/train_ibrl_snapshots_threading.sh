#!/usr/bin/env bash
# 方案 A · Step 1: retrain the SOURCE IBRL expert AND dump a quality-spectrum of
# policy checkpoints {random -> expert} during training (train_rl.py
# --snapshot_period), for SimDist diverse data generation (CLAUDE.md §17).
#
# Faithfully re-runs the existing source recipe by pointing --config_path at the
# existing source run's dumped cfg.yaml (same q_agent arch, hand_init), but uses
# the frozen SOURCE RL agent as the IBRL prior (--sgft_source_agent) instead of
# the cfg's flow-BC bc_policy. Adds --snapshot_period + a new save_dir.
#
# Snapshots land at $SAVE_DIR/snapshot_<step>.pt (alongside cfg.yaml so each is
# loadable). The final expert is $SAVE_DIR/model0.pt.
#
# Usage:
#   bash scripts/train_ibrl_snapshots_threading.sh
#   GPU=1 SEED=1 SNAP=12500 STEPS=200000 bash scripts/train_ibrl_snapshots_threading.sh
#
# Env vars:
#   GPU          CUDA device id                       (default 0)
#   SEED         training seed                        (default 1)
#   SNAP         snapshot every N train steps         (default 12500 -> 16 snapshots over 200k)
#   STEPS        total train steps                    (default 200000)
#   SRC_CFG      source recipe cfg.yaml to replicate  (default threading_ibrl_source_seed1)
#   PRIOR_AGENT  frozen BC head used as IBRL prior     (default threading_target_mlp_bc_n20_seed1/model0.pt)
#   SAVE_DIR     output dir                           (default exps/mimicgen/threading_ibrl_snapshots_seed<SEED>)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

SEED="${SEED:-1}"
SNAP="${SNAP:-12500}"
STEPS="${STEPS:-200000}"
SRC_CFG="${SRC_CFG:-exps/mimicgen/threading_ibrl_source_seed1/cfg.yaml}"
SAVE_DIR="${SAVE_DIR:-exps/mimicgen/threading_ibrl_snapshots_seed${SEED}}"
# IBRL prior = the frozen target n20 MLP-BC head (a BC policy, not a QAgent),
# wired via --bc_policy (the ONLY mechanism that loads a BC head as the IBRL
# prior; --sgft_source_agent expects an RL QAgent and would try to decode this
# ckpt's train_bc cfg.yaml against train_rl's MainConfig → crash). This overrides
# the source cfg's flow-BC bc_policy. No sgft shaping / sgft_source_agent here.
PRIOR_AGENT="${PRIOR_AGENT:-/home/Chenyang/ibrl/exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt}"

echo "replicating source recipe from: $SRC_CFG"
echo "save_dir=$SAVE_DIR  snapshot_period=$SNAP  steps=$STEPS"
echo "IBRL prior (bc_policy) = $PRIOR_AGENT"

python train_rl.py \
    --config_path "$SRC_CFG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --snapshot_period "$SNAP" \
    --num_train_step "$STEPS" \
    --bc_policy "$PRIOR_AGENT" \
    "$@"
