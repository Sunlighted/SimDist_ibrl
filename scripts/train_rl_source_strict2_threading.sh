#!/usr/bin/env bash
# Reproduce threading_ibrl_source_strict_seed1 with the CURRENT code, log to
# wandb, save to ...strict2_seed1. Purpose: test the "RL algo got broken
# (DDPG -> TD3)" hypothesis. It loads the ORIGINAL run's saved cfg.yaml
# verbatim (strict ON, D0 native narrow reset = init_dist OFF, act=ibrl,
# D0 flow-BC prior, 400k steps), so the ONLY difference from the original
# run is the code version. If D0 still takes off to ~0.9, the algo is fine
# and the D0-vs-D1 gap is purely setup (reset distribution).
#
# Usage (background + wandb; survives terminal close):
#   nohup bash scripts/run_strict2_threading.sh > exps_strict2.log 2>&1 &
#   disown
#   # then watch:
#   grep -aoE '^[0-9]+: score/score +: [0-9.]+' \
#     exps/mimicgen/threading_ibrl_source_strict2_seed1/train.log
#
# Env vars:
#   GPU   CUDA device id   (default 3)
#
# NOTE: if a previous strict2 attempt is still running, kill it first and
#   rm -rf exps/mimicgen/threading_ibrl_source_strict2_seed1
# (this script does NOT delete it for you, to avoid clobbering a good run).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-3}"
export MUJOCO_GL=egl

SRC_CFG="exps/mimicgen/threading_ibrl_source_strict_seed1/cfg.yaml"
SAVE_DIR="exps/mimicgen/threading_ibrl_source_strict2_seed1"

if [ ! -f "$SRC_CFG" ]; then
  echo "ERROR: original cfg not found: $SRC_CFG" >&2
  exit 1
fi
if [ -d "$SAVE_DIR" ]; then
  echo "WARNING: $SAVE_DIR already exists; train_rl will write into it." >&2
  echo "         rm -rf it first if you want a clean run." >&2
fi

# wb_exp / wb_run / wb_group are NOT CLI args -- train_rl derives them from
# save_dir (-> project "mimicgen", run "..._strict2_seed1", group
# "..._strict2"), so overriding save_dir is enough for wandb naming.
# exec so python replaces this shell (clean signal handling under nohup).
exec python train_rl.py \
    --config_path "$SRC_CFG" \
    --save_dir "$SAVE_DIR" \
    --use_wb 1 \
    "$@"
