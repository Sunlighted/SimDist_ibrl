#!/usr/bin/env bash
# IBRL on the NEW "dynamics-only" Threading_D0 TARGET, warm-started from a
# source-domain IBRL checkpoint used as the base.
#
# Target dynamics gap (NO hole_offset => pure dynamics, no perceptual gap):
#   friction x0.5, mass x3, gripper_kp x0.15, ring_size x0.85, ring_friction x1.5,
#   solref/solimp pinned at 1.0, strict_success ON (dwell 3 + ring-disp 0.04).
# The frozen source policy scores ~13% strict success here (vs 98% on source);
# the grasp phase is feasible (friction-cone margin ~10x), so this is a
# learnable dynamics gap. See release/cfgs/mimicgen/threading_d0_state_ibrl_newtarget.yaml.
#
# "Base" = the source IBRL checkpoint warm-starts THIS run's agent via
# --load_pretrained_agent. LOAD_POLICY_ONLY controls how much transfers:
#   0 (default) : warm-start BOTH actor + critic  (full "base" — fastest start,
#                 but the source critic's Q was learned on source dynamics)
#   1           : warm-start the ACTOR/policy only; critic re-initialized
#                 (safer if the source critic mistransfers to new dynamics)
#
# The IBRL BC prior (bc_policy in the YAML) is the source-domain flow BC — IBRL
# competes the RL actor against it. That is SEPARATE from the warm-start base.
#
# Usage:
#   bash scripts/train_rl_ibrl_newtarget_threading.sh
#   GPU=1 SEED=2 bash scripts/train_rl_ibrl_newtarget_threading.sh
#   LOAD_POLICY_ONLY=1 bash scripts/train_rl_ibrl_newtarget_threading.sh
#   BASE_AGENT=exps/.../latest.pt bash scripts/train_rl_ibrl_newtarget_threading.sh
#   bash scripts/train_rl_ibrl_newtarget_threading.sh --num_train_step 200000
#
# Env vars:
#   GPU              CUDA device id                       (default 0)
#   TASK             task tag for run name                (default threading)
#   METHOD           method tag for run name              (default ibrl_newtgt)
#   SEED             training seed                        (default 1)
#   BASE_AGENT       source IBRL ckpt to warm-start from  (default exps/mimicgen/threading_ibrl_source_seed${SEED}/latest.pt)
#   LOAD_POLICY_ONLY 0=warm-start actor+critic, 1=actor only  (default 0)
#
# Eval the trained checkpoint (target + source, strict-gated):
#   python evaluate/eval.py --weight exps/mimicgen/threading_ibrl_newtgt_seed1/latest.pt \
#       --mode rl --num_game 100 --shift_from_cfg 1 --strict_success 1   # target
#   python evaluate/eval.py --weight exps/mimicgen/threading_ibrl_newtgt_seed1/latest.pt \
#       --mode rl --num_game 100 --shift_mode none --strict_success 1     # source

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-ibrl_newtgt}"
SEED="${SEED:-1}"
LOAD_POLICY_ONLY="${LOAD_POLICY_ONLY:-0}"
BASE_AGENT="${BASE_AGENT:-exps/mimicgen/threading_ibrl_source_seed${SEED}/latest.pt}"

if [[ ! -f "$BASE_AGENT" ]]; then
  echo "ERROR: warm-start base checkpoint not found:" >&2
  echo "    $BASE_AGENT" >&2
  echo "Pass BASE_AGENT=/path/to/source/latest.pt or train a source agent first." >&2
  exit 1
fi

CONFIG="release/cfgs/mimicgen/threading_d0_state_ibrl_newtarget.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "warm-start base   : $BASE_AGENT"
echo "load_policy_only  : $LOAD_POLICY_ONLY  (0=actor+critic, 1=actor only)"
echo "config            : $CONFIG"
echo "save_dir          : $SAVE_DIR"

python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --load_pretrained_agent "$BASE_AGENT" \
    --load_policy_only "$LOAD_POLICY_ONLY" \
    "$@"
