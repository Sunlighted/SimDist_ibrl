#!/bin/bash
# One IBRL arm on the peg expert-demo prior + final physical check.
# The prior is the gated BC from scripts/pipeline_peg_expert.sh (independent n=100 = 0.24).
#
#   ARM=preload1000 GPU=1 EXTRA="--preload_num_data 1000" \
#     setsid bash scripts/run_peg_ibrl_arm.sh > logs/peg_ibrl_preload1000.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
ARM="${ARM:?set ARM}"
EXTRA="${EXTRA:-}"

RL_CFG=release/cfgs/mimicgen/peg_insertion_state_ibrl_source_expert_hard.yaml
BC=exps/mimicgen/peg_source_mlp_bc_expert_n1000_board_hard_seed1/model0.pt
RL_DIR=exps/mimicgen/peg_ibrl_source_expert_board_hard_${ARM}_seed1

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }
[ -s "$BC" ] || { echo "FATAL: no BC prior $BC"; exit 1; }

stamp "IBRL arm=$ARM extra=[$EXTRA] gpu=$GPU"
$PY train_rl.py --config_path "$RL_CFG" --save_dir "$RL_DIR" \
    --bc_policy "$BC" --seed 1 --snapshot_period 25000 --use_wb 0 $EXTRA
echo "train_rl rc=$?"
echo "snapshots: $(ls "$RL_DIR"/snapshot_*.pt 2>/dev/null | wc -l)"
grep -E "^ *[0-9]+: score/score " "$RL_DIR/train.log" 2>/dev/null | awk '{printf "%s %s  ", $1, $4} END {print ""}'

stamp "verify model0 (n=100)"
$PY scripts/verify_peg_ibrl.py --weight "$RL_DIR/model0.pt" --n 100 \
    --out "$RL_DIR/verify_model0_n100.npz" 2>&1 | grep -E "^\[env\]|^\[load_model\]|^success|^penetr|^max insertion|^tilt|^tip|^actor"
stamp "done"
