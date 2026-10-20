#!/bin/bash
# TRAIN_PLAN stage 3, attempt 2 (2026-10-19): the 1M-step run with preload_num_data 10 ended at 0.257 independent
# (below its own BC prior 0.34; 0.00 for the first 375k steps). The ONE change: preload all 1000 scripted-expert demos
# into the demo replay -- the Peg pipeline's identical recipe went 0.01 (preload 10) -> 0.92 (preload 1000).
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU:-2}" WANDB_MODE=offline
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml
RL_DIR=exps/sim2real/threading_robotiq_ibrl_preload1000_seed1
$PY train_rl.py --config_path release/cfgs/sim2real/threading_robotiq_state_ibrl.yaml --save_dir "$RL_DIR" \
    --bc_policy exps/sim2real/threading_robotiq_rnn_bc_seed1/model0.pt --preload_num_data 1000 --seed 1 \
    --snapshot_period 100000 --use_wb 0
echo "train_rl rc=$?"
$PY evaluate/eval.py --weight "$RL_DIR/model0.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
    --hand_init_perturb_joint_sigma 0.1 --sim2real_cell "$CELL" > "$RL_DIR/eval300.log" 2>&1
grep -E "EVAL ENV|success rate" "$RL_DIR/eval300.log" | tail -2
echo "=== done ==="
