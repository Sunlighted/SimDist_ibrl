#!/bin/bash
# One arm of the IBRL experiments in real2sim/zed_capture/calib_20261015/IBRL_EXPERIMENTS.md: train (1M steps) ->
# independent 300-episode evals (own init range + the full cell range). Same BC prior and demos for every arm.
#   ARM=e1_plain SEED=2 GPU=0 bash scripts/run_ibrl_robotiq_arm.sh
#   ARM=e2_failreplay SEED=1 GPU=0 EXTRA="--fail_replay_prob 0.5" bash scripts/run_ibrl_robotiq_arm.sh
#   ARM=e3_scale025 SEED=1 GPU=1 EXTRA="--sim2real_overrides layout_scale=0.25" bash scripts/run_ibrl_robotiq_arm.sh
# Detach it (CLAUDE.md #11):  setsid bash scripts/run_ibrl_robotiq_arm.sh > logs/<arm>.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
: "${ARM:?set ARM}"; SEED="${SEED:-1}"; EXTRA="${EXTRA:-}"
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU:-0}" WANDB_MODE=offline
PY="${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python}"
# 2026-10-23: defaults = the v5 data and prior (cell with the 10-21 / 10-22 physics fixes; all older data deleted)
DATA="${DATA:-/user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v5/dataset.hdf5}"
BC="${BC:-exps/sim2real/threading_robotiq_rnn_bc_n3000v5_h1000_ema_ep6000_seed1/model0.pt}"
CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml
DIR=exps/sim2real/ibrl_robotiq_${TAG:-v3}_${ARM}_seed${SEED}
[ -s "$DATA" ] || { echo "FATAL: no demos at $DATA (IBRL_EXPERIMENTS.md §1: copy or regenerate)"; exit 1; }
[ -s "$BC" ] || { echo "FATAL: no BC prior $BC (git pull)"; exit 1; }
mkdir -p logs exps/sim2real
echo "=== [$(date '+%F %T')] $ARM seed $SEED: $EXTRA -> $DIR ==="
# shellcheck disable=SC2086
$PY train_rl.py --config_path release/cfgs/sim2real/threading_robotiq_state_ibrl.yaml --save_dir "$DIR" \
    --bc_policy "$BC" --preload_datapath "$DATA" --preload_num_data 1000 --seed "$SEED" \
    --snapshot_period 100000 --use_wb 0 $EXTRA
echo "train_rl rc=$?"
[ -s "$DIR/model0.pt" ] || { echo "FATAL: no $DIR/model0.pt"; exit 1; }
# (a) the arm's own init range (eval env rebuilt from its cfg, overrides included)
$PY evaluate/eval.py --weight "$DIR/model0.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
    --hand_init_perturb_joint_sigma 0.1 > "$DIR/eval300_own.log" 2>&1
# (b) the FULL cell range (--sim2real_cell forces the plain cell: no overrides)
$PY evaluate/eval.py --weight "$DIR/model0.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
    --hand_init_perturb_joint_sigma 0.1 --sim2real_cell "$CELL" > "$DIR/eval300_full.log" 2>&1
for f in own full; do echo "$f: $(grep -E '^score: ' $DIR/eval300_$f.log | tail -1) | $(grep -m1 'EVAL ENV' $DIR/eval300_$f.log)"; done
echo "=== [$(date '+%F %T')] done ==="
