#!/bin/bash
# Iterated IBRL (tasks §32 style) on the Threading Robotiq cell, all on ONE GPU:
#   1. independent 300-episode strict evals of the iter1 candidates (model0.pt = TopkSaver max, snapshot_800000.pt);
#   2. the better one's HYBRID (max-Q over its RL actor + its BC-RNN prior) becomes the prior of a fresh 2M-step IBRL.
# Everything else = iter1 (threading_robotiq_state_ibrl.yaml, preload 1000, seed 1).
# Detach it (CLAUDE.md #11):
#   setsid env GPU=2 bash scripts/run_threading_robotiq_ibrl_iter2.sh > logs/ibrl_robotiq_iter2.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU:-2}" WANDB_MODE=offline
PY="${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python}"
DATA="${DATA:-/user_data/Chenyang/simdist_data/threading_robotiq_expert_n1000/dataset.hdf5}"
SRC=exps/sim2real/threading_robotiq_ibrl_preload1000_seed1
DIR=exps/sim2real/threading_robotiq_ibrl_iter2_seed1
mkdir -p logs

echo "=== [$(date '+%F %T')] evals ==="
for c in model0 snapshot_800000; do
    $PY evaluate/eval.py --weight "$SRC/$c.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
        --hand_init_perturb_joint_sigma 0.1 > "$SRC/eval300_$c.log" 2>&1 &
done
wait
best=""; best_s=-1
for c in model0 snapshot_800000; do
    s=$(grep -E '^score: ' "$SRC/eval300_$c.log" | tail -1 | sed -E 's/^score: ([0-9.]+).*/\1/')
    echo "$c: score=$s | $(grep -m1 'EVAL ENV' "$SRC/eval300_$c.log")"
    [ -n "$s" ] || { echo "FATAL: no score for $c"; exit 1; }
    if awk "BEGIN{exit !($s > $best_s)}"; then best=$c; best_s=$s; fi
done
echo "prior = $SRC/$best.pt (independent 300: $best_s)"

echo "=== [$(date '+%F %T')] iter2 -> $DIR ==="
$PY train_rl.py --config_path release/cfgs/sim2real/threading_robotiq_state_ibrl.yaml --save_dir "$DIR" \
    --bc_policy none --sgft_source_agent "$SRC/$best.pt" --source_agent_use_hybrid 1 --q_agent.sgft_shaping 0 \
    --preload_datapath "$DATA" --preload_num_data 1000 --num_train_step 2000000 --seed 1 \
    --snapshot_period 100000 --use_wb 0
echo "train_rl rc=$? [$(date '+%F %T')]"
