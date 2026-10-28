#!/bin/bash
# BC-RNN+GMM prior on the fixed Robotiq cell (2026-10-21, README "10-21 gripper contact"): the stage-2 recipe
# (release/cfgs/sim2real/threading_robotiq_state_bc_rnn.yaml) on threading_robotiq_expert_n1000_v3, then independent
# 300-episode strict evals of both kept checkpoints (model0 = best in-training eval, model1 = the one train_bc returns).
#   (CFG= / DIR= / DATA= / EXTRA="--num_epoch 6000 ..." override: e.g. the single-step prior threading_robotiq_state_bc_rnn_tp1.yaml)
#   setsid env GPU=2 bash scripts/run_threading_robotiq_bc_v3.sh > logs/bc_robotiq_v3.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU:-2}" WANDB_MODE=offline
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
DATA="${DATA:-/user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v5/dataset.hdf5}"
CELL="${CELL:-real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml}"
CFG="${CFG:-release/cfgs/sim2real/threading_robotiq_state_bc_rnn.yaml}"
DIR="${DIR:-exps/sim2real/threading_robotiq_rnn_bc_v5_seed1}"
echo "=== [$(date '+%F %T')] BC -> $DIR ==="
$PY train_bc.py --config_path "$CFG" --save_dir "$DIR" --seed "${SEED:-1}" \
    --use_wb 0 --dataset.path "$DATA" ${EXTRA:-}
echo "train_bc rc=$? (1 is normal: train_bc.py ends in assert False)"
grep -q "best_ckpt_score" "$DIR/train.log" || { echo "FATAL: BC did not finish"; exit 1; }
grep -m1 "dataset" "$DIR/cfg.yaml"; grep -m1 "path:" "$DIR/cfg.yaml"
for c in model0 model1; do
    [ -s "$DIR/$c.pt" ] || continue
    $PY evaluate/eval.py --weight "$DIR/$c.pt" --mode bc --num_game 300 --mp 10 --strict_success 1 \
        --hand_init_perturb_joint_sigma 0.1 --sim2real_cell "$CELL" > "$DIR/eval300_$c.log" 2>&1
    echo "$c: $(grep -E '^score: ' "$DIR/eval300_$c.log" | tail -1) | $(grep -m1 'EVAL ENV' "$DIR/eval300_$c.log")"
done
echo "=== [$(date '+%F %T')] done ==="
