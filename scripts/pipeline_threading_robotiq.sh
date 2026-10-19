#!/bin/bash
# TRAIN_PLAN.md (calib_20261015) stages 2-3 on the Threading ROBOTIQ real cell, local, one GPU:
#   0. wait for the scripted-expert demos (scripts/collect_threading_robotiq_demos.py, merged dataset.hdf5)
#   1. BC-RNN+GMM prior                 (release/cfgs/sim2real/threading_robotiq_state_bc_rnn.yaml)
#   2. GATE: independent BC eval, 300 episodes, strict, sigma 0.1, in the cell  -> >= 0.30 or stop
#   3. IBRL state expert, 1M steps      (release/cfgs/sim2real/threading_robotiq_state_ibrl.yaml)
#   4. independent IBRL eval, 300 episodes (TRAIN_PLAN gate >= 0.9)
# Launch detached (CLAUDE.md #11):
#   setsid bash scripts/pipeline_threading_robotiq.sh > logs/pipeline_threading_robotiq.log 2>&1 < /dev/null & disown
# Each step checks the previous step's ARTIFACT, not its exit code (train_bc.py exits 1 on success).
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export WANDB_MODE=offline
PY=/home/Chenyang/.conda/envs/ibrl/bin/python

DATA_DIR=/user_data/Chenyang/simdist_data/threading_robotiq_expert_n1000
DATA=$DATA_DIR/dataset.hdf5
CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml
BC_CFG=release/cfgs/sim2real/threading_robotiq_state_bc_rnn.yaml
BC_DIR=exps/sim2real/threading_robotiq_rnn_bc_seed1
RL_CFG=release/cfgs/sim2real/threading_robotiq_state_ibrl.yaml
RL_DIR=exps/sim2real/threading_robotiq_ibrl_seed1
BC_GATE="${BC_GATE:-0.30}"

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }

# ---- 0. demos -------------------------------------------------------------
stamp "0. waiting for $DATA"
until [ -f "$DATA_DIR/DONE" ]; do sleep 60; done
[ -s "$DATA" ] || { echo "FATAL: no $DATA"; exit 1; }
$PY - <<EOF
import h5py, numpy as np
d = h5py.File("$DATA")["data"]
lens = [d[k].attrs["num_samples"] for k in d]
print(f"demos: {len(d)}, steps mean {np.mean(lens):.0f} min {min(lens)} max {max(lens)}, "
      f"state dim {d['demo_0/obs/state'].shape[1]}, prop dim {d['demo_0/obs/prop'].shape[1]}")
assert len(d) >= 900 and max(lens) < 400
EOF
[ $? -eq 0 ] || { echo "FATAL: dataset check failed"; exit 1; }

# ---- 1. BC prior ------------------------------------------------------------
if ! grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null; then
    stamp "1. BC-RNN+GMM"
    $PY train_bc.py --config_path "$BC_CFG" --save_dir "$BC_DIR" --seed 1 --use_wb 0
    echo "train_bc rc=$? (1 is normal: train_bc.py ends in assert False)"
fi
grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null || { echo "FATAL: BC did not finish"; exit 1; }
grep -m1 "eval env uses the demos' sim2real cell" "$BC_DIR/train.log" || echo "WARNING: BC eval env without the cell?"
grep -E "best_ckpt_score" "$BC_DIR/train.log" | tail -1

# ---- 2. BC gate ---------------------------------------------------------------
stamp "2. BC gate: 300 episodes, strict, sigma 0.1, in the cell"
$PY evaluate/eval.py --weight "$BC_DIR/model0.pt" --mode bc --num_game 300 --mp 10 --strict_success 1 \
    --hand_init_perturb_joint_sigma 0.1 --sim2real_cell "$CELL" > "$BC_DIR/eval300.log" 2>&1
grep -E "EVAL ENV|score|success" "$BC_DIR/eval300.log" | tail -4
BC_SCORE=$($PY - <<EOF
import re
txt = open("$BC_DIR/eval300.log").read()
m = re.findall(r"^score: ([0-9.]+), time", txt, re.M)     # eval.py's summary line
print(m[-1] if m else "nan")
EOF
)
stamp "BC independent score: $BC_SCORE (gate $BC_GATE)"
$PY -c "import sys; s=float('$BC_SCORE'); sys.exit(0 if s == s and s >= $BC_GATE else 1)" || {
    echo "STOP: BC below the gate -- find out why before IBRL (feedback-gate-rl-on-bc)"; exit 2; }

# ---- 3. IBRL ------------------------------------------------------------------
stamp "3. IBRL state, 1M steps"
$PY train_rl.py --config_path "$RL_CFG" --save_dir "$RL_DIR" --bc_policy "$BC_DIR/model0.pt" --seed 1 \
    --snapshot_period 100000 --use_wb 0
echo "train_rl rc=$?"
grep -m2 "prior\|SGFT" "$RL_DIR/train.log" || true
grep -E "^ *[0-9]+: score/score " "$RL_DIR/train.log" 2>/dev/null | tail -8

# ---- 4. independent eval ------------------------------------------------------
stamp "4. IBRL independent eval: 300 episodes"
$PY evaluate/eval.py --weight "$RL_DIR/model0.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
    --hand_init_perturb_joint_sigma 0.1 --sim2real_cell "$CELL" > "$RL_DIR/eval300.log" 2>&1
grep -E "EVAL ENV|score|success" "$RL_DIR/eval300.log" | tail -4
stamp "done"
