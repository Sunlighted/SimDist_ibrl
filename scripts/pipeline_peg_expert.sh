#!/bin/bash
# PegInsertion_D0 source expert from CLOSED-LOOP scripted-expert demos (local, one GPU).
# Replaces the MimicGen-demo chain (scripts/pipeline_peg_mimicgen.sh): its open-loop
# insertion segments contain no corrective behaviour, and BC on them scored ~0.03
# under the fixed task (docs/real2sim.md §5a2).
#   1. collect : 1000 scripted-expert demos, board jitter +-10 mm / +-5 deg
#   2. BC      : single-step MLP prior      (peg_insertion_state_source_mlp_bc_expert_hard.yaml)
#   3. IBRL    : state-based, snapshot every 25k (peg_insertion_state_ibrl_source_expert_hard.yaml)
#   4. verify  : penetration / tilt / depth of the final policy
#
#   setsid bash scripts/pipeline_peg_expert.sh > logs/pipeline_peg_expert.log 2>&1 < /dev/null & disown
# Every step checks the previous step's ARTIFACT, not its exit code
# (train_bc.py exits 1 on success by design).
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python

DATA_DIR=release/data/peg_insertion/demos_expert_n1000_board_hard
DATA=$DATA_DIR/dataset.hdf5
BC_CFG=release/cfgs/mimicgen/peg_insertion_state_source_mlp_bc_expert_hard.yaml
BC_DIR=exps/mimicgen/peg_source_mlp_bc_expert_n1000_board_hard_seed1
RL_CFG=release/cfgs/mimicgen/peg_insertion_state_ibrl_source_expert_hard.yaml
RL_DIR=exps/mimicgen/peg_ibrl_source_expert_board_hard_seed1

BC_MIN="${BC_MIN:-0.20}"   # IBRL starts only if BC's independent n=100 success >= this

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }

# a collector left running by an earlier (stopped) launch of this script: wait for it
while ps -eo args | grep -q "^$PY scripts/[c]ollect_peg_demos.py"; do sleep 30; done

# ---- 1. collect ----------------------------------------------------------
if [ ! -s "$DATA" ]; then
    stamp "1. collect 1000 scripted-expert demos"
    $PY scripts/collect_peg_demos.py --num_demos 1000 --out_dir "$DATA_DIR" --seed 0 \
        --sigma 0.1 --episode_length 300 --state_only 1 \
        --board_pos_noise 0.01 --board_yaw_noise 5
fi
[ -s "$DATA" ] || { echo "FATAL: no $DATA"; exit 1; }
$PY - <<EOF
import h5py, numpy as np
f = h5py.File("$DATA"); d = f["data"]
lens = [d[k].attrs["num_samples"] for k in d]
print(f"demos: {len(d)}, steps mean {np.mean(lens):.1f} min {min(lens)} max {max(lens)}")
assert len(d) >= 1000, "fewer than 1000 demos"
assert max(lens) < 350, "eval_episode_len 350 must exceed the longest demo"
EOF
[ $? -eq 0 ] || { echo "FATAL: dataset check failed"; exit 1; }

# ---- 2. BC prior ---------------------------------------------------------
if [ ! -s "$BC_DIR/model0.pt" ] || ! grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null; then
    stamp "2. BC (MLP)"
    $PY train_bc.py --config_path "$BC_CFG" --save_dir "$BC_DIR" --seed 1 --use_wb 0
    echo "train_bc rc=$? (1 is normal: train_bc.py ends in assert False)"
fi
grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null || { echo "FATAL: BC did not finish"; exit 1; }
echo "BC scores (epoch: score):"
grep -E "^ *[0-9]+: score " "$BC_DIR/train.log" | awk '{printf "%s %s  ", $1, $4} END {print ""}'
grep -E "best_ckpt_score" "$BC_DIR/train.log" | tail -1

# ---- 2b. GATE: independent BC eval; stop here if it is too low ------------
# (in-training best is a max over 40 noisy n=50 evals -> optimistic, CLAUDE.md #14)
stamp "2b. BC gate: independent n=100 eval (need >= $BC_MIN)"
$PY scripts/diag_peg_phases.py --kind bc --weight "$BC_DIR/model0.pt" --n 100 \
    > "$BC_DIR/diag_bc_n100.log" 2>&1
grep -A20 "^\[bc\]" "$BC_DIR/diag_bc_n100.log" | grep -v EGL
BC_SR=$(grep -oP "^\[bc\] n=\d+  success \K[0-9.]+" "$BC_DIR/diag_bc_n100.log")
[ -n "$BC_SR" ] || { echo "FATAL: BC gate eval failed (see $BC_DIR/diag_bc_n100.log)"; exit 1; }
if ! $PY -c "import sys; sys.exit(0 if float('$BC_SR') >= float('$BC_MIN') else 1)"; then
    stamp "GATE FAILED: BC success $BC_SR < $BC_MIN -> NOT starting IBRL; expert baseline for comparison"
    $PY scripts/diag_peg_phases.py --kind expert --n 30 > "$BC_DIR/diag_expert_n30.log" 2>&1
    grep -A20 "^\[expert\]" "$BC_DIR/diag_expert_n30.log" | grep -v EGL
    stamp "stopped at BC gate"
    exit 2
fi
stamp "GATE PASSED: BC success $BC_SR >= $BC_MIN"

# ---- 3. IBRL -------------------------------------------------------------
stamp "3. IBRL (state, snapshot every 25k)"
$PY train_rl.py --config_path "$RL_CFG" --save_dir "$RL_DIR" \
    --bc_policy "$BC_DIR/model0.pt" --seed 1 --snapshot_period 25000 --use_wb 0
echo "train_rl rc=$?"
echo "snapshots: $(ls "$RL_DIR"/snapshot_*.pt 2>/dev/null | wc -l)"
grep -E "^ *[0-9]+: score/score " "$RL_DIR/train.log" 2>/dev/null | tail -8

# ---- 4. physical sanity of the final policy -------------------------------
stamp "4. verify model0 (n=100)"
$PY scripts/verify_peg_ibrl.py --weight "$RL_DIR/model0.pt" --n 100 \
    --out "$RL_DIR/verify_model0_n100.npz" 2>&1 | grep -E "^\[env\]|^success|^penetr|^max insertion|^tilt|^tip|^actor"
stamp "done"
