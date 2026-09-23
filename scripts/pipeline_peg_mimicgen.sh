#!/bin/bash
# PegInsertion_D0 source expert from MimicGen demos (local, one GPU):
#   1. MimicGen: 1000 successful demos (board jitter +-10 mm / +-5 deg)
#   2. convert  : tools/dataset_states_to_obs.py -> state dataset
#   3. BC       : single-step MLP prior      (peg_insertion_state_source_mlp_bc_mg.yaml)
#   4. IBRL     : state-based, snapshot every 25k (peg_insertion_state_ibrl_source_mg.yaml)
#
# Launch detached (Bash-tool timeouts kill the process group, CLAUDE.md #11):
#   setsid bash scripts/pipeline_peg_mimicgen.sh > logs/pipeline_peg_mg.log 2>&1 < /dev/null & disown
# Every step checks the previous step's ARTIFACT, not its exit code
# (train_bc.py exits 1 on success by design).
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python

# TAG="" reproduces the first (soft-wall, pre-fix) run; TAG=_hard is the fixed task
# (env/peg_insertion.py 2026-09-23 defaults) with MimicGen insert noise 0.
TAG="${TAG:-_hard}"
INSERT_NOISE="${INSERT_NOISE:-0.0}"
NAME=peg_mg_n1000_board$TAG
GEN=release/data/mimicgen/generated/peg_insertion/$NAME/demo.hdf5
DATA_DIR=release/data/peg_insertion/demos_mg_n1000_board$TAG
DATA=$DATA_DIR/processed_data96.hdf5
BC_CFG=release/cfgs/mimicgen/peg_insertion_state_source_mlp_bc_mg$TAG.yaml
BC_DIR=exps/mimicgen/peg_source_mlp_bc_mg_n1000_board${TAG}_seed1
RL_CFG=release/cfgs/mimicgen/peg_insertion_state_ibrl_source_mg$TAG.yaml
RL_DIR=exps/mimicgen/peg_ibrl_source_mg_board${TAG}_seed1

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }

# ---- 1. generate (skip if already done) ---------------------------------
if [ ! -s "$GEN" ]; then
    stamp "1. MimicGen generate 1000 (guarantee)"
    $PY scripts/gen_peg_mimicgen.py generate --num 1000 --guarantee 1 --seed 10 \
        --name $NAME --board_pos_noise 0.01 --board_yaw_noise 5 --insert_noise "$INSERT_NOISE"
fi
[ -s "$GEN" ] || { echo "FATAL: no $GEN"; exit 1; }
N_GEN=$($PY -c "import h5py;print(len(h5py.File('$GEN')['data']))")
stamp "generated demos: $N_GEN"
[ "$N_GEN" -ge 1000 ] || { echo "FATAL: only $N_GEN demos"; exit 1; }

# ---- 2. convert ----------------------------------------------------------
if [ ! -s "$DATA" ]; then
    stamp "2. states -> obs"
    mkdir -p "$DATA_DIR"
    $PY tools/dataset_states_to_obs.py --dataset "$GEN" --output_folder "$DATA_DIR" \
        --done_mode 0 > "$DATA_DIR/convert.log" 2>&1
fi
[ -s "$DATA" ] || { echo "FATAL: no $DATA"; exit 1; }
$PY - <<EOF
import h5py, numpy as np
f = h5py.File("$DATA"); d = f["data"]
lens = [d[k].attrs["num_samples"] for k in d]
nosucc = sum(1 for k in d if not (d[k]["rewards"][()] == 1).any())
print(f"converted: {len(d)} demos, steps mean {np.mean(lens):.1f} min {min(lens)} max {max(lens)}, no-success {nosucc}")
assert max(lens) < 350, "eval_episode_len 350 must exceed the longest demo"
EOF
[ $? -eq 0 ] || { echo "FATAL: dataset check failed"; exit 1; }

# ---- 3. BC prior ---------------------------------------------------------
if [ ! -s "$BC_DIR/model0.pt" ] || ! grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null; then
    stamp "3. BC (MLP)"
    $PY train_bc.py --config_path "$BC_CFG" --save_dir "$BC_DIR" --seed 1 --use_wb 0
    echo "train_bc rc=$? (1 is normal: train_bc.py ends in assert False)"
fi
grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null || { echo "FATAL: BC did not finish"; exit 1; }
grep -E "^ *[0-9]+: score " "$BC_DIR/train.log" | tail -5
grep -E "best_ckpt_score|final eval" "$BC_DIR/train.log" | tail -2

# ---- 4. IBRL -------------------------------------------------------------
stamp "4. IBRL (state, snapshot every 25k)"
$PY train_rl.py --config_path "$RL_CFG" --save_dir "$RL_DIR" \
    --bc_policy "$BC_DIR/model0.pt" --seed 1 --snapshot_period 25000 --use_wb 0
echo "train_rl rc=$?"
echo "snapshots: $(ls "$RL_DIR"/snapshot_*.pt 2>/dev/null | wc -l)"
grep -E "^ *[0-9]+: score/score " "$RL_DIR/train.log" 2>/dev/null | tail -8

# ---- 5. physical sanity of the final policy (penetration / tilt / depth) --
stamp "5. verify model0 (n=100)"
$PY scripts/verify_peg_ibrl.py --weight "$RL_DIR/model0.pt" --n 100 \
    --out "$RL_DIR/verify_model0_n100.npz" 2>&1 | grep -E "^\[env\]|^success|^penetr|^max insertion|^tilt|^tip|^actor"
stamp "done"
