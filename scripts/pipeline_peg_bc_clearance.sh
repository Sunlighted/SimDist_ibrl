#!/bin/bash
# Does BC reach a useful success rate if the pocket is wider? (sim-only diagnostic)
# Same recipe as scripts/pipeline_peg_expert.sh (closed-loop expert demos, board
# jitter +-10 mm / +-5 deg, MLP BC), but with a pocket clearance CLEAR (m/side),
# and it STOPS after an independent BC eval -- no RL.
#
#   CLEAR=0.003 GPU=1 setsid bash scripts/pipeline_peg_bc_clearance.sh \
#       > logs/peg_bc_clear3mm.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
CLEAR="${CLEAR:?set CLEAR (m per side)}"
TAG=$($PY -c "print(f'{float($CLEAR)*1000:g}mm')")

DATA_DIR=release/data/peg_insertion/demos_expert_n1000_board_hard_clear$TAG
DATA=$DATA_DIR/dataset.hdf5
BC_CFG=release/cfgs/mimicgen/peg_insertion_state_source_mlp_bc_expert_hard.yaml
BC_DIR=exps/mimicgen/peg_source_mlp_bc_expert_n1000_board_hard_clear${TAG}_seed1

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }
stamp "clearance $TAG/side"

if [ ! -s "$DATA" ]; then
    stamp "1. collect 1000 scripted-expert demos"
    $PY scripts/collect_peg_demos.py --num_demos 1000 --out_dir "$DATA_DIR" --seed 0 \
        --sigma 0.1 --episode_length 300 --state_only 1 \
        --board_pos_noise 0.01 --board_yaw_noise 5 --hole_clearance "$CLEAR" \
        | grep -E "\[collect\] (done|wrote)"
fi
[ -s "$DATA" ] || { echo "FATAL: no $DATA"; exit 1; }

if ! grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null; then
    stamp "2. BC (MLP)"
    $PY train_bc.py --config_path "$BC_CFG" --save_dir "$BC_DIR" --seed 1 --use_wb 0 \
        --dataset.path "$DATA" --dataset.hole_clearance "$CLEAR" > "$BC_DIR.stdout" 2>&1
fi
grep -q "best_ckpt_score" "$BC_DIR/train.log" 2>/dev/null || { echo "FATAL: BC did not finish"; exit 1; }
grep -q "hole_clearance: $CLEAR" "$BC_DIR/cfg.yaml" || { echo "FATAL: BC cfg lacks hole_clearance $CLEAR"; exit 1; }
echo "BC in-training evals (epoch score):"
grep -E "^ *[0-9]+: score " "$BC_DIR/train.log" | awk '{printf "%s %s  ", $1, $4} END {print ""}'

stamp "3. independent BC eval n=100"
$PY scripts/diag_peg_phases.py --kind bc --weight "$BC_DIR/model0.pt" --n 100 \
    --hole_clearance "$CLEAR" > "$BC_DIR/diag_bc_n100.log" 2>&1
grep -E "^pocket|^\[bc\]|^  (closest|entered|final)" "$BC_DIR/diag_bc_n100.log"
stamp "done"
