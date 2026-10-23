#!/bin/bash
# One arm of the BC-prior sweep on the fixed Robotiq cell (2026-10-22): train (scripts/run_threading_robotiq_bc_v3.sh
# with EXTRA overrides) on the 10000 v3 demos -> independent 300-episode strict evals of model0 / model1 at the arm's
# own Ta -> the better one re-evaluated at Ta=1 (how IBRL queries the prior) and Ta=4, with failure buckets.
#   setsid env NAME=s0_ep6000 GPU=3 EXTRA="--num_epoch 6000" bash scripts/sweep_threading_robotiq_bc.sh \
#       > logs/bc_sweep_s0_ep6000.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh >/dev/null 2>&1
: "${NAME:?set NAME}"
DIR=exps/sim2real/threading_robotiq_rnn_bc_${TAG:-n3000v5}_${NAME}_seed${SEED:-1}
DATA="${DATA:-/user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v5/dataset.hdf5}"
if [ -e "$DIR/train.log" ]; then echo "FATAL: $DIR already has a run (refusing to overwrite it)"; exit 1; fi
env GPU="${GPU:-2}" SEED="${SEED:-1}" DATA=$DATA DIR=$DIR EXTRA="${EXTRA:-}" bash scripts/run_threading_robotiq_bc_v3.sh
s0=$(grep -E '^score: ' $DIR/eval300_model0.log | tail -1 | awk '{print $2}' | tr -d ,)
s1=$(grep -E '^score: ' $DIR/eval300_model1.log | tail -1 | awk '{print $2}' | tr -d ,)
best=model0; awk "BEGIN{exit !(${s1:-0} > ${s0:-0})}" && best=model1
echo "best $best (model0 $s0 / model1 $s1)"
for ta in 1 4; do
    BC_TA=$ta CUDA_VISIBLE_DEVICES="${GPU:-2}" /home/Chenyang/.conda/envs/ibrl/bin/python scripts/diagnose_threading_robotiq_failures.py \
        --weight $DIR/$best.pt --mode bc --num 300 --workers 10 --out $DIR/diag_ta$ta.npz 2>&1 | grep -A10 "^episodes" | sed "s/^/Ta=$ta /"
done
echo SWEEP_DONE
