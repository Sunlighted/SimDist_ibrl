#!/bin/bash
# Start the whole Dyna trust-region sweep: 11 cells x 3 seeds = 33 runs.
# eval workers run on CPU (evaluate/multi_process_eval.py forces device=cpu), so
# each run holds exactly ONE cuda context -- the 51-context driver wedge from
# CLAUDE.md's gpu-density gotcha does not apply here; the binding constraint is
# CPU (num_eval_proc=10 processes per run during eval) and this box has 128 cores.
set -u
NWORKER=${NWORKER:-6}
cd /home/Chenyang/ibrl
mkdir -p logs
for w in $(seq 0 $((NWORKER-1))); do
  GPU=$((w % 4))
  setsid bash scripts/sweep_dyna_bccoef.sh $w $GPU $NWORKER \
      > logs/sweep_bc_w${w}.log 2>&1 < /dev/null &
  disown
  echo "worker $w -> gpu $GPU  (logs/sweep_bc_w${w}.log)"
done
