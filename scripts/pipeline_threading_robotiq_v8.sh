#!/bin/bash
# v8 (10-30): the v7 chain on the 10-30 cell (z_min_base 0.048 = the robot's --z_min 0.960, z_min_above_table 0.0195).
# The whole Robotiq chain on the 10-29 cell (10-28 + wider in-hand pose DR, prob 0.7; owner 10-28: "we do not know how the real robot will grasp"):
#   A. 3000 scripted-expert demos (24 CPU shards x 125)        -> $DEMOS
#   B. obs/exec_ref for the first 1000 (bit-exact replay)        -> ${DEMOS}_execobs
#   C. BC-RNN prior (the v5 recipe: h1000, EMA, 6000 epochs); gate: best independent 300-episode eval >= BC_GATE
#   D. IBRL exec_obs, 2 seeds x 2M steps (the v5 fr_chunk1_execobs recipe), GPUs $GPUS
#   E. scripts/pipeline_threading_robotiq_stage1.sh: gate 0.9 -> freeze -> pixel data -> SimDist stage 1 -> evals
# Every step skips itself when its product exists, so a re-launch resumes.
#   setsid bash scripts/pipeline_threading_robotiq_v8.sh > logs/pipeline_robotiq_v8.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh >/dev/null 2>&1
export MUJOCO_GL=egl WANDB_MODE=offline
PY="${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python}"   # another machine: export PY=<env>/bin/python
export CELL=real2sim/zed_capture/calib_20261015/threading_real_cell_20261030.yaml
DEMOS="${DEMOS:-/user_data/Chenyang/simdist_data/threading_robotiq_expert_n3000_v8}"
BC_DIR=exps/sim2real/threading_robotiq_rnn_bc_n3000v8_h1000_ema_ep6000_seed1
BC_GATE="${BC_GATE:-0.3}"
GPUS=(${GPUS:-2 3})
IBRL=(exps/sim2real/ibrl_robotiq_v8_fr_chunk1_execobs_seed1 exps/sim2real/ibrl_robotiq_v8_fr_chunk1_execobs_seed2)
stamp() { echo "[v8 $(date -u '+%F %T')] $*"; }
score_of() { grep -E '^score: ' "$1" 2>/dev/null | tail -1 | sed -E 's/^score: ([0-9.]+).*/\1/'; }
stamp "cell $CELL sha256 $(sha256sum "$CELL" | cut -c1-16), git $(git rev-parse --short HEAD)"

# ---- A. demos ---------------------------------------------------------------------------------------------------
if [ ! -e "$DEMOS/DONE" ]; then
  mkdir -p "$DEMOS"; git rev-parse HEAD > "$DEMOS/git_head.txt"; sha256sum "$CELL" > "$DEMOS/cell_sha256.txt"
  stamp "A. demos -> $DEMOS"
  for i in $(seq 208 231); do        # seeds disjoint from v1-v7 (100-207)
    [ -s "$DEMOS/shard_$i/dataset.hdf5" ] && continue
    CUDA_VISIBLE_DEVICES="" nice -n 5 $PY scripts/collect_threading_robotiq_demos.py --cell "$CELL" --num_demos 125 \
        --shard $i --out_dir "$DEMOS/shard_$i" > "$DEMOS/shard_$i.log" 2>&1 &
  done
  wait
  $PY scripts/collect_threading_robotiq_demos.py --merge "$DEMOS"/shard_*/ --out_dir "$DEMOS" > "$DEMOS/merge.log" 2>&1
  n=$($PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1])['data']))" "$DEMOS/dataset.hdf5" 2>/dev/null)
  [ "${n:-0}" -ge 2900 ] || { stamp "FATAL: only ${n:-0} demos"; exit 1; }
  grep -h "demos in" "$DEMOS"/shard_*.log | awk '{s+=$3; a+=$6} END{print "[v8] scripted success", s/a, "(" s "/" a ")"}'
  echo done > "$DEMOS/DONE"
fi
stamp "A. $(tail -1 "$DEMOS/merge.log")"

# ---- B. exec_obs ------------------------------------------------------------------------------------------------
if [ ! -s "${DEMOS}_execobs/dataset.hdf5" ]; then
  stamp "B. exec_obs -> ${DEMOS}_execobs"
  CUDA_VISIBLE_DEVICES="" $PY scripts/add_exec_obs_to_demos.py --src "$DEMOS/dataset.hdf5" --out_dir "${DEMOS}_execobs" \
      --num 1000 --workers 24 > "$DEMOS/execobs.log" 2>&1 || { stamp "FATAL: exec_obs replay failed"; tail -5 "$DEMOS/execobs.log"; exit 1; }
fi

# ---- C. BC prior + gate -----------------------------------------------------------------------------------------
if [ -z "$(score_of "$BC_DIR/eval300_model0.log")" ]; then
  stamp "C. BC -> $BC_DIR"
  GPU=${GPUS[0]} DATA="$DEMOS/dataset.hdf5" DIR="$BC_DIR" \
    EXTRA="--eval_period 500 --rnn_policy.rnn_hidden_dim 1000 --rnn_policy.use_ema 1 --num_epoch 6000" \
    bash scripts/run_threading_robotiq_bc_v3.sh
fi
s0=$(score_of "$BC_DIR/eval300_model0.log"); s1=$(score_of "$BC_DIR/eval300_model1.log")
BC="$BC_DIR/model0.pt"; bs=${s0:-0}
if [ -n "$s1" ] && awk -v a="$s1" -v b="$bs" 'BEGIN{exit !(a>b)}'; then BC="$BC_DIR/model1.pt"; bs=$s1; fi
stamp "C. BC model0 ${s0:-?} / model1 ${s1:-?} -> $BC | $(grep -m1 'EVAL ENV' "$BC_DIR/eval300_model0.log")"
awk -v a="$bs" -v g="$BC_GATE" 'BEGIN{exit !(a>=g)}' || { stamp "BC GATE FAILED ($bs < $BC_GATE): stopping before RL"; exit 1; }

# ---- D. IBRL ----------------------------------------------------------------------------------------------------
for k in 0 1; do
  d=${IBRL[$k]}
  if [ -e "$d/train.log" ]; then stamp "D. $d exists (not relaunched)"; continue; fi
  stamp "D. IBRL -> $d (GPU ${GPUS[$k]})"
  ARM=fr_chunk1_execobs TAG=v8 SEED=$((k + 1)) GPU=${GPUS[$k]} DATA="${DEMOS}_execobs/dataset.hdf5" BC="$BC" \
    EXTRA="--fail_replay_prob 0.5 --num_train_step 2000000 --exec_obs 1" \
    setsid bash scripts/run_ibrl_robotiq_arm.sh > logs/ibrl_v8_fr_chunk1_execobs_s$((k + 1)).log 2>&1 < /dev/null &
  disown
  sleep 120                           # the two runs' startup (replay preload) one after the other
done

# ---- E. gate -> freeze -> pixel data -> stage 1 (waits for the snapshots) ----------------------------------------
stamp "E. stage-1 pipeline"
RUNS="${IBRL[*]}" FROZEN=exps/sim2real/threading_robotiq_v8_expert \
  DEST=/user_data/Chenyang/simdist_data/threading_robotiq_pixel_v8 S1=exps/sim2real/threading_robotiq_v8_pixel_stage1_seed1 \
  GPU=${GPUS[1]} bash scripts/pipeline_threading_robotiq_stage1.sh
stamp "done"
