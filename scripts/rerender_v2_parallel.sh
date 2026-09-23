#!/bin/bash
# Re-render the REMAINING v2 shards, several at a time.
#
# The first driver ran ONE process per GPU and the GPUs sat at 40-50% -- the
# bottleneck is the per-episode Python/JPEG loop, not the renderer, so the way
# to go faster is more processes, not a bigger GPU. Collection already does this
# (NUM_PROC=10 per GPU); the re-render never did.
#
# PER_GPU is deliberately modest. Two ways to kill this machine are on record
# (memory: gotcha-gpu-density-mppi-runs): ~50 GL contexts wedge the driver's
# rwsem, and VRAM OOM takes the whole box with it. 3 per GPU at ~2.5 GB each is
# ~8 GB of 98 GB and 6 contexts total.
#
#   setsid bash scripts/rerender_v2_parallel.sh > logs/rerender_v2_par.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
SRC=/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v2
DST=$SRC/2cam_bg_v1
CELL=release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml
EXPERT=exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt
SHARDS="${SHARDS:-3 4 5 6 7 8}"
PER_GPU="${PER_GPU:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-120}"
say() { echo "[par $(date '+%F %T')] $*"; }

# A shard already being written by the first driver must not be started again:
# two writers on one output is silent corruption.
inflight() { pgrep -af 'rerender_simdist_data' | grep -q "part${1}\.hdf5"; }

pids=(); i=0
for b in $SHARDS; do
  out="$DST/part${b}_2cam.hdf5"
  [ -f "$out" ] && { say "part$b done, skip"; continue; }
  inflight "$b" && { say "part$b already in flight, skip"; continue; }
  free_gb=$(df -BG --output=avail /user_data | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    say "ABORT: only ${free_gb}G free on /user_data (< ${MIN_FREE_GB}G). Each shard
      needs ~46G; the re-render has no disk guard, so it would fill the volume."
    exit 1
  fi
  gpu=$(( (i / PER_GPU) % 2 ))
  say "GPU$gpu part$b  (free ${free_gb}G)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/rerender_simdist_data.py \
    --dataset "$SRC/part${b}.hdf5" --out "$out" \
    --expert "$EXPERT" --cell "$CELL" \
    --cameras agentview+robot0_eye_in_hand \
    --rl_image_size 224 --jpeg_quality 90 \
    --visual_dr 1 --visual_dr_seed $((900000 + b * 20000)) \
    >> "logs/rerender_v2_part${b}.log" 2>&1 &
  pids+=($!); i=$((i+1))
  sleep 20      # stagger the GL context creation
done
say "launched ${#pids[@]} workers"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
n=$(ls "$DST"/part*_2cam.hdf5 2>/dev/null | wc -l)
say "workers rc=$rc; $n/8 shards present"
du -sh "$DST"; df -h /user_data | tail -1
