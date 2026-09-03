#!/usr/bin/env bash
# Batched, kill-tolerant continuation of the TPA source_diverse collection.
#
# Why batches: on this shared machine gpu_guard kills the collector when OTHER
# users' VRAM climbs past the threshold (it did at 26% of the first 100k run).
# One long run therefore loses its unflushed tail and needs a manual restart.
# Each batch here is its own process group (setsid), so a guard kill takes out
# only that batch; the wrapper survives, merges what the batch did write, waits
# for VRAM to come back, and starts the next one.
#
# Usage:
#   BATCHES=5 BATCH_TRAJ=15000 GPU=0 bash scripts/collect_tpa_batches.sh
set -uo pipefail
REPO=/home/Chenyang/ibrl
cd "$REPO"

GPU="${GPU:-0}"
NUM_PROC="${NUM_PROC:-8}"
BATCH_TRAJ="${BATCH_TRAJ:-15000}"
BATCHES="${BATCHES:-5}"
SEED_BASE="${SEED_BASE:-2}"          # part0 used seed 1; worker seeds are seed*100000+rank
DEST="${DEST:-/user_data/Chenyang/ibrl_offload/release/data/mimicgen/tpa_d0_source_diverse_v1}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
# VRAM needed per worker, measured: 2706 MiB. Require the batch to fit AND leave
# headroom below the guard line, or the batch dies within minutes of starting.
PER_WORKER_MIB="${PER_WORKER_MIB:-2706}"
GUARD_FRAC="${GUARD_FRAC:-0.75}"
HEADROOM_MIB="${HEADROOM_MIB:-12000}"

need=$(( NUM_PROC * PER_WORKER_MIB + HEADROOM_MIB ))
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU")
guard_mib=$(awk -v t="$total_mib" -v f="$GUARD_FRAC" 'BEGIN{printf "%d", t*f}')

echo "[batches] GPU=$GPU num_proc=$NUM_PROC per-batch=$BATCH_TRAJ x $BATCHES"
echo "[batches] need ${need} MiB free (incl ${HEADROOM_MIB} headroom); guard line ${guard_mib}/${total_mib}"

for b in $(seq 1 "$BATCHES"); do
  seed=$(( SEED_BASE + b - 1 ))
  out="$DEST/_b${b}"
  # ---- wait for room under the guard line ----
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 999999)
    if [ "$(( guard_mib - used ))" -ge "$need" ]; then break; fi
    echo "[batches] $(date +%H:%M:%S) waiting: used=${used}, need $(( need )) below ${guard_mib} -- sleeping 300s"
    sleep 300
  done
  echo "[batches] === batch $b/$BATCHES  seed=$seed  out=$out  (used=${used} MiB) ==="
  rm -rf "$out"; mkdir -p "$out"
  # setsid: gpu_guard kills ITS OWN process group, so isolating the batch keeps
  # this wrapper alive to merge + continue.
  setsid env GPU="$GPU" NUM_PROC="$NUM_PROC" NUM_TRAJ="$BATCH_TRAJ" MAX_LEN=1000 \
      HAND_INIT_SIGMA=0.1 HYBRID=1 SNAP_HYBRID=0 GUARD_THRESH="$GUARD_FRAC" SEED="$seed" \
      RUN=exps/mimicgen/tpa_ibrl_source_snapshots_seed1 \
      EXPERT=exps/mimicgen/tpa_source_expert_frozen/model0.pt \
      OUT="$out" \
      bash scripts/generate_simdist_data_threading.sh --flush_every 250 \
      > "logs/tpa_gen_b${b}.log" 2>&1 < /dev/null &
  wait $! || echo "[batches] batch $b exited nonzero (likely a guard kill) -- salvaging"

  # ---- salvage whatever landed, merged or not ----
  if [ -f "$out/dataset.hdf5" ]; then
    mv "$out/dataset.hdf5" "$DEST/part${b}.hdf5"
    echo "[batches] batch $b complete -> part${b}.hdf5"
  elif ls "$out"/_shard_*.hdf5 >/dev/null 2>&1; then
    "$PY" - "$DEST/part${b}.hdf5" "$out"/_shard_*.hdf5 <<'PYEOF'
import sys, os
sys.path.insert(0, "/home/Chenyang/ibrl")
from scripts.generate_simdist_data import merge_shards
out, paths = sys.argv[1], sys.argv[2:]
n, ns = merge_shards(paths, out)
print(f"[salvage] {n} traj, {ns} success -> {out} ({os.path.getsize(out)/1e9:.2f} GB)")
PYEOF
    echo "[batches] batch $b PARTIAL -> part${b}.hdf5"
  else
    echo "[batches] batch $b produced nothing"
  fi
  rm -rf "$out"
  "$PY" -c "
import h5py, glob, os
tot=0
for p in sorted(glob.glob('$DEST/part*.hdf5')):
    with h5py.File(p,'r') as f: n=len(f['data'].keys()); tot+=n
    print(f'  {os.path.basename(p)}: {n}')
print(f'[batches] cumulative: {tot} traj')
"
done
echo "[batches] all batches done $(date)"
