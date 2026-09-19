#!/bin/bash
# SimDist stage-1 source data on the REAL-CELL Threading geometry, in batches.
# Generalised from collect_tpa_batches.sh (same guard / salvage / merge logic).
#
# Env comes from the expert's cfg (sim2real_cell -> release/cfgs/sim2real/
# threading_real_cell.yaml), i.e. measured arena + Franka ready reset + the cell's
# DR (base height/xy/yaw/tilt; visual DR is inert for state data). Collection-time
# noise on top: action-noise segments sigma U[0.05,0.2], joint init sigma 0.1.
#
# Run one chain per GPU, disjoint batch ids (seeds = SEED_BASE + b):
#   GPU=2 BATCH_IDS="1 3 5 7" setsid bash scripts/collect_threading_realcell_batches.sh \
#       > logs/realcell_collect_gpu2.log 2>&1 < /dev/null & disown
#   GPU=3 BATCH_IDS="2 4 6 8" setsid bash ... > logs/realcell_collect_gpu3.log ...
set -uo pipefail
REPO=/home/Chenyang/ibrl
cd "$REPO"

GPU="${GPU:-2}"
NUM_PROC="${NUM_PROC:-20}"
BATCH_TRAJ="${BATCH_TRAJ:-25000}"
BATCH_IDS="${BATCH_IDS:-1 2 3 4 5 6 7 8}"
SEED_BASE="${SEED_BASE:-100}"
RUN="${RUN:-exps/mimicgen/threading_realcell_ibrl_seed1}"
EXPERT="${EXPERT:-$RUN/model0.pt}"
CELL="${CELL:-release/cfgs/sim2real/threading_real_cell.yaml}"
DEST="${DEST:-/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_source_diverse_v1}"
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
PER_WORKER_MIB="${PER_WORKER_MIB:-2706}"
GUARD_FRAC="${GUARD_FRAC:-0.75}"
HEADROOM_MIB="${HEADROOM_MIB:-12000}"

mkdir -p "$DEST" logs
# Pin the cell the data is collected on (the yaml is read at worker start; it
# must not change mid-collection). First chain to start writes the snapshot.
if [ ! -f "$DEST/cell_snapshot.yaml" ]; then
  cp "$CELL" "$DEST/cell_snapshot.yaml"
  sha256sum "$EXPERT" > "$DEST/expert.sha256"
  git rev-parse HEAD > "$DEST/git_head.txt"; git diff > "$DEST/git_diff.patch"
fi
cmp -s "$CELL" "$DEST/cell_snapshot.yaml" \
  || { echo "[batches] ERROR: $CELL differs from $DEST/cell_snapshot.yaml"; exit 1; }

need=$(( NUM_PROC * PER_WORKER_MIB + HEADROOM_MIB ))
total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU")
guard_mib=$(awk -v t="$total_mib" -v f="$GUARD_FRAC" 'BEGIN{printf "%d", t*f}')
echo "[batches] GPU=$GPU num_proc=$NUM_PROC per-batch=$BATCH_TRAJ ids=[$BATCH_IDS] dest=$DEST"
echo "[batches] need ${need} MiB free; guard line ${guard_mib}/${total_mib}"

for b in $BATCH_IDS; do
  seed=$(( SEED_BASE + b ))
  out="$DEST/_b${b}"
  if [ -f "$DEST/part${b}.hdf5" ]; then echo "[batches] part${b} exists, skipping"; continue; fi
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 999999)
    if [ "$(( guard_mib - used ))" -ge "$need" ]; then break; fi
    echo "[batches] $(date +%H:%M:%S) waiting: used=${used} MiB -- sleeping 300s"
    sleep 300
  done
  echo "[batches] === batch $b seed=$seed out=$out (used=${used} MiB) $(date) ==="
  rm -rf "$out"; mkdir -p "$out"
  env GPU="$GPU" NUM_PROC="$NUM_PROC" NUM_TRAJ="$BATCH_TRAJ" MAX_LEN=400 \
      HAND_INIT_SIGMA=0.1 HYBRID=1 SNAP_HYBRID=1 GUARD_THRESH="$GUARD_FRAC" SEED="$seed" \
      RUN="$RUN" EXPERT="$EXPERT" OUT="$out" \
      bash scripts/generate_simdist_data_threading.sh --flush_every 250 \
      > "logs/realcell_gen_b${b}.log" 2>&1 < /dev/null
  rc=$?
  [ $rc -eq 0 ] || echo "[batches] batch $b exited rc=$rc (likely a guard kill) -- salvaging"

  if [ -f "$out/dataset.hdf5" ]; then
    mv "$out/dataset.hdf5" "$DEST/part${b}.hdf5"
    cp "$out/env_cfg.json" "$DEST/env_cfg_part${b}.json" 2>/dev/null || true
    echo "[batches] batch $b complete -> part${b}.hdf5"
  elif ls "$out"/_shard_*.hdf5 >/dev/null 2>&1; then
    "$PY" - "$DEST/part${b}.hdf5" "$out"/_shard_*.hdf5 <<'PYEOF'
import sys
sys.path.insert(0, "/home/Chenyang/ibrl")
from scripts.generate_simdist_data import merge_shards
tot, ns = merge_shards(sys.argv[2:], sys.argv[1])
print(f"[batches] salvaged {tot} traj ({ns} success) -> {sys.argv[1]}")
PYEOF
  else
    echo "[batches] batch $b produced nothing -- see logs/realcell_gen_b${b}.log"
  fi
  cmp -s "$CELL" "$DEST/cell_snapshot.yaml" \
    || { echo "[batches] ERROR: $CELL changed during batch $b -- stopping"; exit 1; }
  "$PY" -c "
import glob, h5py, os
tot = 0
for p in sorted(glob.glob('$DEST/part*.hdf5')):
    with h5py.File(p, 'r') as f: n = len(f['data'].keys()); tot += n
    print(f'  {os.path.basename(p)}: {n}')
print(f'[batches] cumulative: {tot} traj')
"
done
echo "[batches] chain on GPU $GPU done $(date)"
