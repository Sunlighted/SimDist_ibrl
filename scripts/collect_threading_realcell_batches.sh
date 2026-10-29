#!/bin/bash
# SimDist stage-1 source data on a REAL-CELL geometry, in batches (Threading by
# default; Square via RUN/CELL/DEST -- see docs/infra.md §15i).
# Generalised from collect_tpa_batches.sh (same guard / salvage / merge logic).
#
# Env comes from $CELL below, passed through to the generator as
# --sim2real_cell. It used to come from the expert's cfg instead, which is the
# agentview-only cell -- CELL was then only a provenance snapshot, so setting it
# to the 2-camera cell would have recorded a cell that was never rendered with. Collection-time
# noise on top: action-noise segments sigma U[0.05,0.2], joint init sigma 0.1.
#
# Run one chain per GPU, disjoint batch ids (seeds = SEED_BASE + b):
#   GPU=2 BATCH_IDS="1 3 5 7" setsid bash scripts/collect_threading_realcell_batches.sh \
#       > logs/realcell_collect_gpu2.log 2>&1 < /dev/null & disown
#   GPU=3 BATCH_IDS="2 4 6 8" setsid bash ... > logs/realcell_collect_gpu3.log ...
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

GPU="${GPU:-2}"
NUM_PROC="${NUM_PROC:-20}"
BATCH_TRAJ="${BATCH_TRAJ:-25000}"
BATCH_IDS="${BATCH_IDS:-1 2 3 4 5 6 7 8}"
SEED_BASE="${SEED_BASE:-100}"
RUN="${RUN:-exps/mimicgen/threading_realcell_ibrl_seed1}"
EXPERT="${EXPERT:-$RUN/model0.pt}"
# ★ the CALIBRATED-WRIST cell: same file as the collection cell in every field
# except `cameras` (gated in scripts/test_sim2real.py), and it is the only place
# the wrist pose + fovy live. Using the plain cell here would render a wrist at
# robosuite's DEFAULT pose (0.05, 0, 0) and fovy 75 -- silently, since a camera
# always renders something.
CELL="${CELL:-release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml}"
DEST="${DEST:-/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_source_diverse_v1}"
# Cameras to RENDER now. ★ 2026-09-23: BOTH, and the cell below is the
# *_calibrated_wrist one, so a collection no longer needs a follow-up re-render.
# History (kept so nobody re-introduces the split): the wrist extrinsic used to
# be unresolved, so collection was agentview-only and the wrist was added later
# by scripts/rerender_simdist_data.py. That indirection cost a full extra pass
# and, worse, the v2 collection was pointed straight at a 2-camera training
# config and silently loaded ZERO episodes (every one failed the "has all the
# image keys" probe). The re-render still exists and is still the right tool for
# CHANGING a camera after the fact -- it just is not part of the happy path.
CAMS="${CAMS:-agentview+robot0_eye_in_hand}"
RL_IMAGE_SIZE="${RL_IMAGE_SIZE:-224}"
JPEG_Q="${JPEG_Q:-90}"
STORE_STATE="${STORE_STATE:-1}"
PY="${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python}"   # another machine: export PY=<env>/bin/python
PER_WORKER_MIB="${PER_WORKER_MIB:-2706}"
GUARD_FRAC="${GUARD_FRAC:-0.75}"
HEADROOM_MIB="${HEADROOM_MIB:-12000}"
# Per-batch log prefix. The default keeps Threading's historical names; a second
# task MUST set its own or its batch logs overwrite Threading's (same b ids).
LOG_TAG="${LOG_TAG:-realcell}"

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
# A cell may `inherit` another (square_real_cell.yaml is a 2-key override of
# threading_real_cell.yaml), so the cmp above only covers the CHILD: a camera
# re-calibration in the parent would land mid-collection unrecorded. Hash the
# whole chain too. Written on first start, verified after every batch.
cell_chain() { "$PY" -c "
import os, sys, yaml
p = os.path.abspath(sys.argv[1])
while p:
    print(p)
    par = (yaml.safe_load(open(p)) or {}).get('inherit')
    p = os.path.join(os.path.dirname(p), par) if par else None
" "$1"; }
if [ ! -f "$DEST/cell_chain.sha256" ]; then
  cell_chain "$CELL" | xargs sha256sum > "$DEST/cell_chain.sha256"
  for f in $(cell_chain "$CELL"); do cp "$f" "$DEST/$(basename "$f").snapshot"; done
fi
sha256sum -c --status "$DEST/cell_chain.sha256" \
  || { echo "[batches] ERROR: the cell or a file it inherits differs from $DEST/cell_chain.sha256"; exit 1; }

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
      RUN="$RUN" EXPERT="$EXPERT" OUT="$out" CELL="$CELL" \
      bash scripts/generate_simdist_data_threading.sh --flush_every 250 \
      --rl_cameras "$CAMS" --rl_image_size "$RL_IMAGE_SIZE" \
      --jpeg_quality "$JPEG_Q" --store_sim_state "$STORE_STATE" --sim2real_cell "$CELL" \
      > "logs/${LOG_TAG}_gen_b${b}.log" 2>&1 < /dev/null
  rc=$?
  [ $rc -eq 0 ] || echo "[batches] batch $b exited rc=$rc (likely a guard kill) -- salvaging"

  if [ -f "$out/dataset.hdf5" ]; then
    mv "$out/dataset.hdf5" "$DEST/part${b}.hdf5"
    cp "$out/env_cfg.json" "$DEST/env_cfg_part${b}.json" 2>/dev/null || true
    echo "[batches] batch $b complete -> part${b}.hdf5"
  elif ls "$out"/_shard_*.hdf5 >/dev/null 2>&1; then
    REPO="$REPO" "$PY" - "$DEST/part${b}.hdf5" "$out"/_shard_*.hdf5 <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["REPO"])
from scripts.generate_simdist_data import merge_shards
tot, ns = merge_shards(sys.argv[2:], sys.argv[1])
print(f"[batches] salvaged {tot} traj ({ns} success) -> {sys.argv[1]}")
PYEOF
  else
    echo "[batches] batch $b produced nothing -- see logs/${LOG_TAG}_gen_b${b}.log"
  fi
  cmp -s "$CELL" "$DEST/cell_snapshot.yaml" \
    || { echo "[batches] ERROR: $CELL changed during batch $b -- stopping"; exit 1; }
  sha256sum -c --status "$DEST/cell_chain.sha256" \
    || { echo "[batches] ERROR: an inherited cell file changed during batch $b -- stopping"; exit 1; }
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
