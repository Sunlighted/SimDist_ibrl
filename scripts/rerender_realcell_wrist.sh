#!/bin/bash
# Add the WRIST camera to the RealCell pixel shards, from stored sim_state.
#
# The collection pass rendered agentview only (the wrist extrinsic is still
# unresolved, docs/infra.md §14a R7). Because every episode carries its MuJoCo
# state and its DR draw, the wrist is a RENDER-ONLY pass -- no rollouts, and the
# frames come out index-aligned with the agentview frames of the SAME
# transitions (gate: scripts/test_rerender.py). Re-run it verbatim once the real
# extrinsic lands; only the cell yaml changes.
#
# Only the first LIMIT episodes per shard are re-rendered: stage 1 caps the
# buffer at seq_replay_max_transitions (6.5M) and loads source_replay_num_data
# per shard, so anything beyond that is never seen by training.
#
#   bash scripts/rerender_realcell_wrist.sh              # 8 shards, 4 per GPU
#   LIMIT=4500 GPUS="2 3" bash scripts/rerender_realcell_wrist.sh
set -uo pipefail
cd /home/Chenyang/ibrl
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
SRC="${SRC:-/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v1}"
DST="${DST:-$SRC/2cam}"
EXPERT="${EXPERT:-exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt}"
LIMIT="${LIMIT:-4500}"
GPUS="${GPUS:-2 3}"
CAM="${CAM:-robot0_eye_in_hand}"
mkdir -p "$DST" logs

i=0
pids=()
for f in "$SRC"/part*.hdf5; do
  b=$(basename "$f" .hdf5)
  out="$DST/${b}_2cam.hdf5"
  [ -f "$out" ] && { echo "[wrist] $out exists, skipping"; continue; }
  gpu=$(echo $GPUS | cut -d' ' -f$(( i % $(echo $GPUS | wc -w) + 1 )))
  echo "[wrist] $b -> $out on GPU $gpu (first $LIMIT episodes)"
  CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/rerender_simdist_data.py \
      --dataset "$f" --out "$out" --expert "$EXPERT" \
      --cameras "$CAM" --copy_cameras agentview \
      --rl_image_size 224 --jpeg_quality 90 --limit "$LIMIT" \
      > "logs/wrist_rerender_${b}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[wrist] done $(date); failures=$fail"
$PY - "$DST" <<'PYEOF'
import glob, h5py, os, sys
tot = 0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*_2cam.hdf5"))):
    with h5py.File(p, "r") as f:
        g = f["data"]; n = len(g)
        k = sorted(g.keys(), key=lambda x: int(x.split("_")[1]))[0]
        cams = [c[:-6] for c in g[k]["obs"].keys() if c.endswith("_image")]
    print(f"  {os.path.basename(p)}: {n} episodes, cameras={cams}")
    tot += n
print(f"[wrist] total {tot} episodes with both cameras")
PYEOF
