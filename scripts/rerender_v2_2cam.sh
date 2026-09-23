#!/bin/bash
# Re-render the v2 shards into a 2-camera set with WORKING photo backgrounds.
#
# Why a re-render and not a re-collect: rollouts are unchanged, only pixels are.
# The shards carry sim_state + model_state, so every frame is reproducible and
# stays index-aligned with the transitions the rollouts produced.
#
# Two things this pass fixes that the collection got wrong:
#   1. only agentview was rendered, while the training config asks for two
#      cameras -- the loader then skipped 100% of the episodes, silently;
#   2. dr_background_gray stripped geom_matid to -1 and _bg_texids resolves the
#      photo target THROUGH the material, so every episode got flat gray. The
#      stored model_state carries that -1, so the repair has to happen here too
#      (apply_sim2real restores it from the pristine nominal).
# Collection itself is fixed going forward (2 cameras + the calibrated cell), so
# this script is for CHANGING an existing dataset, not part of the happy path.
#
#   setsid bash scripts/rerender_v2_2cam.sh > logs/rerender_v2.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
SRC=/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v2
DST=$SRC/2cam_bg_v1
CELL=release/cfgs/sim2real/threading_real_cell_calibrated_wrist.yaml
EXPERT=exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt
mkdir -p "$DST"
say() { echo "[rerender $(date '+%F %T')] $*"; }

cp "$CELL" "$DST/cell_snapshot.yaml"
sha256sum "$CELL" "$EXPERT" > "$DST/provenance.sha256"
say "src=$SRC  dst=$DST"; cat "$DST/provenance.sha256"

# Two GPUs, disjoint shards. Seeds are per-shard and per-episode inside the
# script, and the chosen photo index is written into each episode's attrs, so
# the whole set is reproducible without storing any pixels twice.
run_chain() {
  local gpu=$1; shift
  for b in "$@"; do
    out="$DST/part${b}_2cam.hdf5"
    [ -f "$out" ] && { say "part$b exists, skipping"; continue; }
    say "GPU$gpu part$b -> $out"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/rerender_simdist_data.py \
      --dataset "$SRC/part${b}.hdf5" --out "$out" \
      --expert "$EXPERT" --cell "$CELL" \
      --cameras agentview+robot0_eye_in_hand \
      --rl_image_size 224 --jpeg_quality 90 \
      --visual_dr 1 --visual_dr_seed $((900000 + b * 20000)) \
      >> "logs/rerender_v2_part${b}.log" 2>&1 \
      || { say "ERROR on part$b (see logs/rerender_v2_part${b}.log)"; return 1; }
  done
}
run_chain 0 1 3 5 7 & A=$!
run_chain 1 2 4 6 8 & B=$!
wait $A; rcA=$?
wait $B; rcB=$?
n=$(ls "$DST"/part*_2cam.hdf5 2>/dev/null | wc -l)
say "chains rc=$rcA/$rcB; $n/8 shards in $DST"
[ "$n" -ge 8 ] || { say "ABORT: incomplete"; exit 1; }
du -sh "$DST"
say "done"
