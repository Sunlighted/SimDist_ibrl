#!/bin/bash
# One queue for the whole real-cell v2 rebuild, after the 2026-09-23 re-calibration.
#
# Why a rebuild: the object LAYOUT (tripod yaw -30 so the arm inserts away from
# the camera), the agentview FOV (50 deg, from the factory hd720 intrinsics -- the
# old 60.845 was HD2K's and simply wrong for the recorded stream), the board
# (0.753 m square, tape-measured) and the lighting (strip at z=3.0, 5 of 8 casting
# shadows, all fitted to the real board) ALL changed. Everything downstream of the
# cell is therefore void; v1 is archived, not deleted.
#
#   A  MimicGen demos -> convert -> strict audit -> MLP BC prior -> state IBRL
#   B  SimDist stage-1 pixel data (agentview, 224, JPEG q90, sim_state stored)
#   C  two stage-1 distillation arms (base vs full augmentation)
#
# Each stage refuses to start on a bad input rather than failing silently later.
# Launch detached (iron law #11):
#   setsid bash scripts/queue_realcell_v2.sh > logs/queue_realcell_v2.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline

DEST=/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v2
RL_DIR=exps/mimicgen/threading_realcell_ibrl_seed1
GEN=release/data/mimicgen/generated/threading_realcell
say() { echo "[queue $(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }

say "cell fingerprint:"; sha256sum release/cfgs/sim2real/threading_real_cell.yaml

# ---------------------------------------------------------------- A: state RL
if [ ! -f "$RL_DIR/model0.pt" ]; then
  say "A: demos -> BC prior -> IBRL   (scripts/pipeline_realcell.sh)"
  bash scripts/pipeline_realcell.sh || die "stage A failed, see the log above"
fi
[ -f "$RL_DIR/model0.pt" ] || die "no $RL_DIR/model0.pt after stage A"
say "A done. expert:"; sha256sum "$RL_DIR/model0.pt"
$PY - <<'PY' || true
import re
for line in open("exps/mimicgen/threading_realcell_ibrl_seed1/train.log", errors="ignore"):
    if "score" in line and "eval" in line.lower():
        last = line.rstrip()
print("[queue] last eval line:", last if 'last' in dir() else "n/a")
PY

# ------------------------------------------------------- B: stage-1 pixel data
# Two chains, disjoint batch ids, one per FREE gpu. 8 x 12500 = 100k episodes,
# matching v1's budget so the arms stay comparable to the published ladder.
if [ ! -f "$DEST/part8.hdf5" ]; then
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                      | awk -F, '$2 < 8000 {print $1}' | tr -d ' ')
  [ "${#FREE[@]}" -ge 2 ] || die "need 2 free GPUs for collection, found ${#FREE[@]}"
  say "B: collecting stage-1 pixel data on GPUs ${FREE[0]} and ${FREE[1]} -> $DEST"
  GPU=${FREE[0]} BATCH_IDS="1 3 5 7" BATCH_TRAJ=12500 NUM_PROC=10 PER_WORKER_MIB=3300 \
    DEST=$DEST LOG_TAG=realcell_v2 setsid bash scripts/collect_threading_realcell_batches.sh \
    > logs/realcell_v2_collect_a.log 2>&1 </dev/null &
  A=$!
  until [ -f "$DEST/git_diff.patch" ]; do sleep 5; done   # let chain A lay down the provenance first
  GPU=${FREE[1]} BATCH_IDS="2 4 6 8" BATCH_TRAJ=12500 NUM_PROC=10 PER_WORKER_MIB=3300 \
    DEST=$DEST LOG_TAG=realcell_v2 setsid bash scripts/collect_threading_realcell_batches.sh \
    > logs/realcell_v2_collect_b.log 2>&1 </dev/null &
  B=$!
  wait $A; wait $B
fi
n=$(ls "$DEST"/part*.hdf5 2>/dev/null | wc -l)
[ "$n" -ge 8 ] || die "collection produced only $n/8 shards (see logs/realcell_v2_collect_*.log)"
say "B done: $n shards in $DEST"

# ------------------------------------------------------ C: two distillation arms
# One GPU each, paired A/B: identical except the augmentation block.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F, '$2 < 8000 {print $1}' | tr -d ' ')
[ "${#FREE[@]}" -ge 2 ] || die "need 2 free GPUs for distillation, found ${#FREE[@]}"
say "C: stage-1 distillation, base on GPU ${FREE[0]}, aug on GPU ${FREE[1]}"
CUDA_VISIBLE_DEVICES=${FREE[0]} setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_v2.yaml \
  --source_agent "$RL_DIR/model0.pt" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_seed1 \
  > logs/realcell_v2_stage1_base.log 2>&1 </dev/null &
CUDA_VISIBLE_DEVICES=${FREE[1]} setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_aug_v2.yaml \
  --source_agent "$RL_DIR/model0.pt" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_aug_seed1 \
  > logs/realcell_v2_stage1_aug.log 2>&1 </dev/null &
wait
say "C done -- all stages finished"
