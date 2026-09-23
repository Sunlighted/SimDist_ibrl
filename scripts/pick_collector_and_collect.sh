#!/bin/bash
# Wait for both IBRL arms, pick the collector by a PRE-REGISTERED rule, then run
# stage B (pixel data) and stage C (two distillation arms).
#
# The rule is fixed here, BEFORE the numbers exist, because "pick whichever looks
# better afterwards" is the winner's curse this project has already paid for
# (iron law #3: a post-hoc pick was inflated 2.4x and one "+0.018 peak" came back
# as -0.011). Both arms are scored on the SHIPPED cell (+-5 deg) -- that is the
# distribution the collector will actually roll out on, and the fixed-yaw arm
# being tested off its training distribution is the honest question, not a bug.
#
#   DEFAULT = the +-5 arm (owner's preference; it matches the collection cell).
#   Switch only if fixed-yaw beats it by MORE THAN 0.05 at n=500.
#   0.05 is this project's stated unreadable-below floor; at n=500 and p~0.9 the
#   SE of a difference is ~0.019, so 0.05 is ~2.6 SE -- a real gap, not noise.
#
#   setsid bash scripts/pick_collector_and_collect.sh > logs/pick_collector.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline

JIT=exps/mimicgen/threading_realcell_ibrl_seed1
FIX=exps/mimicgen/threading_realcell_fixedyaw_ibrl_seed1
CELL=release/cfgs/sim2real/threading_real_cell.yaml          # the SHIPPED (+-5) cell
DEST=/user_data/Chenyang/ibrl_offload/release/data/mimicgen/threading_realcell_pixel_v2
N=500
say() { echo "[pick $(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }

say "waiting for both IBRL arms to finish"
while pgrep -f 'train_rl.py --config_path release/cfgs/sim2real/threading_realcell' >/dev/null; do sleep 120; done
for d in "$JIT" "$FIX"; do [ -f "$d/model0.pt" ] || die "no $d/model0.pt"; done
say "both arms done"; sha256sum "$JIT/model0.pt" "$FIX/model0.pt"

score() {   # $1 = run dir, $2 = tag
  local out=logs/pick_eval_$2.log
  $PY -u evaluate/eval.py --weight "$1/model0.pt" --mode rl --num_game $N --mp 10 \
      --sim2real_cell "$CELL" --strict_success 1 --hand_init_perturb_joint_sigma 0.1 \
      --no_render 1 > "$out" 2>&1
  # Parse the SUCCESS RATE, not "score:". eval.py's LAST score-ish line is
  # `average score: 88.40` -- a PERCENTAGE -- so `tail -1` on "score:" returns
  # 88.40 and the 0.05 decision threshold silently becomes 0.05 percentage
  # points, i.e. the pre-registered rule would flip on pure noise.
  $PY -c "
import re, sys
m = re.findall(r'success rate:\s*([0-9.]+)%', open('$out', errors='ignore').read())
assert m, 'no success-rate line in $out'
v = float(m[-1]) / 100.0
assert 0.0 <= v <= 1.0, v
print(f'{v:.4f}')"
}
say "scoring both arms on the SHIPPED cell, n=$N, strict, sigma=0.1"
S_JIT=$(score "$JIT" jitter); S_FIX=$(score "$FIX" fixed)
[ -n "${S_JIT:-}" ] && [ -n "${S_FIX:-}" ] || die "could not parse a score (see logs/pick_eval_*.log)"
say "jitter(+-5) = $S_JIT   fixed(-30) = $S_FIX   (both on the +-5 cell)"

PICK=$JIT; WHY="default: the +-5 arm, and it matches the collection cell"
if $PY -c "import sys; sys.exit(0 if $S_FIX - $S_JIT > 0.05 else 1)"; then
  PICK=$FIX; WHY="fixed-yaw beat it by >0.05 ($S_FIX vs $S_JIT), the pre-registered switch"
fi
say "COLLECTOR = $PICK   ($WHY)"
printf '%s\n' "jitter=$S_JIT" "fixed=$S_FIX" "pick=$PICK" "why=$WHY" > logs/collector_choice.txt

# stratified: where does the cost land, if anywhere
for y in -35 -30 -25; do
  c=/tmp/cell_yaw$y.yaml
  sed "s/^layout_tripod_z_rot_deg:.*/layout_tripod_z_rot_deg: [$y, $y]/" "$CELL" > "$c"
  $PY -u evaluate/eval.py --weight "$PICK/model0.pt" --mode rl --num_game 300 --mp 10 \
        --sim2real_cell "$c" --strict_success 1 --hand_init_perturb_joint_sigma 0.1 --no_render 1 \
        > logs/pick_eval_yaw$y.log 2>&1
  s=$($PY -c "
import re
m = re.findall(r'success rate:\s*([0-9.]+)%', open('logs/pick_eval_yaw$y.log', errors='ignore').read())
print(f'{float(m[-1])/100:.4f}' if m else 'NA')")
  say "  stratified yaw $y deg: $s"
done

# ------------------------------------------------------------ B: pixel data
if [ ! -f "$DEST/part8.hdf5" ]; then
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1<8000' | wc -l)" -ge 2 ]; do
    say "waiting for 2 free GPUs"; sleep 300
  done
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                      | awk -F, '$2 < 8000 {print $1}' | tr -d ' ')
  say "B: collecting 100k pixel episodes on GPUs ${FREE[0]},${FREE[1]} with $PICK -> $DEST"
  GPU=${FREE[0]} BATCH_IDS="1 3 5 7" BATCH_TRAJ=12500 NUM_PROC=10 PER_WORKER_MIB=3300 \
    RUN=$PICK CELL=$CELL DEST=$DEST LOG_TAG=realcell_v2 \
    setsid bash scripts/collect_threading_realcell_batches.sh > logs/realcell_v2_collect_a.log 2>&1 </dev/null &
  A=$!
  until [ -f "$DEST/git_diff.patch" ]; do sleep 5; done
  GPU=${FREE[1]} BATCH_IDS="2 4 6 8" BATCH_TRAJ=12500 NUM_PROC=10 PER_WORKER_MIB=3300 \
    RUN=$PICK CELL=$CELL DEST=$DEST LOG_TAG=realcell_v2 \
    setsid bash scripts/collect_threading_realcell_batches.sh > logs/realcell_v2_collect_b.log 2>&1 </dev/null &
  B=$!; wait $A; wait $B
fi
n=$(ls "$DEST"/part*.hdf5 2>/dev/null | wc -l)
[ "$n" -ge 8 ] || die "collection produced only $n/8 shards"
say "B done: $n shards"

# --------------------------------------------------- C: two distillation arms
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F, '$2 < 8000 {print $1}' | tr -d ' ')
[ "${#FREE[@]}" -ge 2 ] || die "need 2 free GPUs for distillation"
say "C: distillation, base on GPU ${FREE[0]}, aug on GPU ${FREE[1]}"
CUDA_VISIBLE_DEVICES=${FREE[0]} setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_v2.yaml \
  --source_agent "$PICK/model0.pt" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_seed1 \
  > logs/realcell_v2_stage1_base.log 2>&1 </dev/null &
CUDA_VISIBLE_DEVICES=${FREE[1]} setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_aug_v2.yaml \
  --source_agent "$PICK/model0.pt" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_aug_seed1 \
  > logs/realcell_v2_stage1_aug.log 2>&1 </dev/null &
wait
say "all done"
