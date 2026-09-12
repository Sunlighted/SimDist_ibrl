#!/bin/bash
# Phase 0 for the F1-vs-F5 comparison on Square and Coffee: find, per task, the
# F5 lambda that puts it at the SAME held-out D_SRC as that task's F1 cell.
#
# ⚠️ F5's small-deviation stiffness is lambda/tau, NOT lambda (simdist 43's
# gotcha #1: it once ran 1400x too strong because the F1 habit of lambda~1 was
# copied over). Threading's calibrated pair is tau 0.03 / lambda 0.013 -- a
# Threading number. Each task gets its own sweep.
#
# The F1 cell defines the target and uses that task's OWN baseline lambda
# (Square 0.1, Coffee 1.0 -- a 10x spread, which is the non-transferability the
# lambda-tuning line A documents).
#
# Every cell runs --bc_anchor_holdout 1 and is judged on HELD-OUT D_SRC: the
# held-out pool is a uniform full draw, never provenance-subset, so it is the one
# metric under which forms are comparable (see calibrate_anchor_forms.sh).
#
#   bash scripts/calibrate_f5_sqcof.sh            # both tasks
#   TASKS=square GPUS=0,2 bash scripts/calibrate_f5_sqcof.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}

OUT=${OUT:-exps/anchform_cal}
GPUS=${GPUS:-0,2,3}
TAU=${TAU:-0.03}
IFS=',' read -r -a G <<< "$GPUS"
TASKS=${TASKS:-"square coffee"}
mkdir -p "$OUT"

spec() {  # spec <task> -> CK|DEMO|POOL|F1LAM|DIS|PSET
  case $1 in
    square) echo "exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/latest.pt|release/data/mimicgen/square_sat_400000_s1/dataset.hdf5|release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5|0.1|0,0.05,0.1,0.3|--planner_set init_std=0.15,base_policy_std=0.0" ;;
    coffee) echo "exps/mimicgen/coffee_s1g5x_dynens/latest.pt|release/data/mimicgen/coffee_g5x_sat400k_s1/dataset.hdf5|release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5|1.0|0,0.01,0.05,0.1|" ;;
  esac
}

CELLS=()
for t in $TASKS; do
  IFS='|' read -r CK DEMO POOL F1LAM DIS PSET <<< "$(spec $t)"
  for f in "$CK" "$DEMO" "$POOL"; do
    [ -f "$f" ] || { echo "MISSING ($t) $f"; exit 1; }
  done
  CELLS+=("${t}_f1|--bc_anchor_coef $F1LAM")
  # F1 lambda sweep: what lambda would put THIS task's deployed form at the same
  # held-out working point as the others? Square ships lambda 0.1 -> D_SRC 0.0747,
  # while Threading and Coffee ship 1.0 -> ~0.032. Measured, not extrapolated:
  # the power law D_SRC ~ lambda^-0.328 predicts 1.3 here, and 43's gotcha #2 is
  # three conclusions drawn from that kind of extrapolation, all later reversed.
  for lam in ${F1SWEEP:-}; do
    CELLS+=("${t}_f1lam_${lam}|--bc_anchor_coef $lam")
  done
  # 0.005/0.007/0.009 added after the first pass: Square's F1 sits at a much
  # looser working point (held-out D_SRC 0.0747 vs Coffee/Threading's ~0.032,
  # because its baseline lambda is 0.1 not 1.0), so its F5 match is below 0.013.
  # The 0.003->0.013 step moves D_SRC 0.233->0.032, a cliff -- the saturating
  # form loses its grip once d^2/tau is large -- so interpolation is not safe and
  # the region has to be sampled.
  for lam in 0.003 0.005 0.006 0.007 0.009 0.011 0.013 0.05 0.2; do
    CELLS+=("${t}_f5_${lam}|--bc_anchor_coef $lam --bc_anchor_sat $TAU")
  done
done

run_cell() {  # run_cell <gpu> <tag> <flags...>
  local gpu=$1 tag=$2; shift 2
  local task=${tag%%_*}
  IFS='|' read -r CK DEMO POOL F1LAM DIS PSET <<< "$(spec $task)"
  [ -f "$OUT/$tag/train.log" ] && grep -q "heldout_D_SRC" "$OUT/$tag/train.log" 2>/dev/null && {
      echo "[skip] $tag"; return; }
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_simdist_adapt.py \
    --stage1_ckpt "$CK" --save_dir "$OUT/$tag" --seed 1 \
    --target_demo_path "$DEMO" --bc_anchor_path "$POOL" --bc_anchor_holdout 1 \
    --max_transitions 0 --num_eval_episode 10 --num_eval_proc 10 \
    --eval_disagree_coefs "$DIS" \
    --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
    --collect_disagree_coef 0.5 --original_mppi 1 \
    --use_wb 0 --state_only_no_render 1 $PSET $@ > "$OUT/$tag.log" 2>&1
  echo "[done] $tag"
}

for i in "${!G[@]}"; do
  (
    for j in "${!CELLS[@]}"; do
      [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
      IFS='|' read -r tag flags <<< "${CELLS[$j]}"
      run_cell "${G[$i]}" "$tag" $flags
    done
  ) &
done
wait
echo "== F5 CALIBRATION DONE =="
$PY scripts/report_f5_cal.py "$OUT"
