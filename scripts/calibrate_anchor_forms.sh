#!/bin/bash
# Phase 0 for the anchor-loss ablation: find, per FORM, the coefficient that puts
# it at the SAME working point as the deployed anchor (F1, bc_anchor_coef 1.0).
#
# Why this is not optional: the form axis and the strength axis are confounded
# unless every arm is pinned to the same D_SRC (= mean|pi_theta(s_src) -
# pi_src(s_src)|). simdist 43 built 53 of these cells for exactly that reason,
# and its gotcha #1 is F5 first running 1400x too strong because a saturating
# form's stiffness is lambda/tau, not lambda. The published lambdas were
# calibrated on the SINGLE-head stage-1 at 40k; this batch runs on the 5-head
# stage-1, where stage-2 starts from a different policy head, so D_SRC(lambda)
# moves and they have to be re-measured.
#
# Each cell = stage-2 + the round-0 eval only (--max_transitions 0), ~8 min.
# Reads [D_SRC] <d> (L2mean ..) bc_loss <b> out of train.log.
#
#   bash scripts/calibrate_anchor_forms.sh          # full sweep
#   GPUS=0,1 bash scripts/calibrate_anchor_forms.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}

OUT=${OUT:-exps/anchor_cal}
CK=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
DEMO=release/data/mimicgen/threading_d0_sat_160000_s1/dataset.hdf5
POOL=${POOL:-release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5}
GPUS=${GPUS:-0,1,2}
IFS=',' read -r -a G <<< "$GPUS"
mkdir -p "$OUT"

for f in "$CK" "$DEMO" "$POOL"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done

# tag|extra flags   -- F1 first: it defines the target D_SRC.
CELLS=(
  "f1_lam1.0|--bc_anchor_coef 1.0"
  "f2wd_1e-5|--bc_anchor_coef 0.0 --bc_weight_decay 1e-5 --bc_measure_dsrc 1"
  "f2wd_1e-4|--bc_anchor_coef 0.0 --bc_weight_decay 1e-4 --bc_measure_dsrc 1"
  "f2wd_1e-3|--bc_anchor_coef 0.0 --bc_weight_decay 1e-3 --bc_measure_dsrc 1"
  "f3l2_0.1|--bc_anchor_coef 0.0 --bc_l2init_coef 0.1 --bc_measure_dsrc 1"
  "f3l2_0.25|--bc_anchor_coef 0.0 --bc_l2init_coef 0.25 --bc_measure_dsrc 1"
  "f3l2_1.0|--bc_anchor_coef 0.0 --bc_l2init_coef 1.0 --bc_measure_dsrc 1"
  "f4a_0.15|--bc_anchor_coef 0.15 --bc_anchor_eps 0.05"
  "f4a_0.3|--bc_anchor_coef 0.3 --bc_anchor_eps 0.05"
  "f4a_0.5|--bc_anchor_coef 0.5 --bc_anchor_eps 0.05"
  "f4a_2|--bc_anchor_coef 2 --bc_anchor_eps 0.05"
  "f4a_10|--bc_anchor_coef 10 --bc_anchor_eps 0.05"
  "f4b_100|--bc_anchor_coef 100 --bc_anchor_eps 0.20"
  "f4b_300|--bc_anchor_coef 300 --bc_anchor_eps 0.20"
  "f5_0.013|--bc_anchor_coef 0.013 --bc_anchor_sat 0.03"
  "f5_0.05|--bc_anchor_coef 0.05 --bc_anchor_sat 0.03"
  "f5_0.2|--bc_anchor_coef 0.2 --bc_anchor_sat 0.03"
  "dat_0.3|--bc_anchor_coef 0.3 --bc_anchor_data_actions 1"
  "dat_1.0|--bc_anchor_coef 1.0 --bc_anchor_data_actions 1"
  "dat_3.0|--bc_anchor_coef 3.0 --bc_anchor_data_actions 1"
  "es_src_0.3|--bc_anchor_coef 0.3 --bc_anchor_expert_only 1 --bc_anchor_success_only 1"
  "es_src_1.0|--bc_anchor_coef 1.0 --bc_anchor_expert_only 1 --bc_anchor_success_only 1"
  "es_src_3.0|--bc_anchor_coef 3.0 --bc_anchor_expert_only 1 --bc_anchor_success_only 1"
  "es_dat_0.3|--bc_anchor_coef 0.3 --bc_anchor_expert_only 1 --bc_anchor_success_only 1 --bc_anchor_data_actions 1"
  "es_dat_1.0|--bc_anchor_coef 1.0 --bc_anchor_expert_only 1 --bc_anchor_success_only 1 --bc_anchor_data_actions 1"
  "es_dat_3.0|--bc_anchor_coef 3.0 --bc_anchor_expert_only 1 --bc_anchor_success_only 1 --bc_anchor_data_actions 1"
  "es_dat_10|--bc_anchor_coef 10 --bc_anchor_expert_only 1 --bc_anchor_success_only 1 --bc_anchor_data_actions 1"
  "es_dat_30|--bc_anchor_coef 30 --bc_anchor_expert_only 1 --bc_anchor_success_only 1 --bc_anchor_data_actions 1"
)

run_cell() {  # run_cell <gpu> <tag> <flags...>
  local gpu=$1 tag=$2; shift 2
  [ -f "$OUT/$tag/train.log" ] && grep -q "\[D_SRC\]" "$OUT/$tag/train.log" 2>/dev/null && {
      echo "[skip] $tag"; return; }
  CUDA_VISIBLE_DEVICES=$gpu $PY -u train_simdist_adapt.py \
    --stage1_ckpt "$CK" --save_dir "$OUT/$tag" --seed 1 \
    --target_demo_path "$DEMO" --target_num_data 20 \
    --bc_anchor_path "$POOL" \
    --max_transitions 0 --num_eval_episode 10 --num_eval_proc 10 \
    --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
    --collect_disagree_coef 0.5 --original_mppi 1 \
    --use_wb 0 --state_only_no_render 1 $@ > "$OUT/$tag.log" 2>&1
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
echo "== CALIBRATION DONE =="
$PY scripts/report_anchor_cal.py "$OUT"
