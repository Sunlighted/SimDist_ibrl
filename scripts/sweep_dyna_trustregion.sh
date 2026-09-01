#!/bin/bash
# Dyna stage-3 sweep: policy trust region (CLAUDE.md 42/43) + real_ratio.
#
# WHY: the existing dyna_h5 arm is NET HARMFUL -- paired over 20 runs, the
# 75-100k steady state is 0.132 vs 0.278 at round-0 (t=-3.72, p=0.001, 15/20
# runs worse). Every seed collapses to ~0.00 within the first 5k while q_mean
# RISES (0.57 -> 0.70) -- the DPG step walks the BC policy into model error.
# So this sweep is about STOPPING THE BLEEDING, not about tuning for gain.
#
# Design: one-at-a-time around the existing baseline (CLAUDE.md rule 9), so the
# 10-seed dyna_h5 arm IS the control cell -- no extra runs spent on it.
#   baseline: policy_lr=1e-5  bc_coef=0.1  real_ratio=0.5  src_anchor_coef=0
# Seeds 1/101/201 match the baseline's, and stage-2 BC is unaffected by every
# knob here, so round-0 is IDENTICAL per seed => the paired delta
# (steady - round0) is the read-out, which is what got t=-3.72 above.
#
# Usage:  bash scripts/sweep_dyna_trustregion.sh <worker_id> <gpu> [nworker]
#   or:   bash scripts/launch_dyna_sweep.sh          (starts all workers)
set -u
WORKER=${1:?worker id}
GPU=${2:?gpu id}
ROOT=/home/Chenyang/ibrl
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
STAGE1=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_sat_400000_s1/dataset.hdf5
NWORKER=${3:-6}

cd $ROOT
export PYTHONPATH=$ROOT OMP_NUM_THREADS=1
export MUJOCO_PY_MUJOCO_PATH=/home/Chenyang/.mujoco/mujoco210
export WANDB_MODE=offline          # gotcha: wandb online deadlocks long runs
export CUDA_VISIBLE_DEVICES=$GPU

# cell_name -> extra args (everything else is byte-identical to dyna_h5)
declare -A CELL=(
  [plr0]="--dyna.policy_lr 0.0"
  [plr1e6]="--dyna.policy_lr 1e-6"
  [plr3e6]="--dyna.policy_lr 3e-6"
  [bc1]="--dyna.bc_coef 1.0"
  [bc10]="--dyna.bc_coef 10.0"
  # Anchor coefficient is swept over 3 decades, NOT pinned at the stage-2 value of
  # ~1.0 (CLAUDE.md 42/43): there the anchor competes with a BC MSE of the same
  # units, here it competes with -Q, which lives in [0,1]. Smoke test measured
  # src_anchor_loss=9e-4 against |pi_loss|=0.635 at coef=1 -- a 0.1% contribution,
  # i.e. coef=1 is almost certainly a no-op here. CLAUDE.md rule 9: re-scan the
  # knob when the setting changes.
  [anc1]="--dyna.src_anchor_coef 1.0 --dyna.src_anchor_target stage2"
  [anc10]="--dyna.src_anchor_coef 10.0 --dyna.src_anchor_target stage2"
  [anc100]="--dyna.src_anchor_coef 100.0 --dyna.src_anchor_target stage2"
  # Same coefficient as anc10 so the pair isolates ONE thing: which frozen policy
  # the anchor points at (the post-stage-2 policy we are trying to protect, vs
  # literal pi_src). CLAUDE.md 43 says pi_src, but that was measured in stage-2
  # where the two coincide; rule 10 wants an arm that separates them.
  [ancsrc10]="--dyna.src_anchor_coef 10.0 --dyna.src_anchor_target src"
  [rr025]="--dyna.real_ratio 0.25"
  [rr075]="--dyna.real_ratio 0.75"
  [rr100]="--dyna.real_ratio 1.0"
  # stage-2 anchor (CLAUDE.md 42/43) -- already implemented, zero new code. The
  # dynens_h5_slow_anchor arm shows what it buys BEFORE stage-3 even starts:
  # round-0 goes 0.274 -> 0.597. Held at the baseline's 400k demo set on purpose
  # so this stays a single-variable cell (the slow_anchor arm also swapped in the
  # 160k unsaturated-collector demos, CLAUDE.md 38, worth +0.143 on its own).
  [bcanc1]="--bc_anchor_coef 1.0"
)
ORDER=(plr0 plr1e6 plr3e6 bc1 bc10 anc1 anc10 anc100 ancsrc10 rr025 rr075 rr100 bcanc1)

# seed-major order: every cell gets its seed-1 run before any seed-101 starts,
# so a first read across all 10 cells is available after ~1/3 of the budget.
i=0
for SEED in 1 101 201; do
  for CELL_NAME in "${ORDER[@]}"; do
    if (( i % NWORKER == WORKER )); then
      RUN=dyna_${CELL_NAME}_seed${SEED}
      DIR=exps/simdist_threading_3arm/$RUN
      if [[ -f $DIR/train.log ]] && grep -q "^100000: score/score" $DIR/train.log 2>/dev/null; then
        echo "[w$WORKER] SKIP $RUN (already complete)"
      else
        echo "[w$WORKER] $(date '+%m-%d %H:%M') START $RUN  ${CELL[$CELL_NAME]}"
        $PY train_simdist_dyna.py \
          --stage1_ckpt $STAGE1 --target_demo_path $DEMOS \
          --save_dir $DIR --seed $SEED \
          --max_transitions 100000 --eval_every_rounds 5 \
          --num_eval_episode 100 --num_eval_proc 10 \
          --eval_hand_init_perturb_joint_sigma 0.1 \
          ${CELL[$CELL_NAME]} > $DIR.launch.log 2>&1
        echo "[w$WORKER] $(date '+%m-%d %H:%M') DONE  $RUN rc=$?"
      fi
    fi
    i=$((i+1))
  done
done
echo "[w$WORKER] all tasks finished"
