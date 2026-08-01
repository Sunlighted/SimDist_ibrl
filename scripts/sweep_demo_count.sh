#!/usr/bin/env bash
# Data-scaling sweep: how few demos train an MLP-BC to ~target success?
# Fixes head/hparams to the baseline cfg, varies ONLY dataset.num_data, then
# 500-game strict TARGET eval (target shift + hand_init sigma 0.1). Prints a
# success-vs-N table so you can read off the minimum N that hits the threshold.
#
#   DATA=release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5 \
#   TAG=expert GPU=0 NUMS="20 50 100 200 350 500" bash scripts/sweep_demo_count.sh
#
# Run two sweeps (expert vs recovery) on two GPUs in parallel by launching this
# twice with different DATA/TAG/GPU.
set -u
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
conda activate ibrl
export PYTHONPATH=/home/Chenyang/ibrl:${PYTHONPATH:-} MUJOCO_GL=egl
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

DATA="${DATA:-release/data/mimicgen/threading_d0_target_expert_n500/dataset.hdf5}"
TAG="${TAG:-expert}"
BASECFG="${BASECFG:-exps/mimicgen/threading_target_only_n60_recovery_AC40_seed1/cfg.yaml}"
SEED="${SEED:-1}"
GPU="${GPU:-0}"
NUMS="${NUMS:-20 50 100 200 350 500}"
NGAME="${NGAME:-500}"
MP="${MP:-25}"
EPOCH="${EPOCH:-2000}"          # set 1000 for a faster coarse pass
EVAL_LEN="${EVAL_LEN:-401}"     # 401 is safe for any success-only set
export CUDA_VISIBLE_DEVICES="$GPU"

SUMMARY="exps/mimicgen/demoscale_${TAG}_seed${SEED}_summary.txt"
echo "# data-scaling sweep | tag=$TAG seed=$SEED data=$DATA" | tee "$SUMMARY"
echo "# N  success(500g, target+sigma0.1, strict)" | tee -a "$SUMMARY"

for n in $NUMS; do
  SAVE="exps/mimicgen/threading_demoscale_${TAG}_n${n}_seed${SEED}"
  TLOG="/tmp/scale_${TAG}_n${n}_seed${SEED}_train.log"
  ELOG="/tmp/scale_${TAG}_n${n}_seed${SEED}_eval.log"
  echo "=== [N=$n] train -> $SAVE ==="
  python -u train_bc.py --config_path "$BASECFG" \
    --dataset.path "$DATA" --dataset.num_data "$n" \
    --dataset.eval_episode_len "$EVAL_LEN" \
    --num_epoch "$EPOCH" --seed "$SEED" --save_dir "$SAVE" \
    > "$TLOG" 2>&1
  if [ ! -f "$SAVE/model0.pt" ]; then
    echo "N=$n  TRAIN_FAILED (see $TLOG)" | tee -a "$SUMMARY"; continue
  fi
  echo "=== [N=$n] 500-game strict TARGET eval ==="
  python -u evaluate/eval.py --weight "$SAVE/model0.pt" --mode bc \
    --num_game "$NGAME" --mp "$MP" --seed 1 --shift_from_cfg 1 \
    --strict_success 1 --hand_init_perturb_joint_sigma 0.1 \
    > "$ELOG" 2>&1
  rate=$(grep -i 'success rate' "$ELOG" | tail -1)
  echo "N=$n  ${rate:-EVAL_FAILED}" | tee -a "$SUMMARY"
done

echo "=========================================="
echo "summary -> $SUMMARY"
cat "$SUMMARY"
