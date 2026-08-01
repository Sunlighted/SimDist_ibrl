#!/usr/bin/env bash
# Data-efficiency sweep for flow BC on Threading_D0 SOURCE.
# Trains one flow BC per demo-count, eval pinned to SOURCE (mode=none) so the
# best-ckpt selection reflects in-distribution success, not the (old, clip-
# through-prone) target. Network/hparams inherited from the rerun cfg; only
# dataset.num_data varies => clean success-vs-#demos curve.
#
# Usage (single point):
#   N=20 GPU=1 SEED=1 EPOCH=2000 bash scripts/sweep_flow_bc_datasize.sh
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh
set -eo pipefail

N=${N:?set N=demo count}
GPU=${GPU:-0}
SEED=${SEED:-1}
EPOCH=${EPOCH:-2000}
DATA_PATH=${DATA_PATH:-}      # empty => use base cfg's dataset.path
EVAL_SIGMA=${EVAL_SIGMA:-}    # empty => use base cfg's eval init; else joint-space gaussian σ
TAG=${TAG:-}                  # save_dir / wandb-run suffix, e.g. _sigma0.1
BASE_CFG=exps/mimicgen/threading_flow_bc_seed1/cfg.yaml
SAVE=exps/mimicgen/threading_flow_bc${TAG}_n${N}_seed${SEED}

EXTRA=()
[[ -n "$DATA_PATH"  ]] && EXTRA+=(--dataset.path "$DATA_PATH")
[[ -n "$EVAL_SIGMA" ]] && EXTRA+=(--eval_hand_init_perturb_joint_sigma "$EVAL_SIGMA")

echo "[sweep] N=${N} seed=${SEED} gpu=${GPU} epoch=${EPOCH} tag=${TAG} eval_sigma=${EVAL_SIGMA:-default} -> ${SAVE}"
CUDA_VISIBLE_DEVICES=${GPU} python train_bc.py \
  --config_path ${BASE_CFG} \
  --dataset.num_data ${N} \
  --eval_domain_shift.mode none \
  --num_epoch ${EPOCH} \
  --seed ${SEED} \
  --use_wb 1 \
  "${EXTRA[@]}" \
  --save_dir ${SAVE}
