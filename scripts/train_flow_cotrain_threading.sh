#!/usr/bin/env bash
# Flow-head BC co-training on Threading_D0: a LOT of SOURCE demos
# (threading_d0_source_sigma0.1) + the 20 TARGET demos (threading_d0_target_dynamics)
# mixed at a fixed per-batch ratio, fed to a StateFlowPolicy. Launches TWO runs:
#   - WEIGHTED   : source samples drawn ∝ GT-gap confidence c=exp(-gap/GAP_TAU)
#   - UNWEIGHTED : gap_tau=1e6 ⇒ all source weights ≈ 1 (plain co-training)
# Both evaluate on SOURCE + TARGET every eval period and SAVE by the TARGET score
# (TopkSaver ranks on target; score_source is logged for tracking).
#
# Usage:
#   bash scripts/train_flow_cotrain_threading.sh                 # both, GPU 1 & 3
#   GPU_W=0 GPU_U=2 bash scripts/train_flow_cotrain_threading.sh # pick GPUs
#   ONLY=weighted   bash scripts/train_flow_cotrain_threading.sh # just one arm
#   ONLY=unweighted bash scripts/train_flow_cotrain_threading.sh
#   SRC=weak bash scripts/train_flow_cotrain_threading.sh        # BC-policy source data
#   FG=1 ONLY=weighted bash scripts/train_flow_cotrain_threading.sh  # foreground (no nohup)
#   SEED=2 bash scripts/train_flow_cotrain_threading.sh
#   bash scripts/train_flow_cotrain_threading.sh --num_epoch 1000    # extra CLI overrides
#
# Env vars:
#   GPU_W      CUDA device for the WEIGHTED run        (default 1)
#   GPU_U      CUDA device for the UNWEIGHTED run      (default 3)
#   SEED       training seed                           (default 1)
#   GAP_TAU    weighted-arm confidence sharpness       (default 0.01)
#   SRC        strong | weak (which source dataset)    (default strong)
#   ONLY       weighted | unweighted | both           (default both)
#   FG         1 = run foreground (only with ONLY=...) (default 0 = nohup bg)
#
# Re-eval a saved ckpt later as a flow BC:
#   python evaluate/eval.py --weight exps/mimicgen/threading_flow_cotrain_weighted_seed1/model0.pt \
#       --mode weighted_bc --num_game 100 --shift_from_cfg 1 --strict_success 1   # target
#   python evaluate/eval.py --weight exps/mimicgen/threading_flow_cotrain_weighted_seed1/model0.pt \
#       --mode weighted_bc --num_game 100 --shift_mode none --strict_success 1     # source

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
conda activate ibrl
source set_env.sh

GPU_W="${GPU_W:-1}"
GPU_U="${GPU_U:-3}"
SEED="${SEED:-1}"
GAP_TAU="${GAP_TAU:-0.01}"
SRC="${SRC:-strong}"
ONLY="${ONLY:-both}"
FG="${FG:-0}"

if [[ "$SRC" == "weak" ]]; then
    CONFIG="release/cfgs/mimicgen/threading_d0_state_weighted_flow_weak.yaml"
    TAG="weak"
else
    CONFIG="release/cfgs/mimicgen/threading_d0_state_weighted_flow_strong.yaml"
    TAG="strong"
fi

run_arm() {
    # $1 = arm name (weighted|unweighted), $2 = gpu, $3 = gap_tau
    local arm="$1" gpu="$2" tau="$3"
    local save_dir="exps/mimicgen/threading_flow_cotrain_${arm}_${TAG}_seed${SEED}"
    local log="${save_dir//\//_}.log"
    echo "==> [$arm] gpu=$gpu gap_tau=$tau src=$TAG save_dir=$save_dir"
    if [[ "$FG" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" python train_weighted_bc.py \
            --config_path "$CONFIG" --seed "$SEED" --gap_tau "$tau" \
            --save_dir "$save_dir" "${EXTRA[@]}"
    else
        CUDA_VISIBLE_DEVICES="$gpu" nohup python train_weighted_bc.py \
            --config_path "$CONFIG" --seed "$SEED" --gap_tau "$tau" \
            --save_dir "$save_dir" "${EXTRA[@]}" > "$log" 2>&1 &
        echo "    -> PID $! ; log: $log"
    fi
}

EXTRA=("$@")

case "$ONLY" in
    weighted)   run_arm weighted   "$GPU_W" "$GAP_TAU" ;;
    unweighted) run_arm unweighted "$GPU_U" 1000000 ;;
    both)
        run_arm weighted   "$GPU_W" "$GAP_TAU"
        run_arm unweighted "$GPU_U" 1000000
        ;;
    *) echo "ONLY must be weighted|unweighted|both (got '$ONLY')" >&2; exit 1 ;;
esac

if [[ "$FG" != "1" && "$ONLY" == "both" ]]; then
    echo "==> both launched in background. tail logs with:"
    echo "    tail -f exps_mimicgen_threading_flow_cotrain_weighted_${TAG}_seed${SEED}.log"
fi
