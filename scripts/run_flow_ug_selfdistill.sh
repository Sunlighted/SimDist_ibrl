#!/usr/bin/env bash
# Minimal experiment: does self-distilling flow+UG successes back into the flow
# policy improve flow-only / flow+UG on the TARGET? (CLAUDE.md §24 follow-up.)
#
# Pipeline:
#   1. collect N successful flow+UG rollouts on TARGET (teacher = flow.pt + UG)
#   2. self-distill them (+ 20 anchor demos) into a fresh flow from flow_source.pt
#   3. eval 4 arms x 500 games on TARGET (strict success), Ta=1:
#        original flow-only / original flow+UG / v2 flow-only / v2 flow+UG
#
# Env knobs (defaults in []):
#   SEED[1] GPU[0] N_SUCCESS[300] NUM_ENVS[10] COEFF[0.3] TARGET_RATIO[0.25]
#   FT_STEPS[20000] EVAL_GAME[500]
set -e
cd "$(dirname "$0")/.."
source set_env.sh 2>/dev/null || true

SEED=${SEED:-1}
GPU=${GPU:-0}
N_SUCCESS=${N_SUCCESS:-300}
NUM_ENVS=${NUM_ENVS:-10}
COEFF=${COEFF:-0.3}
TARGET_RATIO=${TARGET_RATIO:-0.25}
FT_STEPS=${FT_STEPS:-20000}
EVAL_GAME=${EVAL_GAME:-500}

FLOW_DIR=exps/mimicgen/threading_flow_ug_seed${SEED}
FLOW=${FLOW_DIR}/flow.pt
SOURCE_FLOW=${FLOW_DIR}/flow_source.pt
WM=exps/mimicgen/threading_simdist_adapt_seed${SEED}/model0.pt
RUN_DIR=exps/mimicgen/threading_flow_ug_selfdistill_seed${SEED}
V2_DIR=${RUN_DIR}/flow_v2
COLLECTED=${RUN_DIR}/collected_chunks.pt
V2_FLOW=${V2_DIR}/flow.pt

export CUDA_VISIBLE_DEVICES=${GPU}
mkdir -p "${RUN_DIR}"

echo "============================================================"
echo "[1/3] collect ${N_SUCCESS} flow+UG successes on TARGET (coeff=${COEFF})"
echo "============================================================"
python collect_flow_ug_rollouts.py \
    --flow "${FLOW}" --wm_ckpt "${WM}" \
    --steering_coeff "${COEFF}" --num_success "${N_SUCCESS}" \
    --num_envs "${NUM_ENVS}" --seed "${SEED}" \
    --out_dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/collect.log"

echo "============================================================"
echo "[2/3] self-distill into flow_v2 (target_ratio=${TARGET_RATIO})"
echo "============================================================"
python finetune_flow_selfdistill.py \
    --source_flow "${SOURCE_FLOW}" --collected "${COLLECTED}" \
    --save_dir "${V2_DIR}" --steps "${FT_STEPS}" \
    --eval_every 1000 --num_eval_game 100 \
    --target_ratio "${TARGET_RATIO}" --seed "${SEED}" 2>&1 | tee "${RUN_DIR}/finetune.log"

echo "============================================================"
echo "[3/3] 4-arm eval x ${EVAL_GAME} games on TARGET"
echo "============================================================"
run_eval () {  # $1=label $2=flow $3=extra_ug_args
    echo "----- ${1} -----"
    python eval_flow_ug.py --flow "${2}" --ta_eval 1 \
        --shift target --num_game "${EVAL_GAME}" --seed "${SEED}" ${3} \
        2>&1 | tee "${RUN_DIR}/eval_${1}.log" | grep -E "\[result\]|\[setup\]"
}

run_eval "orig_flow_only"  "${FLOW}"    ""
run_eval "orig_flow_ug"    "${FLOW}"    "--wm_ckpt ${WM} --use_ug 1 --steering_coeff ${COEFF}"
run_eval "v2_flow_only"    "${V2_FLOW}" ""
run_eval "v2_flow_ug"      "${V2_FLOW}" "--wm_ckpt ${WM} --use_ug 1 --steering_coeff ${COEFF}"

echo "============================================================"
echo "[done] results collected under ${RUN_DIR}/eval_*.log"
echo "  compare: orig_flow_only(~58) v2_flow_only | orig_flow_ug(~79) v2_flow_ug"
echo "============================================================"
