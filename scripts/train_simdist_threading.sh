#!/bin/bash
# SimDist baseline (state-based): a TD-MPC2-style latent world model + MPPI,
# distilled in the SOURCE sim then finetuned (dynamics+policy) on TARGET demos.
# (rl/world_model.py + rl/simdist_planner.py + rl/simdist_agent.py +
#  train_simdist.py; CLAUDE.md §17.)
#
# Two stages back-to-back in one run:
#   Stage 1 (SOURCE, online): distill encoder/dynamics/reward/value/policy. Data
#     is collected MIXED — frozen SOURCE expert (+noise) for the first
#     EXPERT_COLLECT_STEPS env steps, then the world model's own MPPI planner.
#   Stage 2 (offline): finetune ONLY dynamics + policy heads on the TARGET demos.
# Execution/eval = MPPI on the world model; TARGET strict-success is the metric.
#
# Env vars:
#   SEED                  random seed (default 1)
#   SOURCE_AGENT          frozen source expert for stage-1 collection
#                         (default threading_ibrl_source_seed1; "none" = pure on-policy)
#   EXPERT_COLLECT_STEPS  expert-guided env steps before MPPI self-collects (default 100000)
#   DISTILL_STEPS         stage-1 online env steps (default 200000)
#   FINETUNE_STEPS        stage-2 offline steps (default 30000)
#   Any extra args after the script are forwarded to train_simdist.py.
#
# Examples:
#   bash scripts/train_simdist_threading.sh                              # default (mixed collect)
#   EXPERT_COLLECT_STEPS=0 SOURCE_AGENT=none bash scripts/train_simdist_threading.sh  # pure on-policy
#   bash scripts/train_simdist_threading.sh --simdist.finetune_policy 0  # finetune dynamics only
#   SEED=2 bash scripts/train_simdist_threading.sh                       # another seed
#
# Re-score a saved ckpt standalone:
#   python evaluate/eval.py --weight <save_dir>/latest.pt --mode simdist \
#       --num_game 100 --strict_success 1
set -e

SEED=${SEED:-1}
SOURCE_AGENT=${SOURCE_AGENT:-exps/mimicgen/threading_ibrl_source_seed1/model0.pt}
EXPERT_COLLECT_STEPS=${EXPERT_COLLECT_STEPS:-100000}
DISTILL_STEPS=${DISTILL_STEPS:-200000}
FINETUNE_STEPS=${FINETUNE_STEPS:-30000}
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_simdist_seed${SEED}}

python train_simdist.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_simdist.yaml \
    --seed "${SEED}" \
    --source_agent "${SOURCE_AGENT}" \
    --expert_collect_steps "${EXPERT_COLLECT_STEPS}" \
    --distill_steps "${DISTILL_STEPS}" \
    --simdist.finetune_steps "${FINETUNE_STEPS}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
