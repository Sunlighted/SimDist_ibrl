#!/bin/bash
# Residual off-policy RL (ResFiT-style) on a frozen TARGET-demo BC base policy,
# for Threading_D0 sim2sim dynamics-gap transfer.
#
# a = clip(base(s) + delta(s), -1, 1); base = a frozen BC policy trained on the 20
# TARGET demos; delta = a small residual learned by off-policy RL ONLINE in the
# SOURCE sim, GT-gap weighted (counterfactual replay in the TARGET sim). Deployed
# / evaluated on the TARGET. See rl/residual_agent.py + CLAUDE.md §20.
#
# Env vars:
#   SEED          random seed (default 1)
#   BASE_POLICY   frozen TARGET-demo BC ckpt (default threading_target_mlp_bc_n20_seed1)
#   STEPS         num_train_step (default 200000)
#   FORCE_C       <0 = GT-gap weighting (default -1); 1.0 = unweighted control arm
#   ACTION_SCALE  residual bound (default 0.1)
#   WARMUP        critic_warmup_steps (default 5000)
#   SIGMA         joint-space init noise σ for train+eval (default 0.1, matches the
#                 recovery / weighted-BC batch; set 0.02 for robosuite default)
#   RESUME        1 = continue from <SAVE_DIR>/latest.pt + train_state.pt (default 0)
#   Any extra args after the script are forwarded to train_residual.py.
#
# Examples:
#   bash scripts/train_residual_threading.sh                      # GT-gap weighted
#   FORCE_C=1.0 bash scripts/train_residual_threading.sh          # control: plain residual RL
#   RESUME=1 bash scripts/train_residual_threading.sh             # resume an interrupted run
#   RESUME=1 STEPS=2000000 bash scripts/train_residual_threading.sh   # resume + extend the budget
#
# Standalone re-eval of a saved ckpt on the TARGET:
#   python evaluate/eval.py --weight exps/mimicgen/threading_residual_seed1/latest.pt \
#       --mode residual --num_game 100 --shift_from_cfg 1 --strict_success 1
set -e

SEED=${SEED:-1}
BASE_POLICY=${BASE_POLICY:-exps/mimicgen/threading_target_mlp_bc_n20_seed1/model0.pt}
STEPS=${STEPS:-1000000}
FORCE_C=${FORCE_C:--1.0}
ACTION_SCALE=${ACTION_SCALE:-0.1}
WARMUP=${WARMUP:-5000}
SIGMA=${SIGMA:-0.1}   # joint-space init noise for train+eval (matches recovery batch)
RESUME=${RESUME:-0}   # 1 = continue from <SAVE_DIR>/latest.pt + train_state.pt
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_residual_bc_seed${SEED}}

python train_residual.py \
    --config_path release/cfgs/mimicgen/threading_d0_state_residual.yaml \
    --seed "${SEED}" \
    --residual.base_policy_path "${BASE_POLICY}" \
    --residual.force_c "${FORCE_C}" \
    --residual.action_scale "${ACTION_SCALE}" \
    --residual.critic_warmup_steps "${WARMUP}" \
    --train_hand_init_perturb_joint_sigma "${SIGMA}" \
    --eval_hand_init_perturb_joint_sigma "${SIGMA}" \
    --num_train_step "${STEPS}" \
    --resume "${RESUME}" \
    --save_dir "${SAVE_DIR}" \
    "$@"
