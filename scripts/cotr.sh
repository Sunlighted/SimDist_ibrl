#!/bin/bash

# ==========================================
# 1. Weighted (τ=0.01) Tasks
# ==========================================
echo "Launching Weighted tasks..."

N_SRC=200 METHOD=weighted_bc_n200 GPU=0 bash scripts/train_weighted_bc_threading.sh &
N_SRC=300 METHOD=weighted_bc_n300 GPU=1 bash scripts/train_weighted_bc_threading.sh &

# ==========================================
# 2. Unweighted (τ=1e6) Tasks
# ==========================================
echo "Launching Unweighted tasks..."

N_SRC=1200 GAP_TAU=1000000 METHOD=cotrain_unweighted_n1200 GPU=2 bash scripts/train_weighted_bc_threading.sh &
N_SRC=800 GAP_TAU=1000000 METHOD=cotrain_unweighted_n800 GPU=0 bash scripts/train_weighted_bc_threading.sh &

# ==========================================
# 等待所有后台并行任务结束
# ==========================================
echo "All tasks are running in parallel. Waiting for them to finish..."
wait

echo "All training tasks completed!"

python evaluate/eval.py --weight /home/Chenyang/ibrl/exps/mimicgen/threading_residual_from_recovery_seed1/model0.pt --mode residual --num_game 500 --shift_from_cfg 1 --strict_success 1 --strict_dwell_steps 3 --strict_max_tripod_disp 0.04 --hand_init_perturb_joint_sigma 0.1 --no_render 1