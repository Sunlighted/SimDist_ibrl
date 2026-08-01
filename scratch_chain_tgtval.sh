#!/bin/bash
# Wait for the source-value dynamics-adapt run (PID $FIRST_PID) to finish, then
# launch the TARGET-value comparison run on the same GPU. Emits marker lines so a
# Monitor can track the handoff.
set -u
FIRST_PID=3469132
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl && source set_env.sh 2>/dev/null

echo "[chain] waiting for source-value run PID $FIRST_PID to finish..."
while kill -0 "$FIRST_PID" 2>/dev/null; do sleep 60; done
echo "[chain] source-value run finished; launching TARGET-value run on GPU2"
sleep 10  # let GPU memory free

CUDA_VISIBLE_DEVICES=2 python train_dynamics_adapt.py \
  --dyn_ckpt exps/mimicgen/threading_dyn_diverse_v2_seed1/best.pt \
  --source_agent exps/mimicgen/threading_ibrl_newtgt_seed1/model0.pt \
  --save_dir exps/mimicgen/threading_dyn_adapt_tgtval_seed1 \
  --bc_steps 1000 --adapt_interval 1000 --max_transitions 40000 \
  --dyn_update_steps 1000 --num_eval_episode 500 --num_eval_proc 10 \
  --eval_hand_init_perturb_joint_sigma 0.1 --use_wb 1 \
  > exps_dyn_adapt_tgtval_seed1.log 2>&1
echo "[chain] TARGET-value run exited with code $?"
