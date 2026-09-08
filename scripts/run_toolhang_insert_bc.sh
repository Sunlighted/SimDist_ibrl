#!/bin/bash
# Launch both ToolHangInsert BC priors (single-step MLP + BC-RNN-GMM) in parallel
# on two GPUs. The better of the two becomes the IBRL prior.
set -u
cd /home/Chenyang/ibrl
source set_env.sh
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
mkdir -p logs exps/robomimic

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline setsid $PY train_bc.py \
  --config_path release/cfgs/robomimic/toolhang_insert_state_mlp_bc.yaml \
  --save_dir exps/robomimic/toolhang_insert_mlp_bc_seed1 \
  > logs/toolhang_insert_mlp_bc_seed1.log 2>&1 < /dev/null & disown

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline setsid $PY train_bc.py \
  --config_path release/cfgs/robomimic/toolhang_insert_state_rnn_bc.yaml \
  --save_dir exps/robomimic/toolhang_insert_rnn_bc_seed1 \
  > logs/toolhang_insert_rnn_bc_seed1.log 2>&1 < /dev/null & disown

echo "launched: logs/toolhang_insert_{mlp,rnn}_bc_seed1.log"
