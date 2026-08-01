#!/bin/bash
# Auto-continue: wait for the Threading_Enlarged MimicGen generation to finish,
# convert the demos to ibrl state format, then train the enlarged flow BC prior.
set -e
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl && source set_env.sh

GEN_PID=${1:-1093627}
GEN_DIR=release/data/mimicgen/generated/threading_enlarged
DEMO=$GEN_DIR/demo.hdf5

echo "[orch] waiting for generation PID $GEN_PID ..."
while kill -0 "$GEN_PID" 2>/dev/null; do sleep 60; done
echo "[orch] generation finished. checking output ..."
sleep 5
[ -f "$DEMO" ] || { echo "[orch] ERROR: $DEMO missing — generation likely failed"; exit 1; }
NDEMO=$(python -c "import h5py;print(len(h5py.File('$DEMO','r')['data'].keys()))" 2>/dev/null)
echo "[orch] $DEMO has $NDEMO demos"

echo "[orch] converting to state format ..."
python -u tools/dataset_states_to_obs.py --dataset "$DEMO" \
  --output_folder "$GEN_DIR" --done_mode 2
PROC=$GEN_DIR/processed_data96.hdf5
[ -f "$PROC" ] || { echo "[orch] ERROR: conversion produced no $PROC"; exit 1; }
echo "[orch] conversion done -> $PROC"

# pick the GPU with the least memory used
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | awk '{print $1}' | tr -d ',')
echo "[orch] training enlarged flow BC on GPU $GPU ..."
CUDA_VISIBLE_DEVICES=$GPU python -u train_bc.py \
  --config_path release/cfgs/mimicgen/threading_enlarged_state_flow_bc.yaml \
  --save_dir exps/mimicgen/threading_enlarged_flow_bc_seed1
echo "[orch] DONE. enlarged BC at exps/mimicgen/threading_enlarged_flow_bc_seed1/"
