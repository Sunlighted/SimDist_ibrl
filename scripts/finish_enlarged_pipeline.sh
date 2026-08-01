#!/bin/bash
# Generation already finished (1000 demos, but MimicGen nested the output one
# folder deeper than the orchestrator expected). Resume: convert -> train the
# enlarged BC -> launch the enlarged RL (800k steps) with that BC as prior.
set -e
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl && source set_env.sh

DEMO=release/data/mimicgen/generated/threading_enlarged/threading_enlarged/demo.hdf5
OUT=release/data/mimicgen/generated/threading_enlarged
PROC=$OUT/processed_data96.hdf5

if [ ! -f "$PROC" ]; then
  echo "[fin] converting enlarged demos -> state format ..."
  python -u tools/dataset_states_to_obs.py --dataset "$DEMO" --output_folder "$OUT" --done_mode 2
fi
[ -f "$PROC" ] || { echo "[fin] ERROR: $PROC missing after conversion"; exit 1; }
echo "[fin] converted -> $PROC"

echo "[fin] training enlarged flow BC on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 python -u train_bc.py \
  --config_path release/cfgs/mimicgen/threading_enlarged_state_flow_bc.yaml \
  --save_dir exps/mimicgen/threading_enlarged_flow_bc_seed1
[ -f exps/mimicgen/threading_enlarged_flow_bc_seed1/model0.pt ] || { echo "[fin] ERROR: enlarged BC model0.pt missing"; exit 1; }
echo "[fin] enlarged BC done."

echo "[fin] launching enlarged RL (800k) on GPU 2 ..."
CUDA_VISIBLE_DEVICES=2 nohup python -u train_rl.py \
  --config_path release/cfgs/mimicgen/threading_d0_state_recovery_ibrl_enlarged.yaml \
  --save_dir exps/mimicgen/threading_recovery_enlarged_seed1 \
  > exps_recovery_enlarged.log 2>&1 &
echo "[fin] DONE. enlarged RL launched PID=$! (800k steps). enlarged BC at exps/mimicgen/threading_enlarged_flow_bc_seed1/"
