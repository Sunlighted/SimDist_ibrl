#!/usr/bin/env bash
# Auto-chain: wait for the sigma=0.1 RNN-prior 1M training to FINISH, then
# generate the Square SimDist diverse dataset (200k trajs = 2x Threading's 100k),
# same method as threading_d0_source_diverse_v2 (generate_simdist_data.py:
# expert + snapshots, HYBRID rollout, p_expert 0.5, hand_init_sigma 0.1).
#
#   nohup bash scripts/auto_collect_square_diverse.sh > exps_auto_collect_square.log 2>&1 &
set -uo pipefail
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
conda activate ibrl
source set_env.sh

RUN=exps/mimicgen/square_ibrl_source_rnnprior_sig01_1000k_seed1
CFG_TAG=square_d0_state_ibrl_source_rnnprior_sig01.yaml
FINAL_SNAP="$RUN/snapshot_1000000.pt"

echo "[auto-collect] waiting for training to finish ($FINAL_SNAP + process gone)..."
# Training is DONE when the final (1M-step) snapshot exists AND no train_rl
# process for this config is still running. If the run crashes before 1M, the
# final snapshot never appears and this waits (safe: won't collect off a
# half-trained model). Re-check every 60s.
while true; do
  if [ -f "$FINAL_SNAP" ] && ! pgrep -f "$CFG_TAG" >/dev/null 2>&1; then
    break
  fi
  sleep 60
done
echo "[auto-collect] training finished at $(date -u +%H:%M:%S 2>/dev/null || echo '?'). starting generation."

# Reuse the (task-agnostic) threading generator wrapper with Square vars.
# hand_init_sigma is hardcoded to 0.1 inside it (matches the request).
# snapshots are read from $RUN/snapshot_*.pt (the 20 sigma=0.1 snapshots).
GPU="${GPU:-1}" \
RUN="$RUN" \
EXPERT="$RUN/model0.pt" \
OUT="release/data/mimicgen/square_d0_source_diverse_v2" \
NUM_TRAJ="${NUM_TRAJ:-200000}" \
NUM_PROC="${NUM_PROC:-20}" \
P_EXPERT="${P_EXPERT:-0.5}" \
HYBRID=1 SNAP_HYBRID=1 SEED=1 \
  bash scripts/generate_simdist_data_threading.sh

echo "[auto-collect] DONE. dataset at release/data/mimicgen/square_d0_source_diverse_v2"
