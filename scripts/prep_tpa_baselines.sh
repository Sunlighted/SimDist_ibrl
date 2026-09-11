#!/bin/bash
# Build the LOCAL assets the TPA model-free baselines (iql, rlpd) need, so the
# cluster only ever sees small files. Mirrors scripts/prep_{square,coffee}_baselines.sh.
#
# Run on the workstation (needs the ibrl env, a GPU, and the 10.2 GB
# tpa_d0_source_diverse_v1). Both consumers of the diverse set -- the IQL offline
# pretrain and the RLPD merge -- happen here, so only ~300 MB ships.
#
# Produces:
#   exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt   IQL warm start (shared by all seeds)
#   release/data/mimicgen/tpa_rlpd_prior_s<N>/                RLPD prior set (one per seed)
#
# IBRL needs NO build step on TPA: its prior is the SOURCE EXPERT AGENT, attached
# through sgft_source_agent + source_agent_use_hybrid=1 (already in
# tpa_d0_state_ibrl_target.yaml), not a BC checkpoint. See cluster/job_tpa_6arm.sh.
#
# ⚠️ TPA episodes are 1000 steps, ~6x Threading's, so 2000 source episodes is a
# much bigger RLPD buffer here than on the other tasks. --source_episodes is
# exposed below; drop it if the merged files are unwieldy.
#
# Usage:  bash scripts/prep_tpa_baselines.sh [step]      step = iql | rlpd | all
set -e
cd "$(dirname "$0")/.."
# set_env.sh exports PYTHONPATH/LD_LIBRARY_PATH but does NOT activate conda.
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}

STEP=${1:-all}
SEEDS=${SEEDS:-"1 101 201 301 401"}   # the batch runs 5 seeds
SRC_EPS=${SRC_EPS:-2000}
DEMOS=release/data/mimicgen/tpa_d0_target_expert_n50
SRC=release/data/mimicgen/tpa_d0_source_diverse_v1/dataset.hdf5
BASECFG=release/cfgs/mimicgen/tpa_d0_state_ibrl_target.yaml

for f in "$SRC" "$DEMOS/dataset.hdf5" "$BASECFG"; do
    [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done

# ------------------------------------------------------- IQL offline warm start
# One warm start shared by every seed, as on Threading/Square/Coffee.
if [ "$STEP" = iql ] || [ "$STEP" = all ]; then
  NPZ=release/data/mimicgen/tpa_source_diverse_5k.npz
  [ -f $NPZ ] || $PY scripts/hdf5_to_rollouts_npz.py \
      --hdf5 $SRC --out $NPZ --num_episodes 5000
  IQLDIR=exps/mimicgen/tpa_iql_offline_source
  [ -f $IQLDIR/iql_qagent.pt ] || $PY scripts/pretrain_iql_qagent.py \
      --rollouts $NPZ --demo_path $DEMOS/dataset.hdf5 --demo_num_data 20 \
      --ibrl_cfg $BASECFG --save_dir $IQLDIR --actor_mode rwr
  # LOAD-BEARING remap: pretrain_iql_qagent saves {"critic":..,"actor":..} while
  # train_rl.py --load_pretrained_agent keeps only ALREADY-matching keys, so the
  # nested dict would silently warm-start NOTHING.
  $PY scripts/iql_ckpt_to_agent.py \
      --iql $IQLDIR/iql_qagent.pt --out $IQLDIR/agent_warmstart.pt
fi

# ------------------------------------------------------------ RLPD prior x seeds
if [ "$STEP" = rlpd ] || [ "$STEP" = all ]; then
  for S in $SEEDS; do
    OUT=release/data/mimicgen/tpa_rlpd_prior_s$S
    [ -f $OUT/dataset.hdf5 ] && { echo "[skip] rlpd seed$S"; continue; }
    echo "[rlpd] seed$S"
    # --task is REQUIRED: the default is Threading's 37-dim packing and TPA is
    # 51-dim (9 prop + 42 object; verified element-wise, see merge_prior_dataset).
    $PY scripts/merge_prior_dataset.py \
      --source $SRC --source_episodes $SRC_EPS \
      --target $DEMOS --target_episodes 20 \
      --out $OUT --task ThreePieceAssembly_D0 --seed $S
  done
fi
echo "PREP DONE $(date)"
