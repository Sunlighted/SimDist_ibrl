#!/bin/bash
# Build every LOCAL asset the Square model-free baselines need, so the cluster
# only ever sees small files.
#
# Run this on the workstation (it needs the ibrl env, a GPU, and the 5.2 GB
# square_d0_source_rotw1_diverse). Afterwards `cluster/sync_to_cluster.sh`
# ships ~300 MB instead of 5.5 GB -- the two things that actually consume the
# diverse set (the IQL offline pretrain and the RLPD merge) both happen here.
#
# Produces:
#   exps/mimicgen/square_iql_offline_source/agent_warmstart.pt    IQL warm start (shared)
#   release/data/mimicgen/square_rlpd_prior_s<N>/                 RLPD prior set (x10)
#
# IBRL's prior needs NO build step: it is the EXISTING source MLP BC,
# exps/mimicgen/square_source_mlp_bc_n1000_seed1/model0.pt, used unmodified.
# Threading's recipe (fine-tune that BC on the 20 target demos for 60 epochs)
# was measured and DESTROYS the policy on Square. On the rotw1+target+sigma0.1
# protocol at n=100:
#     source MLP BC, untouched      0.26   (0.22 on a re-check)
#     source RNN-BC,  untouched     0.17
#     fine-tuned  3/6/12/22/40/60 epochs   0.00  (all six)
#     from scratch 12/30/60/150/400 epochs 0.00  (all five)
# It is not a harness artifact (source weights re-probed under the fine-tune cfg
# still gave 0.22) and not a data-pipeline bug (from-scratch BC loss fell
# 0.223 -> 0.031, i.e. it fit the demos, and SimDist stage-2 reaches 0.53 on the
# SAME 20 demos). 20 demos is simply not enough for a raw-state MLP BC on Square,
# and the source BC at 0.26 has no margin to absorb being pulled toward them.
# Why SimDist survives it: stage-2 trains only a small head on a FROZEN encoder
# distilled from 200k diverse episodes (docs/simdist.md 24), so the
# representation cannot be destroyed.
#
# Usage:  bash scripts/prep_square_baselines.sh [step]
#         step = iql | rlpd | all   (default all)
set -e
cd "$(dirname "$0")/.."
# NOTE set_env.sh only exports PYTHONPATH/LD_LIBRARY_PATH -- it does NOT activate
# conda, so the PATH export below is unconditional (a `||` fallback never fires).
source set_env.sh >/dev/null 2>&1 || true
export PATH=${IBRL_PY:-/home/Chenyang/.conda/envs/ibrl/bin}:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=1

STEP=${1:-all}
SEEDS="1 101 201 301 401 501 601 701 801 901"
DEMOS=release/data/mimicgen/square_sat_600000_s   # the 600k collector; see docs/tasks.md 41
SRC=release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5
BASECFG=release/cfgs/mimicgen/square_d0_state_baseline_target_rotw1.yaml

# ------------------------------------------------------- IQL offline warm start
# One warm start shared by all seeds, exactly as Threading did (all five
# iql_seed* point at the same exps/mimicgen/iql_offline_source/agent_warmstart.pt).
if [ "$STEP" = iql ] || [ "$STEP" = all ]; then
  NPZ=release/data/mimicgen/square_source_rotw1_diverse_5k.npz
  [ -f $NPZ ] || python scripts/hdf5_to_rollouts_npz.py \
      --hdf5 $SRC --out $NPZ --num_episodes 5000
  IQLDIR=exps/mimicgen/square_iql_offline_source
  [ -f $IQLDIR/iql_qagent.pt ] || python scripts/pretrain_iql_qagent.py \
      --rollouts $NPZ --demo_path ${DEMOS}1/dataset.hdf5 --demo_num_data 20 \
      --ibrl_cfg $BASECFG --save_dir $IQLDIR --actor_mode rwr
  # LOAD-BEARING remap: pretrain_iql_qagent saves {"critic":..,"actor":..} while
  # train_rl.py --load_pretrained_agent keeps only ALREADY-matching keys, so the
  # nested dict silently warm-starts NOTHING. See scripts/iql_ckpt_to_agent.py.
  python scripts/iql_ckpt_to_agent.py \
      --iql $IQLDIR/iql_qagent.pt --out $IQLDIR/agent_warmstart.pt
fi

# ------------------------------------------------------------ RLPD prior x10
if [ "$STEP" = rlpd ] || [ "$STEP" = all ]; then
  for S in $SEEDS; do
    OUT=release/data/mimicgen/square_rlpd_prior_s$S
    [ -f $OUT/dataset.hdf5 ] && { echo "[skip] rlpd seed$S"; continue; }
    echo "[rlpd] seed$S"
    # --task Square_D0 is REQUIRED: the default is Threading's 37-dim packing and
    # square obs is 23-dim (the assert in merge_prior_dataset catches it, but
    # only after reading the whole source file).
    python scripts/merge_prior_dataset.py \
      --source $SRC --source_episodes 2000 \
      --target ${DEMOS}$S --target_episodes 20 \
      --out $OUT --task Square_D0 --seed $S
  done
fi
echo "PREP DONE $(date)"
