#!/usr/bin/env bash
# Online TD-MPC2 + IBRL prior on Threading_D0 (CLAUDE.md §28).
#
# The online-model-based-RL counterpart of the SimDist distillation: same latent
# world model + same MPPI, but learned online. DOMAIN picks the arm:
#   DOMAIN=source  -> train + eval in the nominal env      (prior = source flow BC)
#   DOMAIN=target  -> train + eval in the §12 target env   (prior = 20-demo target BC)
#
# Usage:
#   bash scripts/train_tdmpc2_threading.sh                        # source arm
#   DOMAIN=target GPU=1 bash scripts/train_tdmpc2_threading.sh    # target arm
#   IBRL=0 bash scripts/train_tdmpc2_threading.sh                 # vanilla TD-MPC2 ablation
#   MPC=0 bash scripts/train_tdmpc2_threading.sh                  # no planning (pi + IBRL select)
#   PRIOR_MARGIN=0.1 bash scripts/train_tdmpc2_threading.sh       # permanent prior preference
#   ... any extra pyrallis overrides are passed through: "$@"
set -euo pipefail
cd "$(dirname "$0")/.."
source set_env.sh

DOMAIN=${DOMAIN:-source}
SEED=${SEED:-1}
GPU=${GPU:-0}
STEPS=${STEPS:-400000}
IBRL=${IBRL:-1}
MPC=${MPC:-1}
PRIOR_MARGIN=${PRIOR_MARGIN:-0.0}              # 0 = vanilla IBRL (pure greedy max-Q)
PRIOR_MARGIN_INIT=${PRIOR_MARGIN_INIT:--1.0}   # <0 = anneal OFF (vanilla IBRL default)
PRIOR_MARGIN_STEPS=${PRIOR_MARGIN_STEPS:-50000}
MIX_RL_RATE=${MIX_RL_RATE:-0.5}                # demo fraction = 1 - this. 1.0 = no demo mix
NUM_PRIOR_TRAJS=${NUM_PRIOR_TRAJS:-24}

CFG="release/cfgs/mimicgen/threading_d0_state_tdmpc2_${DOMAIN}.yaml"
[ -f "$CFG" ] || { echo "no config for DOMAIN=$DOMAIN ($CFG)"; exit 1; }
SAVE_DIR=${SAVE_DIR:-exps/mimicgen/threading_tdmpc2_${DOMAIN}_seed${SEED}}

echo "config     : $CFG"
echo "save_dir   : $SAVE_DIR"
echo "gpu / seed : $GPU / $SEED"
echo "ibrl=$IBRL  mpc=$MPC  num_prior_trajs=$NUM_PRIOR_TRAJS  mix_rl_rate=$MIX_RL_RATE"
echo "prior_margin=$PRIOR_MARGIN  anneal_init=$PRIOR_MARGIN_INIT (<0 = off, vanilla IBRL)"

CUDA_VISIBLE_DEVICES=$GPU python train_tdmpc2.py \
    --config_path "$CFG" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    --num_train_step "$STEPS" \
    --mix_rl_rate "$MIX_RL_RATE" \
    --tdmpc2.ibrl "$IBRL" \
    --tdmpc2.mpc "$MPC" \
    --tdmpc2.prior_margin "$PRIOR_MARGIN" \
    --tdmpc2.num_prior_trajs "$NUM_PRIOR_TRAJS" \
    --prior_margin_init "$PRIOR_MARGIN_INIT" \
    --prior_margin_steps "$PRIOR_MARGIN_STEPS" \
    "$@"
