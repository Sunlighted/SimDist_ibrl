#!/bin/bash
# Chain: wait for thr300k_ibrl_m0_qsrc_seed1 (PID passed in) to exit, then launch
# the SAME arm at constant margin 0.01. That completes the 2x2 whose missing cell
# was flagged: we have (cold,m0) and (qsrc,m0) and historical (cold,m.05/.1)@100k,
# but nothing at (qsrc, m0.01).
#
# Waits on the PID, NOT on `pgrep -f` -- pgrep -f matches this script's own
# command line and the wait would never terminate (CLAUDE.md rule: pgrep self-match).
set -u
cd /home/Chenyang/ibrl
source set_env.sh
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export WANDB_MODE=offline

WAIT_PID=${1:?usage: chain_thr300k_qsrc_m001.sh <pid-to-wait-for>}
S=1
PRED_LOG=logs/thr300k_ibrl_m0_qsrc_seed$S.log

echo "[chain $(date)] waiting for PID $WAIT_PID (thr300k_ibrl_m0_qsrc_seed$S)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[chain $(date)] PID $WAIT_PID exited"
echo "[chain] predecessor last eval: $(grep -E '^[0-9]+: score/score' $PRED_LOG | tail -1 | tr -s ' ')"
if grep -qE "buffer_transitions|other/step +300000" "$PRED_LOG" 2>/dev/null; then
    echo "[chain] predecessor reached its budget cleanly"
else
    echo "[chain] WARNING predecessor has no completion marker -- launching anyway"
fi

DEMOS=release/data/mimicgen/threading_d0_sat_160000_s$S/dataset.hdf5
BC_PRIOR=exps/mimicgen/il_ft_target_s160_seed$S/model0.pt
SRC_EXPERT=exps/mimicgen/threading_ibrl_source_seed1/model0.pt
IBRL_CFG=release/cfgs/mimicgen/threading_ibrl_m001.yaml
for f in "$DEMOS" "$BC_PRIOR" "$SRC_EXPERT" "$IBRL_CFG"; do
    [ -e "$f" ] || { echo "[chain] MISSING $f -- aborting"; exit 1; }
done

echo "[chain $(date)] launching thr300k_ibrl_m001_qsrc_seed$S on GPU1"
CUDA_VISIBLE_DEVICES=1 setsid $PY train_rl.py \
    --config_path $IBRL_CFG --seed $S \
    --save_dir exps/mimicgen/thr300k_ibrl_m001_qsrc_seed$S \
    --bc_policy $BC_PRIOR --preload_datapath $DEMOS --preload_num_data 20 \
    --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
    --prior_margin_init 0.01 --prior_margin_final 0.01 \
    --num_warm_up_episode 0 --count_warmup_transitions 1 \
    --num_train_step 300000 --use_wb 1 --state_only_no_render 1 \
    > logs/thr300k_ibrl_m001_qsrc_seed$S.log 2>&1 < /dev/null &
disown
echo "[chain $(date)] launched, pid $!"
