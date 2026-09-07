#!/bin/bash
# Threading 300k long-run: margin-crossover experiment (4 arms, seed 1).
#
#   GPU1  thr300k_ibrl_m0_qsrc     model-free IBRL, margin 0, SOURCE-EXPERT CRITIC warm-start
#   GPU1  thr300k_dyna_m0demo      Dyna-IBRL,       margin 0, demos preloaded into the TD replay
#   GPU0  thr300k_dyna_m001        Dyna-IBRL,       margin 0.01, demos preloaded
#   GPU0  thr300k_ibrl_m0_cold     model-free IBRL, margin 0, COLD critic (no warm-start)
#
# Everything is on the sat_160000 demo set (both sides), 20 demos, seed 1.
# NO env warm-up on the IBRL side: --num_warm_up_episode 0 makes
# train_rl.py:757 see replay.num_episode(20) >= num_warm_up_episode(0+20) and
# skip warm_up() entirely -- zero unbilled target rollouts, matching Dyna, whose
# max_transitions charges for every collected step.
#
# save_dir MUST be exps/mimicgen/<name>: run_cfg.py:12 derives the wandb project
# from save_dir.split("/")[-2], and the group from the name minus the seed token.
set -e
cd /home/Chenyang/ibrl
source set_env.sh
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export WANDB_MODE=offline          # online wandb deadlocks long runs (skb_wait_for_more_packets)
mkdir -p logs

S=1
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s$S/dataset.hdf5
BC_PRIOR=exps/mimicgen/il_ft_target_s160_seed$S/model0.pt        # best slot for THIS run (train.log: "retuning the best model .../model0.pt")
SRC_EXPERT=exps/mimicgen/threading_ibrl_source_seed1/model0.pt
STAGE1=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
IBRL_CFG=release/cfgs/mimicgen/threading_ibrl_m001.yaml

for f in "$DEMOS" "$BC_PRIOR" "$SRC_EXPERT" "$STAGE1" "$IBRL_CFG"; do
    [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

ibrl_common=(--config_path $IBRL_CFG --seed $S
             --bc_policy $BC_PRIOR --preload_datapath $DEMOS --preload_num_data 20
             --prior_margin_init 0.0 --prior_margin_final 0.0
             --num_warm_up_episode 0 --count_warmup_transitions 1
             --num_train_step 300000 --use_wb 1 --state_only_no_render 1)

dyna_common=(--stage1_ckpt $STAGE1 --seed $S
             --target_demo_path $DEMOS --target_num_data 20
             --max_transitions 300000 --adapt_interval 1000 --eval_every_rounds 5
             --num_eval_episode 100 --num_eval_proc 10
             --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1
             --dyna.preload_demos 1 --dyna.prior_margin_steps 50000
             --use_wb 1 --state_only_no_render 1)

launch () {  # launch <gpu> <logname> <cmd...>
    local gpu=$1 name=$2; shift 2
    echo "[launch] GPU$gpu  $name"
    CUDA_VISIBLE_DEVICES=$gpu setsid "$@" > logs/$name.log 2>&1 < /dev/null &
    disown
    sleep 20   # stagger MuJoCo/CUDA context creation
}

# ---- GPU 1 ----------------------------------------------------------------
# arm1: --load_policy_only 0 --load_critic_only 1 is load-bearing. The yaml ships
# load_policy_only: 1, which would warm-start the ACTOR instead of the critic.
launch 1 thr300k_ibrl_m0_qsrc_seed$S $PY train_rl.py "${ibrl_common[@]}" \
    --save_dir exps/mimicgen/thr300k_ibrl_m0_qsrc_seed$S \
    --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1

launch 1 thr300k_dyna_m0demo_seed$S $PY train_simdist_dyna_ibrl.py "${dyna_common[@]}" \
    --save_dir exps/mimicgen/thr300k_dyna_m0demo_seed$S \
    --dyna.prior_margin_init 0.0 --dyna.prior_margin_final 0.0

# ---- GPU 0 ----------------------------------------------------------------
launch 0 thr300k_dyna_m001_seed$S $PY train_simdist_dyna_ibrl.py "${dyna_common[@]}" \
    --save_dir exps/mimicgen/thr300k_dyna_m001_seed$S \
    --dyna.prior_margin_init 0.01 --dyna.prior_margin_final 0.01

# arm4: --load_pretrained_agent '' is load-bearing. The yaml points at the source
# expert; without the override this arm silently warm-starts its actor from it and
# stops being a cold start (and differs from arm1 in TWO fields, not one).
launch 0 thr300k_ibrl_m0_cold_seed$S $PY train_rl.py "${ibrl_common[@]}" \
    --save_dir exps/mimicgen/thr300k_ibrl_m0_cold_seed$S \
    --load_pretrained_agent '' --load_policy_only 1 --load_critic_only 0

echo
echo "launched 4 runs. tail logs/thr300k_*.log"
