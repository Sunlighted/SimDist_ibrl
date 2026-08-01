#!/usr/bin/env bash
# Wait for the transition-match BC runs to finish, then evaluate core vs
# transition-matched (RL-collected σ0.02) checkpoints under DETERMINISTIC init
# (hand_init_perturb=0,0,0 — same as core's in-training eval), 500 games each.
# This is the table that settles "does transition-matched σ0.02 catch up to core".
cd /home/Chenyang/ibrl
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh
GPU=${GPU:-3}

echo "[final_eval] waiting for tmatch training (PIDs 994931/994932/994933) ..."
while kill -0 994931 2>/dev/null || kill -0 994932 2>/dev/null || kill -0 994933 2>/dev/null; do
  sleep 60
done
echo "[final_eval] training done; running deterministic 500-game eval"

RUNS=(
  "core20:threading_flow_bc_n20_seed1"
  "core50:threading_flow_bc_n50_seed1"
  "core100:threading_flow_bc_n100_seed1"
  "tmCore20:threading_flow_bc_sig002_tmCore20_n73_seed1"
  "tmCore50:threading_flow_bc_sig002_tmCore50_n182_seed1"
  "tmCore100:threading_flow_bc_sig002_tmCore100_n361_seed1"
)
echo "================ RESULTS (deterministic init, 500 games) ================"
for r in "${RUNS[@]}"; do
  name="${r%%:*}"; dir="${r##*:}"
  ckpt="exps/mimicgen/${dir}/model0.pt"
  if [[ ! -f $ckpt ]]; then echo "${name}: NO CKPT (${ckpt})"; continue; fi
  s=$(CUDA_VISIBLE_DEVICES=${GPU} python evaluate/eval.py --weight "$ckpt" --mode bc \
        --num_game 500 --shift_mode none --hand_init_perturb 0,0,0 --mp 10 2>/dev/null \
      | grep -aE "^score:" | tail -1)
  echo "${name}  (${dir}):  ${s}"
done
echo "========================================================================="
