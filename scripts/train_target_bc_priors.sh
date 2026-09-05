#!/bin/bash
# Train the target-domain BC priors that the model-free `ibrl_m001` arms need,
# ON THE WORKSTATION, then ship them with cluster/pack_ibrl_priors.sh.
#
#   bash scripts/train_target_bc_priors.sh                  # all 3 tasks x 10 seeds
#   bash scripts/train_target_bc_priors.sh square coffee    # a subset
#   GPUS=0,1 SEEDS="1 101" bash scripts/train_target_bc_priors.sh threading
#
# Long run -- start it detached (CLAUDE.md rule 11: nohup does NOT survive a tool
# timeout, the child is in the same process group):
#   setsid bash scripts/train_target_bc_priors.sh > bc_priors.log 2>&1 </dev/null & disown
#
# ---------------------------------------------------------------- the protocol
# Every recipe below is job_threading_bc_s160.sh's, unchanged: 60 epochs x 100
# steps, lr 1e-4, batch 256, grad_clip 5.0, 20 demos, fine-tuned from the SOURCE
# BC checkpoint. Keeping Square/Coffee identical to Threading here is deliberate
# -- the three tasks' ibrl arms are meant to be read the same way.
#
# ⚠️ Two known limits of that protocol, inherited not introduced:
#   1. `eval_target_during_training` stays 0, so in-training eval runs in the
#      SOURCE domain and TopkSaver's model0.pt is the source-best checkpoint.
#      That is the conventional "source-best -> target eval" sim2real protocol
#      (train_bc.py:85-91), and it is what the Threading priors did.
#   2. train_bc.py has no init-distribution knob at all, so its eval cannot see
#      Square's rotw1 sector or Coffee's 360 deg nut_z_rot. Since (1) already
#      puts checkpoint selection in the source domain, passing a half-specified
#      target here would be worse than not passing one.
# Neither affects TRAINING -- the demos are target demos. They affect which
# checkpoint gets saved. If the priors come out weak, this is the first thing to
# revisit (turn on eval_target_during_training AND add init-dist support).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/Chenyang/.conda/envs/ibrl/bin/python   # set_env.sh does NOT activate conda
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=$PWD:${PYTHONPATH:-}

SEEDS=${SEEDS:-"1 101 201 301 401 501 601 701 801 901"}
GPUS=${GPUS:-0,1,2}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(threading square coffee)

spec() {  # spec <task> <seed> -> OUT|DEMOS|SRC_BC|TASK_NAME
  case $1 in
    threading) echo "exps/mimicgen/il_ft_target_s160_seed$2|release/data/mimicgen/threading_d0_sat_160000_s$2/dataset.hdf5|exps/mimicgen/threading_source_mlp_bc_n2000_seed1/model0.pt|Threading_D0" ;;
    square)    echo "exps/mimicgen/square_il_ft_target_rotw1_seed$2|release/data/mimicgen/square_sat_400000_s$2/dataset.hdf5|exps/mimicgen/square_source_mlp_bc_n1000_seed1/model0.pt|Square_D0" ;;
    coffee)    echo "exps/mimicgen/coffee_il_ft_target_g5x_seed$2|release/data/mimicgen/coffee_g5x_sat400k_s$2/dataset.hdf5|exps/mimicgen/coffee_source_mlp_bc_n1000_seed1/model0.pt|Coffee_D0" ;;
    *) echo "BADTASK"; return 1 ;;
  esac
}

# ------------------------------------------------------------------- preflight
# Fail on the whole list before spending a GPU-hour on the first one.
JOBS=(); missing=0
for t in "${TASKS[@]}"; do
  for s in $SEEDS; do
    S=$(spec "$t" "$s") || { echo "unknown task '$t' (threading|square|coffee)"; exit 1; }
    IFS='|' read -r OUT DEMOS SRC_BC TASK_NAME <<< "$S"
    for f in "$DEMOS" "$SRC_BC"; do
      [ -f "$f" ] || { echo "MISSING $f"; missing=1; }
    done
    if [ -f "$OUT/model0.pt" ]; then
      echo "skip (exists): $OUT/model0.pt"
    else
      JOBS+=("$t $s")
    fi
  done
done
[ "$missing" = 0 ] || { echo "preflight FAILED"; exit 1; }
[ ${#JOBS[@]} -gt 0 ] || { echo "nothing to do -- every prior already exists."; exit 0; }
echo "== ${#JOBS[@]} priors to train on GPUs $GPUS =="

run_one() {  # run_one <gpu> <task> <seed>
  local gpu=$1 t=$2 s=$3
  IFS='|' read -r OUT DEMOS SRC_BC TASK_NAME <<< "$(spec "$t" "$s")"
  echo "[gpu$gpu] $t seed=$s -> $OUT"
  CUDA_VISIBLE_DEVICES=$gpu $PY train_bc.py \
    --save_dir "$OUT" --seed "$s" --task_name "$TASK_NAME" \
    --dataset.path "$DEMOS" --dataset.num_data 20 --dataset.use_state 1 \
    --load_model "$SRC_BC" \
    --state_policy.hidden_dim 1024 --state_policy.dropout 0.0 \
    --num_epoch 60 --epoch_len 100 --lr 1e-4 --batch_size 256 --grad_clip 5.0 \
    --eval_period 50 --num_eval_episode 50 \
    --eval_hand_init_perturb_joint_sigma 0.1 \
    --use_wb 0 > "$OUT.train.log" 2>&1
  if [ -f "$OUT/model0.pt" ]; then echo "[gpu$gpu] OK   $t seed=$s"
  else echo "[gpu$gpu] FAIL $t seed=$s -- see $OUT.train.log"; fi
}

# round-robin the job list over the GPUs, one serial worker each
for i in "${!GPU_ARR[@]}"; do
  (
    for j in "${!JOBS[@]}"; do
      [ $(( j % ${#GPU_ARR[@]} )) -eq "$i" ] || continue
      run_one "${GPU_ARR[$i]}" ${JOBS[$j]}
    done
  ) &
done
wait
echo "== done. missing priors:"
for t in "${TASKS[@]}"; do
  for s in $SEEDS; do
    IFS='|' read -r OUT _ _ _ <<< "$(spec "$t" "$s")"
    [ -f "$OUT/model0.pt" ] || echo "   $OUT/model0.pt"
  done
done
echo "(empty list above = all present; next: bash cluster/pack_ibrl_priors.sh)"
