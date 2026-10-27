#!/bin/bash
# Planner sweep on the Robotiq exec_obs stage-1 ckpt (2026-10-27): the stage-4 gate run (cfg planner = the ibrl-tuned
# recipe, init_std 0.5, H5) gave MPPI 0.82 < policy head 0.87. Each arm = one independent 300-game strict eval of the
# SAME model0.pt; two at a time on one GPU. Planner horizon > 5 queries an untrained dynamics (wm.horizon 5): not swept.
#   setsid bash scripts/sweep_robotiq_stage1_planner.sh > logs/sweep_robotiq_stage1_planner.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU:-3}" WANDB_MODE=offline
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
S1="${S1:-exps/sim2real/threading_robotiq_pixel_stage1_seed1}"
OUT="$S1/planner_sweep"; mkdir -p "$OUT"
# name | --original_mppi | --planner_set
ARMS=(
  "orig_h5|1|horizon=5"
  "orig_h3|1|horizon=3"
  "orig_h4|1|horizon=4"
  "tuned_std015_h5|0|init_std=0.15"
  "tuned_std015_h3|0|init_std=0.15,horizon=3"
  "tuned_std03_h5|0|init_std=0.3"
  "tuned_std05_h3|0|horizon=3"
  "orig_bstd005_h5|1|base_policy_std=0.05"
)
run() {
  IFS='|' read -r name orig set <<< "$1"
  log="$OUT/$name.log"
  grep -qE '^score: ' "$log" 2>/dev/null && return
  $PY evaluate/eval.py --weight "$S1/model0.pt" --mode simdist --num_game 300 --mp 10 --strict_success 1 \
      --hand_init_perturb_joint_sigma 0.1 --original_mppi "$orig" --planner_set "$set" > "$log" 2>&1
  echo "[sweep $(date -u '+%F %T')] $name: $(grep -E '^score: ' "$log" | tail -1)"
}
for ((i = 0; i < ${#ARMS[@]}; i += 2)); do
  run "${ARMS[$i]}" &
  [ $((i + 1)) -lt ${#ARMS[@]} ] && run "${ARMS[$((i + 1))]}" &
  wait
done
echo "[sweep] done"
for f in "$OUT"/*.log; do printf '%-20s %s | %s\n' "$(basename "$f" .log)" "$(grep -E '^score: ' "$f" | tail -1)" \
  "$(grep -E 'planner\.|ORIGINAL|original' "$f" | tr '\n' ' ' | cut -c1-200)"; done
