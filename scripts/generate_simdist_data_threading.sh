#!/usr/bin/env bash
# 方案 A · Step 2: diverse data generation (SimDist Algorithm 2) using the expert
# + sub-optimal snapshots from Step 1. Output hdf5 is read directly by SimDist's
# 方案 B loader (--source_replay_path). See CLAUDE.md §17.
#
# Usage:
#   bash scripts/generate_simdist_data_threading.sh
#   GPU=1 RUN=exps/mimicgen/threading_ibrl_snapshots_seed1 NUM_TRAJ=2000 \
#     bash scripts/generate_simdist_data_threading.sh
#
# Env vars:
#   GPU       CUDA device id                              (default 0)
#   RUN       snapshots run dir (from Step 1)             (default threading_ibrl_snapshots_seed1)
#   EXPERT    expert ckpt (V_e source)                    (default $RUN/model0.pt)
#   OUT       output dataset dir                          (default threading_d0_source_diverse)
#   NUM_TRAJ  number of trajectories to generate          (default 2000)
#   NUM_PROC  parallel worker processes (each w/ own env)  (default 10)
#   P_EXPERT  prob of expert policy per episode           (default 0.5)
#   SEED      rng seed                                     (default 1)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# GPU memory watchdog (kills the run before the driver crashes). On by default;
# GUARD=0 disables, GUARD_THRESH tunes the used/total danger line.
GUARD="${GUARD:-1}"
GUARD_THRESH="${GUARD_THRESH:-0.92}"

RUN="${RUN:-exps/mimicgen/threading_ibrl_snapshots_seed1_v2}"
EXPERT="${EXPERT:-$RUN/model0.pt}"
OUT="${OUT:-release/data/mimicgen/threading_d0_source_diverse_v2}"
NUM_TRAJ="${NUM_TRAJ:-100000}"
NUM_PROC="${NUM_PROC:-20}"
P_EXPERT="${P_EXPERT:-0.5}"
SEED="${SEED:-1}"
# HYBRID=1 rolls out BOTH the EXPERT and the sub-optimal snapshots as their IBRL
# hybrid (RL actor + BC prior) instead of the bare RL actor — use when the run's
# bare actor leaned on the prior during training.
HYBRID="${HYBRID:-1}"
# SNAP_HYBRID=0 (with HYBRID=1): only the EXPERT rolls out as its IBRL hybrid;
# the sub-optimal snapshots stay bare RL actors (act_method=rl).
SNAP_HYBRID="${SNAP_HYBRID:-1}"
EXPERT_BC_POLICY="${EXPERT_BC_POLICY:-}"
# VALUE_IBRL=1 labels V_e(s) with the HYBRID value max(Q(s,pi_rl), Q(s,prior)) —
# the value of the action the hybrid expert actually executes. Needs HYBRID=1.
VALUE_IBRL="${VALUE_IBRL:-0}"
# VALUE_FULL_ENS=1 removes the RED-Q random-subset noise from the V labels
# (min over all num_q heads instead of a random num_k subset).
VALUE_FULL_ENS="${VALUE_FULL_ENS:-0}"

# collect sub-optimal snapshots from the run dir
CKPTS="$(ls "$RUN"/snapshot_*.pt 2>/dev/null | paste -sd, - || true)"
if [ -z "$CKPTS" ]; then
  echo "[warn] no snapshot_*.pt in $RUN — generation will use the expert only."
fi
echo "expert=$EXPERT"
echo "snapshots=$CKPTS"
echo "out=$OUT  num_traj=$NUM_TRAJ  num_proc=$NUM_PROC  p_expert=$P_EXPERT"

CMD=(python scripts/generate_simdist_data.py
    --expert "$EXPERT"
    --checkpoints "$CKPTS"
    --out_dir "$OUT"
    --num_traj "$NUM_TRAJ"
    --num_proc "$NUM_PROC"
    --p_expert "$P_EXPERT"
    --sigma_min 0.05 --sigma_max 0.2
    --noise_min 1 --noise_max 5
    --nonoise_min 5 --nonoise_max 10
    --max_len 400
    --hand_init_sigma 0.1
    --expert_hybrid "$HYBRID"
    --snapshot_hybrid "$SNAP_HYBRID"
    --expert_bc_policy "$EXPERT_BC_POLICY"
    --value_use_ibrl "$VALUE_IBRL"
    --value_full_ensemble "$VALUE_FULL_ENS"
    --seed "$SEED"
    "$@")

if [ "$GUARD" = "1" ]; then
  echo "[guard] GPU memory watchdog ON (GPU=$GPU, THRESH=$GUARD_THRESH); set GUARD=0 to disable."
  GPU="$GPU" THRESH="$GUARD_THRESH" bash "$REPO_ROOT/scripts/gpu_guard.sh" "${CMD[@]}"
else
  "${CMD[@]}"
fi
