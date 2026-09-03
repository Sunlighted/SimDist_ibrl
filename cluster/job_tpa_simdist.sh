#!/bin/bash
#SBATCH --job-name=tpa_sd1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x_%j.out
# ^ --account=def-<pi> at submit time:
#       sbatch --account=def-<pi> cluster/job_tpa_simdist.sh
#
# TPA_D0 (ThreePieceAssembly_D0) SimDist STAGE-1 offline distillation.
#   data  release/data/mimicgen/tpa_d0_source_diverse_v1/dataset.hdf5
#         99,750 traj / 40,865,052 transitions / 70.4% success / 10.2 GB
#   cfg   release/cfgs/mimicgen/tpa_d0_state_simdist.yaml
#   out   exps/mimicgen/tpa_simdist_diversev1_h5_seed1
#
# ---------------------------------------------------------------- --mem=48G
# NOT the 24G the other jobs use. Measured on the workstation run: VmHWM =
# 13.2 GB for the training process alone (the replay is 40.9M transitions held
# on CPU), plus mp_eval forks num_eval_proc workers that inherit it copy-on-
# write. 24G leaves no margin; 48G does. If the scheduler makes 48G expensive,
# the knob to turn is `--mp_eval 0` (slower eval, one process), NOT --mem.
#
# ------------------------------------------------- --time=12h and the CHAIN
# Stage-1 is 1M steps at ~10.8 it/s plus 100 eval points ~= 25 h of wall time,
# which does not fit one 12 h allocation. That is what `--resume auto` is for:
# resume.pt (weights + Adam moments + global_step + all four RNG streams) is
# rewritten atomically at every eval point, so a job that hits the wall loses at
# most one eval interval. This script RESUBMITS ITSELF until global_step reaches
# distill_steps, up to CHAIN_MAX links.
#
# Restarting from latest.pt instead would NOT be a resume: TopkSaver stores
# weights only, so Adam's moments would restart at zero and global_step would
# rewind to 0 (the lr/stddev schedules would replay). resume.pt is 84 MB against
# latest.pt's 29 MB -- that difference is exactly the optimizer state.
#
# ------------------------------------------------------------- MIG + MuJoCo
# On MIG-partitioned GPUs `import mujoco` with MUJOCO_GL=egl dies at IMPORT time
# ("your EGL driver does not support the PLATFORM_DEVICE extension"), so this
# job forces MUJOCO_GL=osmesa and passes --state_only_no_render 1 (training and
# eval are both state-based; no GL context is ever needed).
#
# ---------------------------------------------------------------------- wandb
# WANDB_MODE=offline: online wandb deadlocks long runs (threads park in
# __skb_wait_for_more_packets and the training loop freezes while the heartbeat
# keeps ticking). Sync from a LOGIN node afterwards:
#     wandb sync $WANDB_DIR/wandb/offline-run-*
# Every metric is also in train.log, recoverable via scripts/wandb_import_logs.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CFG=release/cfgs/mimicgen/tpa_d0_state_simdist.yaml
SAVE=exps/mimicgen/tpa_simdist_diversev1_h5_seed1
CHAIN=${CHAIN:-1}
CHAIN_MAX=${CHAIN_MAX:-5}

export MUJOCO_GL=osmesa
export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR:-$PWD/wandb}

echo "[job] link $CHAIN/$CHAIN_MAX  jobid=$SLURM_JOB_ID  $(date)"

run() {
  if [ -n "${SIF:-}" ]; then
    apptainer exec --nv -B "$PWD:$PWD" --pwd "$PWD" \
      --env MUJOCO_GL=osmesa,WANDB_MODE=offline "$SIF" python "$@"
  else
    python "$@"
  fi
}

# --resume auto is safe on the FIRST link too: it just prints
# "resume.pt not found -- starting from scratch".
run train_simdist.py --config_path "$CFG" --save_dir "$SAVE" \
    --resume auto --state_only_no_render 1

# ---- chain: resubmit unless stage 1 is finished ----
DONE=$(python - "$SAVE" "$CFG" <<'PY'
import sys, os, torch, yaml
save, cfg_path = sys.argv[1], sys.argv[2]
total = yaml.safe_load(open(cfg_path))["distill_steps"]
p = os.path.join(save, "resume.pt")
if not os.path.exists(p):
    print(0); sys.exit()
s = torch.load(p, map_location="cpu", weights_only=False)
print(1 if int(s.get("global_step", 0)) >= int(total) else 0)
PY
)
if [ "$DONE" = "1" ]; then
  echo "[job] stage-1 complete."
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
  echo "[job] CHAIN_MAX=$CHAIN_MAX reached without finishing -- resubmit by hand."
else
  echo "[job] not finished; submitting link $((CHAIN+1))"
  sbatch --account="${SLURM_JOB_ACCOUNT}" \
         --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX="$CHAIN_MAX"${SIF:+,SIF="$SIF"} \
         cluster/job_tpa_simdist.sh
fi
