#!/bin/bash
#SBATCH --job-name=tpa_sd1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x_%j.out
# ^ --account=def-<pi> at submit time. Prefer the wrapper, which preflights:
#       ACCOUNT=def-<pi> bash cluster/submit_tpa_simdist.sh
#
# TPA_D0 (ThreePieceAssembly_D0) SimDist STAGE-1 offline distillation.
#   data  release/data/mimicgen/tpa_d0_source_diverse_v1/dataset.hdf5
#         99,750 traj / 40,865,052 transitions / 70.4% success / 9.6 GB
#   cfg   release/cfgs/mimicgen/tpa_d0_state_simdist.yaml
#   out   exps/mimicgen/tpa_simdist_diversev1_h5_seed1
#
# The env-setup block below is COPIED from job_threading_dyna_plain.sh /
# job_square_baselines.sh. Do not "simplify" it: every module load and every
# --env is load-bearing and was added to fix a specific failure on fir (see
# job_square_baselines.sh for the per-flag diagnosis). An earlier version of
# this file called apptainer without `module load apptainer` and died with
# "apptainer: command not found".
#
# ---------------------------------------------------------------- --mem=48G
# NOT the 24G the other jobs use. Measured on the workstation run: VmHWM =
# 13.2 GB for the training process alone (the replay is 40.9M transitions held
# on CPU), plus mp_eval forks workers that inherit it copy-on-write. 24G leaves
# no margin. If 48G is hard to schedule, turn `--mp_eval 0` (slower eval, one
# process), NOT --mem.
#
# ------------------------------------------------- --time=12h and the CHAIN
# Stage-1 is 1M steps at ~10.8 it/s plus 100 eval points ~= 25 h, which does not
# fit one 12 h allocation. `--resume auto` handles it: resume.pt (weights + Adam
# moments + global_step + all four RNG streams) is rewritten atomically at every
# eval point, so hitting the wall costs at most one eval interval. This script
# RESUBMITS ITSELF until the last logged eval step reaches distill_steps, up to
# CHAIN_MAX links.
#
# Restarting from latest.pt would NOT be a resume: TopkSaver stores weights
# only, so Adam's moments would restart at zero and global_step would rewind to
# 0 (replaying the lr/stddev schedules). resume.pt is 84 MB vs latest.pt's
# 29 MB -- that difference IS the optimizer state.
#
# ------------------------------------------------------------- MIG + MuJoCo
# On MIG-partitioned GPUs `import mujoco` with MUJOCO_GL=egl dies at IMPORT time
# ("your EGL driver does not support the PLATFORM_DEVICE extension"), so this
# job forces MUJOCO_GL=osmesa and passes --state_only_no_render 1 (training and
# eval are both state-based; no GL context is ever needed).
#
# ---------------------------------------------------------------------- wandb
# WANDB_MODE=offline by default: online wandb deadlocks long runs (threads park
# in __skb_wait_for_more_packets; the training loop freezes while the heartbeat
# keeps ticking). Sync from a LOGIN node afterwards:
#     wandb sync $WANDB_DIR/offline-run-*
# Every metric is also in train.log -> scripts/wandb_import_logs.py.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CFG=release/cfgs/mimicgen/tpa_d0_state_simdist.yaml
SAVE=exps/mimicgen/tpa_simdist_diversev1_h5_seed1
DATA=release/data/mimicgen/tpa_d0_source_diverse_v1/dataset.hdf5
EXPERT=exps/mimicgen/tpa_source_expert_frozen/model0.pt
PRIOR=exps/mimicgen/tpa_source_rnn_bc_tp1_n1000_seed1/model0.pt
CHAIN=${CHAIN:-1}
CHAIN_MAX=${CHAIN_MAX:-5}

export MUJOCO_GL=osmesa
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$PWD/wandb}
mkdir -p "$WANDB_DIR" cluster/logs

echo "[job] link $CHAIN/$CHAIN_MAX  jobid=${SLURM_JOB_ID:-none}  $(date)"

# ------------------------------------------------------------- asset preflight
# Fail in seconds, not after several minutes of loading a 9.6 GB replay. The
# prior is the one that is easy to forget: tpa_source_expert_frozen/cfg.yaml
# names it and load_frozen_agent loads it while rebuilding the expert.
for f in "$CFG" "$DATA" "$EXPERT" "$PRIOR"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_tpa_simdist.sh)"; exit 1; }
done

# ------------------------------------------------------------------ environment
# Default to the container when the image is on disk (convention from cc0c665:
# requiring SIF to be exported was a footgun -- a submission that forgot
# `--export=ALL,SIF` fell through to the venv branch and died after serving its
# queue wait). Set SIF explicitly to override the path, USE_VENV=1 to force the
# module/venv route.
if [ "${USE_VENV:-0}" != "1" ] && [ -z "${SIF:-}" ] && [ -f cluster/ibrl.sif ]; then
    SIF=cluster/ibrl.sif
    echo "[env] found cluster/ibrl.sif -- using the container (USE_VENV=1 to override)"
fi
if [ "${USE_VENV:-0}" != "1" ] && [ -n "${SIF:-}" ] && [ ! -f "$SIF" ]; then
    echo "SIF=$SIF does not exist. Build it with cluster/build_sif.sh, or set"
    echo "USE_VENV=1 to use the module/venv route instead."
    exit 1
fi

if [ -n "${SIF:-}" ] && [ "${USE_VENV:-0}" != "1" ]; then
    module purge; module load apptainer
    R=$(realpath "$PWD")
    MG_ASSETS=$R/deps/mimicgen/mimicgen/models/robosuite/assets
    MG_DEST=/usr/local/lib/python3.10/dist-packages/mimicgen/models/robosuite/assets
    MG_BIND=""
    [ -d "$MG_ASSETS" ] && MG_BIND="-B $MG_ASSETS:$MG_DEST"
    PY="apptainer exec --nv -B $R:$R -B ${SLURM_TMPDIR:-/tmp}:/tmp $MG_BIND --pwd $R \
        --env PYTHONPATH=$R/.pylibs:$R --env MUJOCO_GL=$MUJOCO_GL --env WANDB_MODE=$WANDB_MODE \
        --env WANDB_DIR=$WANDB_DIR \
        --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        $SIF python"
else
    module purge; module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
    VENV=${VENV:-$HOME/venvs/ibrl}
    # Fail here, with the reason, rather than letting `source` of a missing file
    # print one line and carry on into a python that does not exist.
    [ -f "$VENV/bin/activate" ] || {
        echo "No venv at $VENV/bin/activate and no container image."
        echo "Either build cluster/ibrl.sif (cluster/build_sif.sh), pass SIF=<path>,"
        echo "or point VENV= at a real virtualenv."
        exit 1
    }
    source "$VENV/bin/activate"
    export PYTHONPATH=$PWD:$PYTHONPATH
    PY="python"
fi
echo "[env] $PY"

# --resume auto is safe on the FIRST link too: it prints
# "resume.pt not found -- starting from scratch".
$PY train_simdist.py --config_path "$CFG" --save_dir "$SAVE" \
    --resume auto --state_only_no_render 1
RC=$?
echo "[job] training exited rc=$RC"

# ------------------------------------------------------------------- the chain
# Progress is read from train.log with grep, NOT python: at this point the venv
# may already be deactivated by a module purge in a resubmitted link, and the
# container route's $PY cannot take a heredoc on stdin.
TOTAL=$(grep -E '^distill_steps:' "$CFG" | head -1 | awk '{print $2}')
LAST=$(grep -ao 'eval @ distill step [0-9]*' "$SAVE/train.log" 2>/dev/null \
       | tail -1 | grep -o '[0-9]*$')
LAST=${LAST:-0}
echo "[job] progress: $LAST / $TOTAL"

if [ "$LAST" -ge "${TOTAL:-0}" ] 2>/dev/null; then
    echo "[job] stage-1 complete."
elif [ "$CHAIN" -ge "$CHAIN_MAX" ]; then
    echo "[job] CHAIN_MAX=$CHAIN_MAX reached without finishing -- resubmit by hand."
elif [ "$RC" -ne 0 ] && [ "$LAST" = "0" ]; then
    # Died before the first eval point: a real error, not a walltime hit.
    # Chaining would just reproduce it CHAIN_MAX times.
    echo "[job] failed before any eval point (rc=$RC) -- NOT chaining. Read the log above."
else
    echo "[job] not finished; submitting link $((CHAIN+1))"
    sbatch --account="${SLURM_JOB_ACCOUNT:?}" \
           --export=ALL,CHAIN=$((CHAIN+1)),CHAIN_MAX="$CHAIN_MAX"${SIF:+,SIF="$SIF"}${USE_VENV:+,USE_VENV="$USE_VENV"} \
           cluster/job_tpa_simdist.sh
fi
