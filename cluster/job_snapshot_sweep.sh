#!/bin/bash
#SBATCH --job-name=snap_sweep
#SBATCH --array=1-90
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
#   JOBS=cluster/jobs_threading_snapsweep.txt \
#     sbatch --account=def-<pi> --array=1-9 --export=ALL,JOBS cluster/job_snapshot_sweep.sh
#
# ONE array task = ONE wandb run = one (sweep, value, seed). It re-scores every
# snapshot of a finished adapt run (snap_0 .. snap_100000, 11 evals of 100 games)
# at a single eval-time setting and logs each point to wandb as it lands:
#     project optim_threading
#     group   <sweep><value>          e.g. penalty_planning0.01, ens_heads3
#     name    <sweep><value>_seed<N>  e.g. penalty_planning0.01_seed1
#     keys    adapt/transitions (x), score/score  -- same as the adapt runs
#
# JOBS line format:  <sweep> <value> <seed> <run_dir> <stage1_dir>
#   sweep = penalty_planning (dynamics_disagree_coef, all heads)
#         | ens_heads        (--dyn_heads, coef 0)
#
# WHY: both knobs only re-score a fixed checkpoint, so they came out of training
# (a sweep inside every training run doubled its wall clock). Optimistic
# exploration (collect_disagree_coef) changes what is collected and stays as
# training arms (optim_* in job_threading_margin.sh).
#
# Time: 11 x ~72-100 s per 100-game eval + model load ~ 15-25 min -> 1:30:00
# stays in the b1 tier (<=3:00:00). Resumable: done snapshots are kept in
# <run>/sweeps/<name>.tsv and the wandb id is derived from the name.
set -uo pipefail
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
# Real-time logging was asked for, so ONLINE by default here (nibi compute nodes
# have egress). The deadlock that made the training scripts default to offline
# bites long-running processes; this driver is one process for ~20 min and
# finish()es. WANDB_MODE=offline + `wandb sync` still works if needed.
export WANDB_MODE=${WANDB_MODE:-online}

cd $SLURM_SUBMIT_DIR
WANDB_KEY_FILE=${WANDB_KEY_FILE:-$HOME/.config/wandb/key}
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_KEY_FILE" ]; then
    WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
if [ "$WANDB_MODE" = "online" ] && [ -z "$WANDB_API_KEY" ] \
   && ! grep -qs 'api\.wandb\.ai' "$HOME/.netrc"; then
    echo "[wandb] no API key and no ~/.netrc entry -> falling back to WANDB_MODE=offline"
    export WANDB_MODE=offline
fi
# Default to the container when the image is on disk. Requiring SIF to be
# exported was a footgun: a submission that forgot `--export=ALL,SIF` fell
# through to the venv branch and died 119 lines in with
#   /home/<user>/venvs/ibrl/bin/activate: No such file or directory
# -- after the scheduler had already spent the queue wait. Set SIF explicitly to
# override the path, or USE_VENV=1 to force the module/venv route.
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

echo "[gpu] node=$(hostname)  job=${SLURM_JOB_ID:-?}  task=${SLURM_ARRAY_TASK_ID:-?}"

JOBS=${JOBS:?set JOBS=<job list>}
LINE=$(grep -vE '^\s*(#|$)' "$JOBS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$LINE" ]; then
  echo "task $SLURM_ARRAY_TASK_ID past the end of $JOBS -- nothing to do"; exit 0
fi
read -r SWEEP VALUE SEED RUN STAGE1 <<< "$LINE"
echo "[jobs] $JOBS  -> sweep=$SWEEP value=$VALUE seed=$SEED run=$RUN"
for f in "$RUN" "$STAGE1/cfg.yaml"; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done
ls "$RUN"/snap_*.pt >/dev/null 2>&1 || { echo "no snap_*.pt in $RUN"; exit 1; }

$PY scripts/eval_snapshot_run.py --run "$RUN" --stage1 "$STAGE1" \
    --sweep "$SWEEP" --value "$VALUE" --seed "$SEED" \
    --project "${PROJECT:-optim_threading}" \
    --num_game "${NUM_GAME:-100}" --mp "${MP:-10}" \
    --python python
