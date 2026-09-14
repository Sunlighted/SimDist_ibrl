#!/bin/bash
#SBATCH --job-name=snap_sweep
#SBATCH --array=1-11
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
#   sbatch --account=def-<pi> --export=ALL,RUN=<dir>,STAGE1=<dir> \
#          cluster/job_snapshot_sweep.sh
#
# Re-score one adapt run's periodic snapshots over the EVAL-TIME grid: ensemble
# size x pessimism coef. One array task per snapshot.
#
# WHY THIS EXISTS. Both knobs only re-score a fixed checkpoint, so sweeping them
# inside training multiplies the wall clock of EVERY training run: a 100-episode
# MPPI eval is 221-282 s and a 100k run has ~20 eval points, i.e. ~1.5 h per
# extra coef or head size. Training one run with --snapshot_every_rounds 10 and
# sweeping here costs the grid ONCE instead of once per arm. collect_disagree_coef
# (optimistic exploration) is NOT an eval-time knob -- it changes what gets
# collected -- so it stays as one training arm per value (optim_* in
# job_threading_margin.sh).
#
# ⚠️ evaluate/eval.py cannot read an adapt save_dir: train_simdist's load_model
# reads cfg.yaml next to the weight, and an adapt dir's cfg.yaml is an
# AdaptConfig dump whose `target_num_data: null` breaks MainConfig parsing. The
# shim dir below pairs the STAGE-1 cfg.yaml with the snapshot weights -- which is
# also the correct env, since the target is defined by the stage-1 cfg (rule 19b).
#
# ⚠️ HEADS can only truncate DOWN, so the run must come from a 10-head stage-1
# (threading_simdist_dynens10_diversev2_h5_seed1 is the only one on any task).
set -uo pipefail

RUN=${RUN:?set RUN=<adapt save_dir holding snap_*.pt>}
STAGE1=${STAGE1:?set STAGE1=<stage-1 dir holding cfg.yaml>}
HEADS=${HEADS:-1,3,5,10}
COEFS=${COEFS:-0,0.01,0.05,0.1,0.2}
NUM_GAME=${NUM_GAME:-100}
MP=${MP:-10}
SIGMA=${SIGMA:-0.1}
STRICT=${STRICT:-1}
PSET=${PSET:-}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=${WANDB_MODE:-offline}

cd $SLURM_SUBMIT_DIR
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

# ---- pick this task's snapshot -------------------------------------------
# sort -V, not `sort -t_ -k2 -n`: the cluster path has underscores of its own
# (SimDist_ibrl, optim_local, ...), so field 2 is not the transition count.
mapfile -t SNAPS < <(ls "$RUN"/snap_*.pt 2>/dev/null | sort -V)
if [ "${#SNAPS[@]}" -eq 0 ]; then
  echo "no snap_*.pt in $RUN -- was it trained with --snapshot_every_rounds ?"
  exit 1
fi
IDX=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
if [ "$IDX" -ge "${#SNAPS[@]}" ]; then
  echo "task $SLURM_ARRAY_TASK_ID past the end (${#SNAPS[@]} snapshots) -- nothing to do"
  exit 0
fi
SNAP=${SNAPS[$IDX]}
TR=$(basename "$SNAP" .pt); TR=${TR#snap_}
echo "[sweep] snapshot $SNAP  (transitions=$TR)  of ${#SNAPS[@]}"

# ---- shim dir: stage-1 cfg + this snapshot -------------------------------
SHIM=$RUN/_eval_shim
mkdir -p "$SHIM"
[ -e "$SHIM/cfg.yaml" ] || ln -s "$(realpath "$STAGE1/cfg.yaml")" "$SHIM/cfg.yaml"
[ -e "$SHIM/$(basename "$SNAP")" ] || ln -s "$(realpath "$SNAP")" "$SHIM/"
W=$SHIM/$(basename "$SNAP")

OUT=$RUN/sweep_${TR}.tsv
: > "$OUT"
echo -e "transitions\theads\tcoef\tsuccess\tnum_game" >> "$OUT"

IFS=',' read -r -a HS <<< "$HEADS"
IFS=',' read -r -a CS <<< "$COEFS"
for h in "${HS[@]}"; do
  for c in "${CS[@]}"; do
    # --planner_set is applied AFTER --original_mppi, so the coef survives the
    # recipe reset that --original_mppi performs.
    P="dynamics_disagree_coef=$c"
    [ -n "$PSET" ] && P="$P,$PSET"
    LOG=$RUN/eval_${TR}_h${h}_c${c}.log
    $PY evaluate/eval.py --weight "$W" --mode simdist \
        --num_game "$NUM_GAME" --mp "$MP" \
        --shift_from_cfg 1 --strict_success "$STRICT" \
        --hand_init_perturb_joint_sigma "$SIGMA" \
        --original_mppi 1 --planner_set "$P" --dyn_heads "$h" \
        > "$LOG" 2>&1
    SR=$(grep -oE "success rate: [0-9.]+%" "$LOG" | tail -1 | grep -oE "[0-9.]+")
    if [ -z "$SR" ]; then
      echo "  FAIL heads=$h coef=$c -- see $LOG"
      tail -3 "$LOG"
    else
      echo -e "$TR\t$h\t$c\t$SR\t$NUM_GAME" >> "$OUT"
      echo "  ok   heads=$h coef=$c  success=${SR}%"
    fi
  done
done
echo "[sweep] wrote $OUT"
