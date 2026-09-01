#!/bin/bash
#SBATCH --job-name=thr_bc160
#SBATCH --array=1-10
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
# PREREQUISITE for the ibrl_m005anneal0 arm of job_threading_margin.sh.
#
# Why this exists: model-free IBRL needs a target-domain BC policy as its action
# prior (--bc_policy). The six that already exist, exps/mimicgen/il_ft_target_
# seed{1,101,201,301,401,501}, are unusable for this batch on two counts:
#
#   1. They cover 6 of the 10 seeds.
#   2. They were fine-tuned on sat_400000 demos. Every other arm in the margin
#      batch uses sat_160000, the standing standard since docs/demo-collection.md
#      38 measured +0.139 from that switch alone. Pairing a sat_400000 prior with
#      sat_160000 preload data would confound the IBRL arm with a demo-standard
#      difference larger than most effects being measured.
#
# So: re-run the same fine-tune on sat_160000 for all 10 seeds, into
# il_ft_target_s160_seed*. Everything except the demo path matches
# exps/mimicgen/il_ft_target_seed1/cfg.yaml.
set -e
cd $SLURM_SUBMIT_DIR
mkdir -p cluster/logs

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=${WANDB_MODE:-offline}

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

JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/jobs_threading_bc_s160.txt)
SEED=${JOB##* }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> bc_s160 seed=$SEED"

OUT=exps/mimicgen/il_ft_target_s160_seed${SEED}
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5
# The source-domain BC policy every seed fine-tunes FROM. Single checkpoint, as
# in the original il_ft_target_seed* runs -- it is seed-1 for all of them.
SRC_BC=exps/mimicgen/threading_source_mlp_bc_n2000_seed1/model0.pt

for f in "$DEMOS" "$SRC_BC"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_threading_margin.sh)"; exit 1; }
done

$PY train_bc.py --save_dir $OUT --seed $SEED --task_name Threading_D0 \
  --dataset.path $DEMOS --dataset.num_data 20 --dataset.use_state 1 \
  --load_model $SRC_BC \
  --num_epoch 60 --epoch_len 100 --lr 1e-4 --batch_size 256 --grad_clip 5.0 \
  --eval_period 50 --num_eval_episode 50 --eval_hand_init_perturb_joint_sigma 0.1 \
  --use_wb ${USE_WB:-0}
RC=$?
echo "[$(date)] task $SLURM_ARRAY_TASK_ID seed=$SEED rc=$RC"
[ -f "$OUT/model0.pt" ] || { echo "INCOMPLETE: $OUT/model0.pt was never written"; exit 1; }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
