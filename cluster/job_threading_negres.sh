#!/bin/bash
#SBATCH --job-name=thr_neg
#SBATCH --array=1-20
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
# Submit (same convention as job_sqcof_dynaibrl.sh; --array MUST equal `wc -l < $JOBS`):
#
#   WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres.txt \
#     sbatch --account=rrg-rhinehar --array=1-20 --export=ALL,JOBS cluster/job_threading_negres.sh
#   # once the N=500 demo sets are synced (collected LOCALLY, scripts/build_negres_n500.sh):
#   WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres_demo500.txt \
#     sbatch --account=rrg-rhinehar --array=1-5 --export=ALL,JOBS cluster/job_threading_negres.sh
#   # value ensemble (existing stage-1, see the vens note below):
#   WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres_vens.txt \
#     sbatch --account=rrg-rhinehar --array=1-5 --export=ALL,JOBS cluster/job_threading_negres.sh
#
# Threading_D0 "negative results" batch: five early ideas, each a change of ONE
# component on top of a FIXED plain_h5 baseline, 5 seeds (1 101 201 301 401),
# matched 100k target transitions. Everything lands in its own wandb project,
# exps/negres_threading (save_dir parent = project, run name = group).
#
#   plain_h5     the baseline: plain SimDist MPPI, single dynamics head, h=5,
#                threading_simdist_diversev2_h5_seed1. Flags identical to the
#                plain_h5 / valabl_dyn arms of job_threading_margin.sh EXCEPT the
#                budget: that plain_h5 is a 300k run, and it writes to
#                exps/simdist_threading_3arm/plain_h5_seed*, which is why this batch
#                uses its own EXP_DIR -- same name, different run, never mix them.
#                valabl_dyn (exps/valabl_threading) IS this exact config at 100k:
#                use it as a free replication check of our baseline, not as the
#                baseline (different code revision / submission).
#   vaml         + --dyn_loss vaml : stage-3 dynamics loss ||z'-sg E(s')||^2
#                -> ||V(f(z,a)) - sg V(E(s'))||^2 (agent.update_dynamics_vaml).
#   vcalib       + --value_calib 1 : each round, after the UNCHANGED dynamics
#                update, regress V on the imagined terminal latent f^H(E(s_0),a)
#                to V_stage1(E(s_H^real)) (agent.update_value_calib).
#   selfdistill  + --selfdistill 1 : each round BC the policy head on the planner's
#                own successful target rollouts (half of each batch from the stage-2
#                demos; starts after 5k transitions; 500 steps/round).
#   demo500      stage-2 BC on N=500 demos with bc_steps
#                8000 instead of 1000. TWO fields by design: at 1000 steps 500 demos
#                get ~6 epochs vs ~157 for 20 => an underfit proposal, and the arm
#                would not test "a stronger proposal".
#   vens         MRS.Q value ensemble (10 heads, bagging 0.5, random priors beta=3,
#                min over heads). Heads are built at stage-1, so this arm needs its
#                own stage-1 = VENS_STAGE1.
#                Uses the EXISTING threading_simdist_vens_diversev2_h5_seed1.
#                ⚠️ It differs from the plain stage-1 in TWO ways, not one: besides
#                the ensemble it distilled the IBRL-HYBRID source value
#                (source_value_ibrl=1) while the plain stage-1 predates that flag and
#                distilled the bare-RL-actor value. Accepted as-is (no re-distill);
#                state it when reporting, and read vens also on planner net gain
#                (score - policy_only), since its encoder/policy head differ too.
#
# Pairing: every arm of seed S reads the SAME demo draw sat_160000_s<S>. demo500
# reads sat_160000_n500_s<S> = those same 25 demos (demo_0..24, so its first 20
# ARE the other arms' 20) + 475 new ones from the same collector on a disjoint
# env-seed block (scripts/build_negres_n500.sh), so
# within-seed deltas are paired; across seeds the draw changes, so the error bar
# contains the single-draw sigma (CLAUDE.md iron law #25).
# Readout: 75-100k window mean per seed (eval every 5 rounds x 100 games), then
# mean +- sd over seeds and the paired delta vs plain_h5; report score/policy_only too.
set -e
cd $SLURM_SUBMIT_DIR
mkdir -p cluster/logs

# MIG-partitioned H100s cannot initialise an EGL device display -- `import mujoco`
# with MUJOCO_GL=egl dies at IMPORT time. osmesa lets the import succeed, and with
# --state_only_no_render 1 no GL context is ever created.
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_API_KEY="${WANDB_API_KEY:-}"
USE_WB=${USE_WB:-1}

# Default to the container when the image is on disk. Requiring SIF to be
# exported was a footgun: a submission that forgot `--export=ALL,SIF` fell
# through to the venv branch and died on
#   /home/<user>/venvs/ibrl/bin/activate: No such file or directory
# after the scheduler had already spent the queue wait.
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
    # Every --env is load-bearing on fir; see job_square_baselines.sh for the
    # diagnosis of each (glibc/libGL ordering, tensorboard/protobuf, CA bundle).
    # ⚠️ Coffee NEEDS the MG_BIND: mimicgen installs non-editable in the image and
    # ships no XML assets, so Coffee_D0 dies on coffee_pod.xml. Square escapes it
    # because NutAssembly's assets come from robosuite itself.
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
    [ -f "$VENV/bin/activate" ] || {
        echo "No venv at $VENV/bin/activate and no container image."
        echo "Either build cluster/ibrl.sif (cluster/build_sif.sh), pass SIF=<path>,"
        echo "or create the venv with cluster/cc_setup.sh on a LOGIN node."
        exit 1
    }
    source "$VENV/bin/activate"
    export PYTHONPATH=$PWD:$PYTHONPATH
    PY="python"
fi
echo "[env] $PY"

JOBS=${JOBS:-cluster/jobs_threading_negres.txt}
[ -f "$JOBS" ] || { echo "no job list at JOBS=$JOBS"; exit 1; }
JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
[ -n "$JOB" ] || { echo "task $SLURM_ARRAY_TASK_ID is past the end of $JOBS \
($(wc -l < "$JOBS") lines) -- --array is wider than the list"; exit 1; }
ARM=${JOB% *}; SEED=${JOB##* }
echo "[jobs] $JOBS"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"
# every arm must run on the same code: record it
echo "[code] $(git rev-parse --short HEAD 2>/dev/null) | $(md5sum train_simdist_adapt.py rl/simdist_agent.py | awk '{print $1}' | tr '\n' ' ')"

EXP_DIR=${EXP_DIR:-exps/negres_threading}
OUT=$EXP_DIR/${ARM}${SUFFIX:-}_seed${SEED}

STAGE1_PLAIN=exps/mimicgen/threading_simdist_diversev2_h5_seed1/latest.pt
VENS_STAGE1=${VENS_STAGE1:-exps/mimicgen/threading_simdist_vens_diversev2_h5_seed1/latest.pt}
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5
NDEMO=20
STAGE1=$STAGE1_PLAIN
EXTRA=""
case $ARM in
  plain_h5)    ;;
  vaml)        EXTRA="--dyn_loss vaml" ;;
  vcalib)      EXTRA="--value_calib 1 --value_calib_steps 1000" ;;
  selfdistill) EXTRA="--selfdistill 1 --selfdistill_steps 500 --selfdistill_warmup_transitions 5000 --selfdistill_demo_frac 0.5" ;;
  demo500)     DEMOS=release/data/mimicgen/threading_d0_sat_160000_n500_s${SEED}/dataset.hdf5
               NDEMO=500
               EXTRA="--bc_steps 8000" ;;
  vens)        STAGE1=$VENS_STAGE1 ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac

for f in "$DEMOS" "$STAGE1" "$(dirname "$STAGE1")/cfg.yaml"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_threading_negres.sh)"; exit 1; }
done
grep -q "value_calib" train_simdist_adapt.py \
    || { echo "train_simdist_adapt.py has no value_calib: sync the code first"; exit 1; }
if [ "$ARM" = vens ] && grep -q "^source_value_ibrl: 1" "$(dirname "$STAGE1")/cfg.yaml"; then
    echo "[note] vens stage-1 distilled the IBRL-HYBRID value; plain distilled the bare-actor value"
fi

# Every flag except STAGE1 / DEMOS / NDEMO / EXTRA is shared by all arms.
# collect_disagree_coef 0: geometric disagreement needs >1 dynamics head.
set +e
$PY train_simdist_adapt.py --stage1_ckpt $STAGE1 --save_dir $OUT --seed $SEED \
  --target_demo_path $DEMOS --target_num_data $NDEMO \
  --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
  --num_eval_episode 100 --num_eval_proc 10 \
  --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
  --collect_disagree_coef 0.0 --original_mppi 1 \
  --use_wb $USE_WB --state_only_no_render 1 $EXTRA
RC=$?
set -e
echo "[$(date)] task $SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED rc=$RC"
grep -q "^100000: score/score" $OUT/train.log 2>/dev/null \
  || { echo "INCOMPLETE: $OUT never logged step 100000"; exit 1; }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
