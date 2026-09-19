#!/bin/bash
#SBATCH --job-name=tpa_plan
#SBATCH --array=1-20
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
#
#   JOBS=cluster/jobs_tpa_planning.txt \
#     sbatch --account=def-<pi> --array=1-20 --export=ALL,JOBS cluster/job_tpa_planning.sh
#   WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_tpa_lamA.txt \
#     sbatch --account=rrg-rhinehar --array=1-40 --export=ALL,JOBS cluster/job_tpa_planning.sh
#
# ThreePieceAssembly_D0 (target kp0.25), the PLANNING methods, 100k transitions,
# into their own wandb project exps/tpa_planning (group = arm name).
#
#   tpa_plan_plain_nd20        single-head stage-1, NO anchor (bc_anchor_coef 0)
#   tpa_plan_dynens_anch3_nd20 5-head stage-1, anchor lambda=3
#   tpa_lamA_<v>_nd20          lambda tuning LINE A on TPA: tpa_plan_dynens_anch3
#                              with bc_anchor_coef = v, NOTHING else changed (the
#                              TPA counterpart of Threading's lamA_* and Square/
#                              Coffee's sq_/cof_lamA_*). lambda=3 IS
#                              tpa_plan_dynens_anch3_nd20 -- not re-run. Unlike
#                              Threading, lambda=0 on the 5-head stage-1 does not
#                              exist yet (tpa_plan_plain is the 1-head stage-1),
#                              so it is in the list. -> exps/lamtune_tpa
#   tpa_valabl_{joint,v2d,d2v,marg}_nd20   the value-adapt ablation, all on the
#                              dynens + lambda=3 recipe; cell (i) "dynamics only"
#                              IS tpa_plan_dynens_anch3_nd20, not re-run
#
# Both use the utd200 stage-3 update (--dyn_update_steps 200), the variant that
# removed TPA's early stage-3 collapse 3/3 seeds (memory
# project_tpa_utd200_replication: +0.063 over the whole curve, +0.017 ns at
# 75-100k). Everything else is copied field-for-field from the local reference
# exps/mimicgen/tpa_adapt100k_utd200_seed1: original_mppi 1 + planner_set
# init_std=0.15, collect_disagree_coef 0, dyn_anchor_coef 0, no eval sigma and
# strict off (the stage-1 cfg's own settings -- TPA's target is defined there,
# docs/infra.md 13).
#
# ------------------------------------------------------------ demos, per seed
# The k-th seed (1, 101, ..., 901 -> k = 1..10) reads its OWN collection
# tpa_d0_target_expert_n50_c<k> (10 non-overlapping batches of 50 from the same
# target expert, same shift -- verified env_cfg-for-env_cfg). CLAUDE.md rule 25:
# multi-seed runs on one shared draw have no demo-sampling variance in their
# error bars, which is this project's largest noise source.
# ⚠️ The TPA model-free baselines (job_tpa_6arm.sh) did NOT do this -- all their
# seeds read the first 20 of tpa_d0_target_expert_n50. Their seed-to-seed spread
# is therefore narrower by construction; compare means, and do not read the
# planning arms' wider bands as "less stable". DEMO_SET=n50 reproduces the
# baselines' draw instead.
# ND=20 (default) or ND=50 sets --target_num_data. ⚠️ 50 is not "more data":
# it is ~35 BC epochs instead of ~98 (memory project_tpa_target_demos_c1_c10),
# so its lambda conclusions need a 50-demo baseline, not the 20-demo numbers.
#
# ------------------------------------------------------------ anchor pool
# tpa_anchor_pool_4k: a uniform 4000-episode, state-only extract of the 10.2 GB
# tpa_d0_source_diverse_v1 (which the cluster does not have). The anchor draws
# bc_anchor_episodes=2000 of whatever pool it is given, so a uniform subsample
# has the same state distribution; checked against the full pool at stage-2 on
# the local reference (D_SRC / bc_loss) before shipping.
#
# ------------------------------------------------------------ eval cadence
# Every 10 rounds (10k transitions), 50 episodes, 10 procs -- the same cadence as
# the TPA baselines so the curves share an x grid. The local reference ran 5/100
# and took 9.8 h, ~80% of it eval (a 100-episode TPA eval is ~1450 s; episodes
# are 1000 steps). policy_only is logged alongside.
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs

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

JOBS=${JOBS:-cluster/jobs_tpa_planning.txt}
[ -f "$JOBS" ] || { echo "no job list at $JOBS"; exit 1; }
JOB=$(grep -vE '^\s*(#|$)' "$JOBS" | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$JOB" ] || { echo "task $SLURM_ARRAY_TASK_ID past the end of $JOBS"; exit 0; }
ARM=${JOB% *}; SEED=${JOB##* }
echo "[jobs] $JOBS"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"

ND=${ND:-20}
case $SEED in
  1) K=1;; 101) K=2;; 201) K=3;; 301) K=4;; 401) K=5;;
  501) K=6;; 601) K=7;; 701) K=8;; 801) K=9;; 901) K=10;;
  *) echo "seed $SEED has no demo collection mapped"; exit 1;;
esac
if [ "${DEMO_SET:-per_seed}" = n50 ]; then
  DEMOS=release/data/mimicgen/tpa_d0_target_expert_n50/dataset.hdf5
else
  DEMOS=release/data/mimicgen/tpa_d0_target_expert_n50_c${K}/dataset.hdf5
fi
POOL=release/data/mimicgen/tpa_anchor_pool_4k/dataset.hdf5
STAGE1_PLAIN=exps/mimicgen/tpa_simdist_plain_diversev1_h5_seed1/latest.pt
STAGE1_DYNENS=exps/mimicgen/tpa_simdist_diversev1_h5_seed1/latest.pt

case $ARM in
  tpa_valabl_*) EXP_DIR=${EXP_DIR:-exps/valabl_tpa} ;;
  tpa_lamA_*)   EXP_DIR=${EXP_DIR:-exps/lamtune_tpa} ;;
  *)            EXP_DIR=${EXP_DIR:-exps/tpa_planning} ;;
esac
OUT=$EXP_DIR/${ARM}_seed${SEED}

case $ARM in
  tpa_plan_plain_nd*)        STAGE1=$STAGE1_PLAIN;  LAM=0.0 ;;
  tpa_plan_dynens_anch3_nd*) STAGE1=$STAGE1_DYNENS; LAM=3.0 ;;
  # lambda tuning: the dynens_anch3 recipe with only the anchor multiplier changed
  tpa_lamA_*_nd*)            STAGE1=$STAGE1_DYNENS; LAM=${ARM#tpa_lamA_}; LAM=${LAM%_nd*} ;;
  # value ablation: same stage-1 / anchor as the dynens planning arm, only the
  # heads that move (and their order) change.
  tpa_valabl_*_nd*)          STAGE1=$STAGE1_DYNENS; LAM=3.0 ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac
# the arm name carries the demo count; refuse a mismatch with ND
case $ARM in *_nd$ND) ;; *) echo "arm $ARM does not match ND=$ND"; exit 1 ;; esac

for f in "$STAGE1" "$DEMOS" "$POOL" "$(dirname "$STAGE1")/cfg.yaml"; do
  [ -f "$f" ] || { echo "MISSING asset: $f"; exit 1; }
done
echo "[setup] stage1=$STAGE1 lambda=$LAM demos=$DEMOS nd=$ND"

COMMON="--stage1_ckpt $STAGE1 --seed $SEED \
  --target_demo_path $DEMOS --target_num_data $ND \
  --bc_anchor_coef $LAM --bc_anchor_path $POOL \
  --dyn_update_steps 200 \
  --planner_set init_std=0.15 --original_mppi 1 \
  --collect_disagree_coef 0.0 --dyn_anchor_coef 0.0 \
  --adapt_interval 1000 \
  --eval_every_rounds 10 --num_eval_episode 50 --num_eval_proc 10 \
  --eval_policy_only 1 \
  --use_wb $USE_WB --state_only_no_render 1"

case $ARM in
  tpa_plan_*|tpa_lamA_*)
    $PY train_simdist_adapt.py $COMMON --save_dir "$OUT" --max_transitions 100000 ;;

  # ---- value-adapt ablation (docs/on-policy-adaptation.md 2) ---------------
  #   valabl_dyn     (i)   dynamics only -- identical to tpa_plan_dynens_anch3,
  #                        so it is NOT in the job list; use that arm as cell (i)
  #   valabl_joint   (ii)  value + dynamics together, 100k
  #   valabl_v2d     (iii) value 100k THEN dynamics 100k
  #   valabl_d2v     (iv)  dynamics 100k THEN value 100k
  #   valabl_marg    (v)   marginal (residual) value + dynamics, 100k
  #
  # The sequential arms run BOTH phases in one job (2 x 100k). Phase A writes to
  # $OUT.phaseA and phase B warm-starts from it via --init_ckpt while KEEPING
  # --stage1_ckpt: that path reads dirname/cfg.yaml as a MainConfig, and an adapt
  # run writes an AdaptConfig there (gotcha-eval-adapt-ckpt-cfg-collision).
  #
  # ⚠️ (v) fits its residual by MSE on Monte-Carlo returns while (ii)-(iv) use the
  # value head's two-hot CE. That is inherent: the head is categorical
  # (discrete_heads=1, 101 bins) and a scalar residual has no logit-space
  # counterpart, so it sits on the DECODED scalar -- the quantity MPPI ranks with.
  #
  # ⚠️ Expected negative. on-policy-adaptation 2 measured all three orders
  # collapsing on Threading and findings 33 gives the mechanism (MPPI's entire
  # candidate-discrimination signal is std_i(gamma^H V) = 0.0068 = 1.4% of range).
  # findings 33 also notes the damage shrinks 8.7x on a dynens base, and TPA's
  # stage-3 is dominated by rollout compounding (memory
  # project_tpa_compounding_rho) -- so TPA is where "value adapt is harmless
  # because dynamics error dominates" would show up if it is true anywhere.
  # This is the three-task confirmation, not a search.
  #
  # ⚠️ TIME. TPA is the slow task: the sequential arms are 2 x 100k. The 100k
  # single-phase arms measured ~9.8 h locally at 5/100 eval; at this 10/50 cadence
  # they are ~5-6 h, so v2d/d2v need ~11-12 h -- inside the 14 h wall, but there
  # is no resume, so a preempted job restarts from zero.
  tpa_valabl_joint_nd*)
    $PY train_simdist_adapt.py $COMMON --save_dir "$OUT" \
      --adapt_mode both  --max_transitions 100000 ;;
  tpa_valabl_marg_nd*)
    $PY train_simdist_adapt.py $COMMON --save_dir "$OUT" \
      --adapt_mode both  --marginal_value 1 --max_transitions 100000 ;;
  tpa_valabl_v2d_nd*)
    $PY train_simdist_adapt.py $COMMON --save_dir "$OUT.phaseA" \
      --adapt_mode value --max_transitions 100000 \
    && $PY train_simdist_adapt.py $COMMON --save_dir "$OUT" \
      --adapt_mode dyn   --max_transitions 100000 \
      --init_ckpt "$OUT.phaseA/latest.pt" ;;
  tpa_valabl_d2v_nd*)
    $PY train_simdist_adapt.py $COMMON --save_dir "$OUT.phaseA" \
      --adapt_mode dyn   --max_transitions 100000 \
    && $PY train_simdist_adapt.py $COMMON --save_dir "$OUT" \
      --adapt_mode value --max_transitions 100000 \
      --init_ckpt "$OUT.phaseA/latest.pt" ;;
esac
