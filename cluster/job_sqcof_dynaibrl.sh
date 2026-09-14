#!/bin/bash
#SBATCH --job-name=sqcof_di
#SBATCH --array=1-40
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
# ^ --account=def-<pi> at submit time:
#       sbatch --account=def-<pi> cluster/job_sqcof_dynaibrl.sh
#
# Dyna-IBRL on Square_D0 (rotw1) and Coffee_D0 (G5x), mirroring the Threading
# batch in cluster/job_threading_margin.sh so the three tasks are read the same
# way. SIX arms per task = {5-head dynens, single-head plain} x {annealed margin,
# constant 0.01, constant 0.01 + SGFT}. Which of them actually run is decided by
# $JOBS + --array, NOT by this file.
#
#   arm                          stage-1   margin(t)              sgft_coef
#   *_dynaibrl                   dynens    0.05 -> 0 over 50k     0
#   *_dynaibrl_m001              dynens    const $CONST_MARGIN     0
#   *_dynaibrl_sgft              dynens    const $CONST_MARGIN     1.0
#   *_dynaibrl_plain             plain     0.05 -> 0 over 50k     0
#   *_dynaibrl_plain_m001        plain     const $CONST_MARGIN     0
#   *_dynaibrl_plain_sgft        plain     const $CONST_MARGIN     1.0
#
# The two _m001 arms were added to FIX the confound below. Before them, a _sgft
# arm differed from its non-shaped counterpart in TWO fields (sgft_coef AND the
# margin schedule), because the annealed shaped arm had been measured to do badly
# and the schedule was quietly dropped for the shaped arms only. _sgft vs _m001
# is now a single-variable pair; _m001 vs the annealed arm is the other single
# variable. ⚠️ _sgft read against the ANNEALED arm is still two fields -- the
# comparison you want is _sgft vs _m001.
#
# ------------------------------------------------------------ what is measured
# Dyna-IBRL has NO planner. score/score is the IBRL hybrid -- argmax over
# {RL actor, FROZEN stage-2 prior} using the critic, one forward pass per step,
# no CEM and no dynamics rollout at deploy time. --eval_policy_only 1 logs the
# frozen prior alongside as score/policy_only, which is the floor every arm has
# to beat (it is constant within a run, since stage 3 never touches the prior).
#
# ------------------------------------------------------------------ the anchor
# ANCHOR_COEF defaults to 0 (off), matching the Threading batch. Turn it on with
#   ANCHOR_COEF=1.0 sbatch --export=ALL,ANCHOR_COEF ...
# Coffee is the task where this is likely to matter: docs/tasks.md 40e measured
# the stage-2 anchor as THE lever there (+0.40 on G5x, against +0.03 for stage-3
# adapt), and Dyna-IBRL starts from exactly that stage-2 policy. Square's own
# adapt arm uses the much smaller 0.1.
#
# The anchor reads a state pool, not the full source set: _build_bc_anchor draws
# bc_anchor_episodes episodes and keeps 40000 states, so the 5.5 GB / 7.7 GB
# originals were replaced with 4000-episode extracts (scripts/extract_anchor_pool.py),
# 62 MB and 266 MB. Keeping 4000 > bc_anchor_episodes=2000 means each seed still
# draws its own subset.
#
# ------------------------------------------------------------- demo / stage-1
# Both tasks: sat_400000 target demos, 25 episodes, --target_num_data 20 (the
# Threading口径). Two stage-1 checkpoints per task, 5-head and single-head at the
# same horizon 5 and the same source data, so plain-vs-ensemble is not confounded
# with horizon.
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

# ⚠️ --array MUST match `wc -l < $JOBS`. It did not: the header said 1-20 while
# the list held 40 lines, so every submission silently dropped the 20 cof_* rows.
JOBS=${JOBS:-cluster/jobs_sqcof_dynaibrl.txt}
[ -f "$JOBS" ] || { echo "no job list at JOBS=$JOBS"; exit 1; }
JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
[ -n "$JOB" ] || { echo "task $SLURM_ARRAY_TASK_ID is past the end of $JOBS \
($(wc -l < "$JOBS") lines) -- --array is wider than the list"; exit 1; }
ARM=${JOB% *}; SEED=${JOB##* }
echo "[jobs] $JOBS"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"

MARGIN_INIT=${MARGIN_INIT:-0.05}
MARGIN_FINAL=${MARGIN_FINAL:-0.0}
MARGIN_STEPS=${MARGIN_STEPS:-50000}
# Constant margin for the SGFT arms; init==final makes margin(t) constant while
# still going through the schedule code path. 0.01 is the best constant measured
# on Threading (single seed, 75-100k steady, policy-only): 0.01 -> 0.873,
# 0.05 -> 0.775, anneal 0.05->0.01 -> ~0.88, anneal 0.05->0.00 -> ~0.855.
# Renamed from SGFT_MARGIN: the constant is used by the shaped arms and the
# unshaped _m001 arms alike. The old name is still accepted as input.
CONST_MARGIN=${CONST_MARGIN:-${SGFT_MARGIN:-0.01}}
ANCHOR_COEF=${ANCHOR_COEF:-0.0}

case $ARM in
  sq_*) TASK=square
        EXP_DIR=${EXP_DIR:-exps/square_rotw1_6arm}
        DEMOS=release/data/mimicgen/square_sat_400000_s${SEED}/dataset.hdf5
        IBRL_CFG=release/cfgs/mimicgen/square_rotw1_ibrl_m001.yaml
        BC_PRIOR=exps/mimicgen/square_il_ft_target_rotw1_seed${SEED}/model0.pt
        # SOURCE-domain IBRL expert, the one stage-1 distilled from (source_agent
        # in square_simdist_rotw1_dynens_h5_seed1/cfg.yaml). Only *_qsrc uses it.
        SRC_EXPERT=exps/mimicgen/square_ibrl_rotw1_sig01_snap_seed1/model0.pt
        STAGE1_DYNENS=exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/latest.pt
        STAGE1_PLAIN=exps/mimicgen/square_simdist_rotw1_h5_seed1/latest.pt
        ANCHOR_PATH=release/data/mimicgen/square_anchor_pool_4k/dataset.hdf5 ;;
  cof_*) TASK=coffee
        EXP_DIR=${EXP_DIR:-exps/coffee_g5x_3arm}
        DEMOS=release/data/mimicgen/coffee_g5x_sat400k_s${SEED}/dataset.hdf5
        IBRL_CFG=release/cfgs/mimicgen/coffee_g5x_ibrl_m001.yaml
        BC_PRIOR=exps/mimicgen/coffee_il_ft_target_g5x_seed${SEED}/model0.pt
        SRC_EXPERT=exps/mimicgen/coffee_ibrl_source_sig01_seed1/model0.pt
        # ⚠️ NOT coffee_simdist_{diversev1,plain}_h5_seed1 -- those are the C50-era
        # stage-1s and carry a DIFFERENT target (friction 0.5 / lid 50 / mass 3 /
        # pod 2). Every already-finished run in exps/coffee_g5x_3arm inherits
        # coffee_s1g5x_* and runs at (0.2, 5.0, 8.0, 5.0), which is also what
        # coffee_d0_state_baseline_target_g5x.yaml and coffee_ibrl_target_G5x_seed1
        # use. Confirm with cfg.yaml's stage1_ckpt of an existing run, never by
        # the stage-1 directory name (the two families are named inconsistently).
        STAGE1_DYNENS=exps/mimicgen/coffee_s1g5x_dynens/latest.pt
        STAGE1_PLAIN=exps/mimicgen/coffee_s1g5x_plain/latest.pt
        ANCHOR_PATH=release/data/mimicgen/coffee_anchor_pool_4k/dataset.hdf5 ;;
  *) echo "unknown arm '$ARM' (expected sq_* or cof_*)"; exit 1 ;;
esac
# The value-adapt ablation gets its own PER-TASK wandb project (see
# job_threading_margin.sh). Per task, not one shared project: comparing the five
# arms WITHIN a task is the question; Square and Coffee scores do not belong on
# the same chart. Override with VALABL_DIR.
case $ARM in
  sq_valabl_*)  EXP_DIR=${VALABL_DIR:-exps/valabl_square} ;;
  cof_valabl_*) EXP_DIR=${VALABL_DIR:-exps/valabl_coffee} ;;
  sq_lamA_*|sq_dsrcB_*)   EXP_DIR=${LAMTUNE_DIR:-exps/lamtune_square} ;;
  cof_lamA_*|cof_dsrcB_*) EXP_DIR=${LAMTUNE_DIR:-exps/lamtune_coffee} ;;
  sq_optim_*)   EXP_DIR=${OPTIM_DIR:-exps/optim_square} ;;
  cof_optim_*)  EXP_DIR=${OPTIM_DIR:-exps/optim_coffee} ;;
  sq_anch_*)    EXP_DIR=${ANCHFORM_DIR:-exps/anchform_square} ;;
  cof_anch_*)   EXP_DIR=${ANCHFORM_DIR:-exps/anchform_coffee} ;;
esac
OUT=$EXP_DIR/${ARM}${SUFFIX:-}_seed${SEED}

for f in "$DEMOS"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_sqcof_dynaibrl.sh)"; exit 1; }
done
case $ARM in
  *_ibrl_m001_qsrc|*_ibrl_m0_qsrc)
    for f in "$BC_PRIOR" "$IBRL_CFG" "$SRC_EXPERT"; do
        [ -f "$f" ] || { echo "MISSING asset: $f  (untar the ibrl-prior bundle)"; exit 1; }
    done ;;
  *_ibrl_m0|*_ibrl_m001)
    # No world model in this arm at all -- it needs the BC prior and the cfg
    # instead. The prior is trained on the workstation and shipped in the tarball
    # (scripts/train_target_bc_priors.sh + cluster/pack_ibrl_priors.sh).
    for f in "$BC_PRIOR" "$IBRL_CFG"; do
        [ -f "$f" ] || { echo "MISSING asset: $f  (untar the ibrl-prior bundle)"; exit 1; }
    done ;;
  sq_plain_d600000|sq_plain_d400000|cof_plain) NEED=$STAGE1_PLAIN
    [ -f "$NEED" ] || { echo "MISSING stage-1: $NEED"; exit 1; } ;;
  *_plain*) NEED=$STAGE1_PLAIN
    [ -f "$NEED" ] || { echo "MISSING stage-1: $NEED"; exit 1; } ;;
  *) NEED=$STAGE1_DYNENS
    [ -f "$NEED" ] || { echo "MISSING stage-1: $NEED"; exit 1; } ;;
esac
if [ "$(echo "$ANCHOR_COEF > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
    [ -f "$ANCHOR_PATH" ] || { echo "MISSING anchor pool: $ANCHOR_PATH"; exit 1; }
    ANCHOR_ARGS="--bc_anchor_coef $ANCHOR_COEF --bc_anchor_path $ANCHOR_PATH"
else
    ANCHOR_ARGS=""
fi

COMMON="--save_dir $OUT --seed $SEED --target_demo_path $DEMOS --target_num_data 20 \
  --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
  --num_eval_episode 100 --num_eval_proc 10 \
  --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
  --use_wb $USE_WB --state_only_no_render 1 $ANCHOR_ARGS"

case $ARM in
  sq_dynaibrl|cof_dynaibrl)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS $COMMON \
      --dyna.prior_margin_init $MARGIN_INIT --dyna.prior_margin_final $MARGIN_FINAL \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  sq_valabl_dyn|sq_valabl_joint|sq_valabl_v2d|sq_valabl_d2v|sq_valabl_marg|cof_valabl_dyn|cof_valabl_joint|cof_valabl_v2d|cof_valabl_d2v|cof_valabl_marg)
    # ---- value-adapt ablation (docs/on-policy-adaptation.md 2) --------------
    #   valabl_dyn     (i)   dynamics only          -- the existing pipeline
    #   valabl_joint   (ii)  value + dynamics, 100k, together
    #   valabl_v2d     (iii) value 100k THEN dynamics 100k
    #   valabl_d2v     (iv)  dynamics 100k THEN value 100k
    #   valabl_marg    (v)   marginal (residual) value + dynamics, 100k
    #
    # Every arm shares this run's stage-1, demos, planner and eval; the only thing
    # that varies is which heads move and in what order. The sequential arms run
    # BOTH phases inside one job (2 x 100k ~= 4.6h) -- phase A writes into
    # $OUT.phaseA and phase B warm-starts from it via --init_ckpt while keeping
    # --stage1_ckpt, since that path reads dirname/cfg.yaml as a MainConfig and an
    # adapt run writes an AdaptConfig there.
    #
    # ⚠️ (v) fits its residual by MSE on Monte-Carlo returns while (ii)-(iv) use
    # the value head's two-hot CE. That is inherent: the head is categorical
    # (discrete_heads=1, 101 bins) and a scalar residual has no logit-space
    # counterpart, so the residual sits on the DECODED scalar -- exactly the
    # quantity MPPI ranks with. Read (v) against (ii) knowing the loss differs.
    #
    # ⚠️ Expected outcome is negative: on-policy-adaptation 2 already measured all
    # three orders collapsing on Threading, and findings 33 gives the mechanism
    # (MPPI's whole candidate-discrimination signal is std_i(gamma^H V)=0.0068).
    # This batch is the three-task confirmation, not a search.
    # Square needs its planner trust region; Coffee uses defaults (see sq_plain).
    case $ARM in sq_*) VA_PSET='--planner_set init_std=0.15,base_policy_std=0.0' ;;
                 *)    VA_PSET='' ;; esac
    VA_COMMON="--stage1_ckpt $STAGE1_PLAIN --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.0 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 $VA_PSET"
    case $ARM in
      *_dyn)   $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT \
                 --adapt_mode dyn   --max_transitions 100000 ;;
      *_joint) $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT \
                 --adapt_mode both  --max_transitions 100000 ;;
      *_marg)  $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT \
                 --adapt_mode both  --marginal_value 1 --max_transitions 100000 ;;
      *_v2d)   $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT.phaseA \
                 --adapt_mode value --max_transitions 100000 \
               && $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT \
                 --adapt_mode dyn   --max_transitions 100000 \
                 --init_ckpt $OUT.phaseA/latest.pt ;;
      *_d2v)   $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT.phaseA \
                 --adapt_mode dyn   --max_transitions 100000 \
               && $PY train_simdist_adapt.py $VA_COMMON --save_dir $OUT \
                 --adapt_mode value --max_transitions 100000 \
                 --init_ckpt $OUT.phaseA/latest.pt ;;
    esac ;;
  sq_anch_f5|cof_anch_f5|sq_anch_f1|cof_anch_f1)
    # ===== F5 (saturating Hellinger^2) vs F1 (the deployed KL penalty) on Square
    # and Coffee. On Threading the two were measured TIED, not F5 ahead: simdist
    # 43 Phase 1 has F5 0.946 +- .014 vs F1 0.957 (delta -0.011, t=0.9), and the
    # v2 calibration puts their bc_loss within 0.1% of each other at the same
    # held-out D_SRC. The question here is whether "the form does not matter"
    # survives a change of task, which is worth its own cells -- F5 is the only
    # form that pushes the deviation tail to the OTHER side of F1 (43 built it
    # exactly to break the tail/bc_loss collinearity).
    #
    # ⚠️ DO NOT run the *_anch_f1 arms unless you need a fresh copy: each task's
    # EXISTING anchored baseline already IS F1 at the same working point --
    # sq_dynens_anch_d400000 (10 seeds) and cof_dynens_anch (10 seeds) match these
    # arms field-for-field (stage-1, demo, num_data None, lambda, planner_set,
    # eval_disagree_coefs, 100k, collect_disagree_coef 0.5). The only difference
    # is --bc_anchor_holdout 1, which records held-out D_SRC and does not touch
    # the loss. The f1 arms are kept only so the pair can be regenerated from one
    # script if a harness detail ever changes. cluster/jobs_sqcof_anchf5.txt lists
    # the F5 arms alone.
    #
    # ---- CALIBRATED per task on HELD-OUT D_SRC, 2026-09-12 (exps/anchform_cal) --
    #   bash scripts/calibrate_f5_sqcof.sh ; python scripts/report_f5_cal.py exps/anchform_cal
    # Each F5 cell is matched to ITS OWN task's F1 working point, tau fixed at 0.03:
    #   task       F1 lambda  F1 held-out  F5 lambda  F5 held-out  F5 bc_loss vs F1
    #   Threading  1.0        0.0317       0.013      0.0318       +0.1%
    #   Coffee     1.0        0.0322       0.011      0.0327       -5.0%
    #   Square     0.1        0.0747       0.006      0.0863       +0.9%
    #
    # ⚠️ The F5/F1 lambda RATIO is not portable either: 0.013 and 0.011 on
    # Threading/Coffee but 0.070 on Square (5x). A saturating form's stiffness is
    # lambda/tau, not lambda -- 43's gotcha #1 is F5 once running 1400x too strong
    # because the F1 habit of lambda~1 was copied across forms. Re-run
    # scripts/calibrate_f5_sqcof.sh for any new task or tau.
    #
    # ⚠️ Each task's F1 sits at a DIFFERENT working point, and deliberately so:
    # Threading/Coffee at ~0.032 are 1.6x TIGHTER than the portable rule
    # train_simdist_adapt.py recommends (held-out D_SRC 0.05, best leave-one-
    # target-out), Square at 0.0747 is 1.5x looser. Square is not simply
    # mis-tuned: lambda 1.0 there measures 0.692 against 0.759 for NO anchor at
    # all, so its looser point is a real finding, and the F1/F5 pair here is
    # matched to that point rather than to Threading's. Cross-TASK comparison of
    # these cells is therefore not apples-to-apples; the d-sweep (dsrcB_* arms) is
    # what puts all three on one axis.
    # ========================================================================
    CAL=${CAL:-0}
    if [ "$CAL" = 1 ]; then F5_ARGS=(--max_transitions 0); else F5_ARGS=(--max_transitions 100000); fi
    case $ARM in
      sq_*) B_DEMOS=release/data/mimicgen/square_sat_400000_s${SEED}/dataset.hdf5
            B_POOL=release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5
            B_DIS="0,0.05,0.1,0.3"
            B_PSET=(--planner_set "init_std=0.15,base_policy_std=0.0")
            F5LAM=${ANCH_F5_SQ:-0.006} ; F1LAM=0.1 ;;
      *)    B_DEMOS=$DEMOS
            B_POOL=release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5
            B_DIS="0,0.01,0.05,0.1"
            B_PSET=()
            F5LAM=${ANCH_F5_COF:-0.011} ; F1LAM=1.0 ;;
    esac
    case $ARM in
      *_f5) FORM=(--bc_anchor_coef $F5LAM --bc_anchor_sat ${ANCH_TAU:-0.03}) ;;
      *)    FORM=(--bc_anchor_coef ${ANCH_F1:-$F1LAM}) ;;
    esac
    for f in "$STAGE1_DYNENS" "$B_DEMOS" "$B_POOL"; do
      [ -f "$f" ] || { echo "MISSING asset: $f"; exit 1; }
    done
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      "${B_PSET[@]}" "${FORM[@]}" \
      --bc_anchor_path $B_POOL --bc_anchor_holdout 1 "${F5_ARGS[@]}" \
      --target_demo_path $B_DEMOS \
      --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_disagree_coefs "${PEN:-0}" \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.5 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  cof_optim_*)
    # exp1 on Coffee: optimistic sweep. Penalty rides along inside each run via
    # a single eval per point (score + policy_only). Head size and pessimism are
    # eval-time knobs and are swept OFFLINE on saved snapshots.
    #
    # Matches cof_dynens_anch field-for-field apart from collect_disagree_coef:
    # same stage-1, coffee_g5x_sat400k per-seed demos, num_data left at None
    # (the whole coffee_g5x_3arm family runs that way), anchor coef 1.0, no
    # planner_set, 100k.
    V=${ARM#cof_optim_}
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --bc_anchor_coef 1.0 \
      --bc_anchor_path release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5 \
      --target_demo_path $DEMOS \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_disagree_coefs "${PEN:-0}" \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef $V --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_optim_*)
    # ===== exp1 on Square: optimistic-exploration sweep, same design as
    # Threading's optim_* (see job_threading_margin.sh). Penalty and ensemble
    # so one arm per optimistic value is enough.
    #
    # ONE eval per point (score + policy_only, PEN default 0). The head and
    # penalty sweeps are eval-time knobs -- they re-score a fixed checkpoint --
    # so they run OFFLINE on saved snapshots instead of doubling the wall clock
    # of every run here. Square could not have swept heads anyway: its 10-head
    # stage-1 is trained to 130k distill steps against the 5-head one's 1.2M
    # (9x), so sweeping off it would measure an undertrained model.
    V=${ARM#sq_optim_}
    O_CK=$STAGE1_DYNENS
    [ -f "$O_CK" ] || { echo "MISSING stage-1: $O_CK"; exit 1; }
    $PY train_simdist_adapt.py --stage1_ckpt $O_CK --save_dir $OUT --seed $SEED \
      --planner_set "init_std=0.15,base_policy_std=0.0" \
      --bc_anchor_coef 0.1 \
      --bc_anchor_path release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5 \
      --target_demo_path release/data/mimicgen/square_sat_600000_s${SEED}/dataset.hdf5 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_disagree_coefs "${PEN:-0}" \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef $V --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_lamA_*|cof_lamA_*)
    # LINE A on Square and Coffee: fixed bc_anchor_coef sweep, the companion to
    # Threading's lamA_* (see job_threading_margin.sh). Shows what the multiplier
    # does per task; the dsrcB_* arms show what happens when the CONSTRAINED
    # quantity is held fixed instead.
    #
    # Each task keeps its own harness, so the lambda axis is the only thing that
    # moves within a task. Note the deployed lambdas already differ 10x between
    # tasks (Square 0.1, Coffee 1.0) at held-out D_SRC 0.0747 vs 0.0322 -- which
    # is the non-portability this line documents.
    #
    # Only one pessimism coef (PEN, default 0): score + policy_only is what this
    # line needs, and the 4-coef sweep is most of the wall clock.
    LV=${ARM##*lamA_}
    case $ARM in
      sq_*) A_DEMOS=release/data/mimicgen/square_sat_600000_s${SEED}/dataset.hdf5
            A_POOL=release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5
            A_PSET=(--planner_set "init_std=0.15,base_policy_std=0.0") ;;
      *)    A_DEMOS=$DEMOS
            A_POOL=release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5
            A_PSET=() ;;
    esac
    for f in "$STAGE1_DYNENS" "$A_DEMOS" "$A_POOL"; do
      [ -f "$f" ] || { echo "MISSING asset: $f"; exit 1; }
    done
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      "${A_PSET[@]}" --bc_anchor_coef $LV --bc_anchor_path $A_POOL \
      --target_demo_path $A_DEMOS \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_disagree_coefs "${PEN:-0}" \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.5 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_dsrcB_*|cof_dsrcB_*)
    # LINE B on Square and Coffee: hold the HELD-OUT D_SRC at d with the dual
    # controller, lambda adaptive. The point of the line is that ONE d transfers
    # across tasks, so these must mirror each task's own anchored baseline
    # field-for-field and change only the anchor's control target:
    #   Square  sq_dynens_anch_d400000 : 5-head stage-1, sat_400000, num_data
    #           None, bc_anchor_coef 0.1, planner_set init_std=0.15/base 0, and
    #           eval_disagree_coefs '0,0.05,0.1,0.3'  (NOT Threading's list)
    #   Coffee  cof_dynens_anch        : coffee_s1g5x_dynens, sat400k, num_data
    #           None, bc_anchor_coef 1.0, no planner_set,
    #           eval_disagree_coefs '0,0.01,0.05,0.1'
    # ⚠️ The per-task anchor lambda differs by 10x (0.1 vs 1.0) -- which is the
    # very non-transferability line A documents, and the reason line B exists.
    # bc_anchor_coef below only SEEDS the controller's search.
    #
    # ⚠️ The anchor pools are the FULL source-diverse sets (5.2 GB / 7.2 GB), the
    # same files the baselines used. The 4k extracts would put these arms on a
    # different state coverage, which simdist 44c measured at -0.123.
    DV=${ARM##*dsrcB_}
    case $ARM in
      sq_*) B_STAGE1=$STAGE1_DYNENS
            B_DEMOS=release/data/mimicgen/square_sat_400000_s${SEED}/dataset.hdf5
            B_POOL=release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5
            B_SEED_LAM=0.1
            B_DIS="0,0.05,0.1,0.3"
            B_PSET=(--planner_set "init_std=0.15,base_policy_std=0.0") ;;
      *)    B_STAGE1=$STAGE1_DYNENS
            B_DEMOS=$DEMOS
            B_POOL=release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5
            B_SEED_LAM=1.0
            B_DIS="0,0.01,0.05,0.1"
            B_PSET=() ;;
    esac
    for f in "$B_STAGE1" "$B_DEMOS" "$B_POOL"; do
      [ -f "$f" ] || { echo "MISSING asset: $f"; exit 1; }
    done
    $PY train_simdist_adapt.py --stage1_ckpt $B_STAGE1 --save_dir $OUT --seed $SEED \
      "${B_PSET[@]}" \
      --bc_anchor_coef $B_SEED_LAM --bc_anchor_target_dsrc $DV \
      --bc_anchor_path $B_POOL \
      --target_demo_path $B_DEMOS \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_disagree_coefs "${PEN:-0}" \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.5 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_plain_d600000|sq_plain_d400000|cof_plain)
    # SimDist MPPI adapt on the single-head stage-1 -- the PLANNING reference the
    # dyna/model-free arms are measured against. Deliverable IS the planner, so
    # score/score is the MPPI score and --eval_policy_only 1 logs the bare policy
    # alongside as score/policy_only. Mirrors job_threading_margin.sh's plain_h5.
    #
    # ⚠️ NO --target_num_data: the whole exps/square_rotw1_6arm and
    # exps/coffee_g5x_3arm plain/dynens family runs with it at None (= every
    # episode in the file), and the arm names below ARE those groups' names, so
    # these runs land in the same wandb group and extend them 100k -> 300k.
    # Passing 20 (as $COMMON does for the dyna arms) would fork a different demo
    # budget into an existing group. Square also picks its demo set by arm name:
    # d600000 -> square_sat_600000, d400000 -> square_sat_400000 ($DEMOS default).
    #
    # collect_disagree_coef 0 and eval_disagree_coefs "" because geometric
    # disagreement needs >1 dynamics head; both are undefined on a single head.
    # ~7h for 300k (round ~30s, MPPI eval ~295s, eval_every_rounds 5).
    case $ARM in
      sq_plain_d600000) PDEMOS=release/data/mimicgen/square_sat_600000_s${SEED}/dataset.hdf5 ;;
      *) PDEMOS=$DEMOS ;;
    esac
    [ -f "$PDEMOS" ] || { echo "MISSING demos: $PDEMOS"; exit 1; }
    # ⚠️ Square needs an explicit planner trust region; Coffee does not. ALL 60
    # runs in exps/square_rotw1_6arm (plain and dynens, d400000 and d600000) were
    # launched with init_std=0.15,base_policy_std=0.0, while every run in
    # exps/coffee_g5x_3arm has planner_set='' (defaults). Omitting it on Square
    # leaves the planner at the stage-1 default init_std -- a much wider trust
    # region, and tasks.md 40c measured this single knob at up to 40 points. That
    # is what made the first sq_plain_* runs look far worse than sq_plain_d400000.
    # CLAUDE.md rule 9: planner hyperparameters do not port across tasks, so this
    # is per-task and must NOT be copied to cof_plain.
    case $ARM in sq_*) PSET=(--planner_set "init_std=0.15,base_policy_std=0.0") ;;
                 *)    PSET=() ;; esac
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      "${PSET[@]}" \
      --target_demo_path $PDEMOS \
      --max_transitions 300000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.0 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_dynaibrl_m0|cof_dynaibrl_m0|sq_dynaibrl_plain_m0|cof_dynaibrl_plain_m0)
    # ⚠️ 300k, not the 100k $COMMON sets: these arms are asked how a margin-free
    # IBRL behaves over a long horizon, matching the 300k model-free *_qsrc arms.
    # The repeated --max_transitions is deliberate -- argparse takes the LAST one,
    # verified. Measured cost on this task family: round ~37s, eval ~168s,
    # eval_every_rounds 5 -> ~5.3h for 300k, inside the 14h wall.
    #
    # NOTE the eval cadence is already dyna's own (adapt_interval 1000 x
    # eval_every_rounds 5 = every 5000), i.e. the same grid the 300k model-free
    # arms now use via log_per_step 5000.
    case $ARM in *_plain_m0) CK=$STAGE1_PLAIN ;; *) CK=$STAGE1_DYNENS ;; esac
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $CK $COMMON \
      --max_transitions 300000 \
      --dyna.prior_margin_init 0.0 --dyna.prior_margin_final 0.0 \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  sq_dynaibrl_m001|cof_dynaibrl_m001)
    # Identical to the arm above except margin(t): CONSTANT $CONST_MARGIN instead
    # of the 0.05 -> 0 anneal. sgft_coef stays 0.
    #
    # This is what gives the *_sgft arms a controlled partner at last: against
    # THIS arm they differ in sgft_coef and nothing else. It also asks the
    # separate question of whether the constant is simply the better schedule --
    # Threading's fixed-margin sweep (single seed, policy-only, 75-100k steady)
    # put 0.01 -> 0.873 above anneal-to-0 -> ~0.855, which is inside that sweep's
    # noise and was never replicated on a fresh seed (CLAUDE.md rules 1 and 3).
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS $COMMON \
      --max_transitions 300000 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  sq_dynaibrl_sgft|cof_dynaibrl_sgft)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS $COMMON \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS --dyna.sgft_coef 1.0 ;;
  sq_dynaibrl_plain|cof_dynaibrl_plain)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $COMMON \
      --dyna.prior_margin_init $MARGIN_INIT --dyna.prior_margin_final $MARGIN_FINAL \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  sq_dynaibrl_plain_m001|cof_dynaibrl_plain_m001)
    # Single-head counterpart of *_dynaibrl_m001, and the single-variable partner
    # of *_dynaibrl_plain_sgft.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $COMMON \
      --max_transitions 300000 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  sq_dynaibrl_plain_sgft|cof_dynaibrl_plain_sgft)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $COMMON \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS --dyna.sgft_coef 1.0 ;;
  sq_ibrl_m0|cof_ibrl_m0)
    # Vanilla IBRL, no margin, no critic warm-start; single-variable partner of
    # *_ibrl_m0_qsrc. See job_threading_margin.sh's ibrl_m0.
    $PY train_rl.py --config_path $IBRL_CFG \
      --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR --preload_datapath $DEMOS \
      --num_train_step 300000 --log_per_step 5000 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_ibrl_m0_qsrc|cof_ibrl_m0_qsrc)
    # ⚠️ 300k (not the 100k the other arms use) + log_per_step 5000 to match
    # dyna's eval cadence. See job_threading_margin.sh's ibrl_m0_qsrc for why.
    # *_ibrl_m001 (no warm-start) stays at 100k and is NOT directly comparable.
    #
    # Plain IBRL (no margin) + source-expert critic warm-start. One field from
    # *_ibrl_m001_qsrc; see job_threading_margin.sh's ibrl_m0_qsrc for why this
    # is the arm that tests whether the margin was only a cold-start crutch.
    $PY train_rl.py --config_path $IBRL_CFG \
      --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR --preload_datapath $DEMOS \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --num_train_step 300000 --log_per_step 5000 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_ibrl_m001_qsrc|cof_ibrl_m001_qsrc)
    # ibrl_m001 + the SOURCE expert's critic as Q's starting point; differs from
    # the arm below in exactly ONE field. See job_threading_margin.sh's
    # ibrl_m001_qsrc comment for the prior evidence (mixed, and none of it is
    # this measurement).
    $PY train_rl.py --config_path $IBRL_CFG \
      --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR --preload_datapath $DEMOS \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --num_train_step 300000 --log_per_step 5000 \
      --prior_margin_init $CONST_MARGIN --prior_margin_final $CONST_MARGIN \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  sq_ibrl_m001|cof_ibrl_m001)
    # Model-free IBRL at the same constant margin as the _m001 dyna arms, same
    # 100k budget, same 20 target demos. No world model, so no stage-1 ckpt.
    #
    # The env comes from $IBRL_CFG, generated by scripts/make_ibrl_m001_cfg.py
    # from the STAGE-1 cfg -- the same source of truth the dyna arms' eval env
    # comes from (train_simdist_adapt.py:398). Regenerate it rather than editing
    # it by hand, and re-read its ⚠️ about Coffee.
    #
    # ⚠️ TWO differences from any dyna arm, not one: no world model AND a
    # different prior (target-domain BC here; the frozen stage-2 distilled policy
    # there). It bounds how much of the margin's value needs a model; it does not
    # isolate the model.
    $PY train_rl.py --config_path $IBRL_CFG \
      --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR --preload_datapath $DEMOS \
      --prior_margin_init $CONST_MARGIN --prior_margin_final $CONST_MARGIN \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac
RC=$?
# NOT `rc=$?` after a $(date): that reports date's exit code and turns a truncated
# run into a green DONE.
echo "[$(date)] task $SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED rc=$RC"
grep -q "^100000: score/score" $OUT/train.log 2>/dev/null \
  || { echo "INCOMPLETE: $OUT never logged step 100000"; exit 1; }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
