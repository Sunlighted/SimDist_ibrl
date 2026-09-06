#!/bin/bash
#SBATCH --job-name=thr_margin
#SBATCH --array=1-60
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
# ^ --account=def-<pi> at submit time:
#       sbatch --account=def-<pi> cluster/job_threading_margin.sh
#
# Threading_D0, matched 100k-transition budget, into exps/simdist_threading_3arm
# so these land in the SAME wandb project as the existing groups and overlay on
# one chart. Four arms x 10 seeds.
#
# The question this batch asks: with the IBRL prior margin annealed 0.05 -> 0,
# how does an actor that learns to ACT compare against the planner it is
# deferring to -- when both are built on the SAME single-head dynamics?
#
#   plain_h5             plain SimDist MPPI, single dynamics head, horizon 5.
#                        The planning reference. Deliverable IS the planner, so
#                        score/score is the MPPI score; --eval_policy_only 1
#                        logs the bare policy alongside as score/policy_only.
#   dynaibrl_plain       Dyna + IBRL prior margin, annealed 0.05 -> 0 over 50k,
#                        on the SAME single-head stage-1 as plain_h5.
#   dynaibrl_plain_m001  the same with margin(t) CONSTANT at CONST_MARGIN (0.01)
#                        instead of the anneal, shaping still off.
#   dynaibrl_plain_sgft  constant margin + SGFT shaping (--dyna.sgft_coef 1.0).
#                        The annealed shaped arm performed badly, so the schedule
#                        was dropped for the shaped arms -- which is why this arm
#                        differs from dynaibrl_plain in TWO fields. Read it
#                        against dynaibrl_plain_m001 (one field: sgft_coef), NOT
#                        against dynaibrl_plain.
#   dynaibrl_dynens_m001 5-head counterpart of dynaibrl_plain_m001; the same
#                        single-variable partner for dynaibrl_dynens_sgft.
#   ibrl_m005anneal0     model-free IBRL, same 0.05 -> 0 margin schedule. No
#                        world model anywhere -- the baseline that says how much
#                        of the margin's value needs a model at all.
#   ibrl_m001            the same at the CONSTANT margin the _m001 dyna arms use,
#                        so the model-free reference exists at both settings.
#
# ------------------------------------------------- why the plain dynamics head
# The existing dynaibrl_* runs on the workstation all sit on the 5-head
# dynens stage-1, so "Dyna+margin vs planning" was confounded with "ensemble vs
# single head". Every arm here that uses a world model uses
# threading_simdist_diversev2_h5_seed1 -- one head, horizon 5, same diverse_v2
# source data as the ensemble. Ensemble effects are then a separate axis, not a
# term inside this comparison.
#
# --------------------------------------------------------- the margin schedule
# margin(t) anneals prior_margin_init -> prior_margin_final over
# prior_margin_steps and then STAYS at final. Evidence for 0.05 as the init, from
# the fixed-margin sweep in rl/simdist_dyna_ibrl.py:
#     margin 0    -> 0.000   argmax decided by critic noise; ~5.8% action
#                            deviation is enough to kill the episode
#     margin 0.05 -> 0.775   best; the actor clears the bar occasionally
#     margin 0.1  -> 0.632   bar too high, the actor essentially never acts
# High early (survive an uncalibrated critic), zero late (let a good actor
# through) is exactly what the schedule buys.
#
# ⚠️ steps=50000 on BOTH the dyna and the model-free arm, so the two see the same
# schedule. That is NOT what ibrl_m005_seed1 did: it used steps=100000 with
# init=final=0.05, i.e. a CONSTANT margin, no anneal at all. If you want the
# model-free arm to match that run's schedule length instead, change
# MARGIN_STEPS below to 100000 -- but change it in both arms or the comparison
# reintroduces the confound this batch removes.
#
# ---------------------------------------------------------------- demo standard
# All four arms use sat_160000 target demos (25 episodes, --target_num_data 20),
# the standing standard. docs/demo-collection.md 38 measured +0.139 from this
# switch alone, and dynens_h5 -> dynens_h5_slow (identical but for this path)
# moved round-0 from 0.274 to 0.417.
#
# ==> This is why ibrl_m005anneal0 does NOT reuse exps/mimicgen/il_ft_target_
#     seed*: those priors were fine-tuned on sat_400000, and only 6 of the 10
#     seeds exist. Run cluster/job_threading_bc_s160.sh FIRST; it produces
#     il_ft_target_s160_seed* for all 10 seeds on the correct demos.
#
# ==> Runs from this file are NOT comparable to the workstation dynaibrl_*,
#     ibrl_m005_seed1, or any pre-2026-08-31 dyna_* run (sat_400000 demos, the
#     dynens stage-1, and for dyna_* the four pre-fix defects documented in
#     cluster/job_threading_dyna_plain.sh).
set -e
cd $SLURM_SUBMIT_DIR
mkdir -p cluster/logs

# MIG-partitioned H100s cannot initialise an EGL device display -- `import mujoco`
# with MUJOCO_GL=egl dies at IMPORT time. osmesa lets the import succeed, and with
# --state_only_no_render 1 no GL context is ever created, so nothing is rendered.
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
# Compute nodes on fir have no internet, and `online` additionally deadlocks long
# runs (a logging thread parks in __skb_wait_for_more_packets and freezes the whole
# process). Default to offline; `wandb sync wandb/offline-run-*` from a login node
# afterwards.
export WANDB_MODE=${WANDB_MODE:-offline}
# The key is NEVER hardcoded here -- this file is tracked by git. See
# job_threading_dyna_plain.sh for the three supply routes and why the key is
# forwarded via APPTAINERENV_* rather than `apptainer --env` (argv is world
# readable in `ps`).
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
USE_WB=${USE_WB:-1}

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

# ------------------------------------------------------------------ preflight
# A failed CUDA init used to surface ~300 lines into train_simdist_dyna_ibrl.py
# as `RuntimeError: CUDA-capable device(s) is/are busy or unavailable` at the
# first .to(device) -- after the queue wait and after the env/asset checks had
# all passed, with nothing in the log to say WHICH device was refused. On a
# MIG-partitioned node the usual causes are (a) the slice is held by another
# process (compute mode is EXCLUSIVE_PROCESS on most CC nodes), (b) the job did
# not actually get the slice CUDA_VISIBLE_DEVICES names, or (c) /dev/nvidia-caps
# is not visible inside the container -- MIG needs it and --nv does not always
# bind it. Which of the three it is, is only distinguishable from the values
# below, so print them and then prove a context can be created BEFORE committing
# 12 h of walltime.
echo "[gpu] node=$(hostname)  job=${SLURM_JOB_ID:-?}  task=${SLURM_ARRAY_TASK_ID:-?}"
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[gpu] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
nvidia-smi -L 2>&1 | sed 's/^/[gpu] /' || echo "[gpu] nvidia-smi -L failed on the host"
ls /dev/nvidia-caps/ 2>/dev/null | tr '\n' ' ' | sed 's/^/[gpu] nvidia-caps: /'; echo
# Retry rather than fail: the observed failure mode is "CUDA-capable device(s)
# is/are busy or unavailable" raised at the FIRST .to(device), on a job whose
# CUDA_VISIBLE_DEVICES, device_count and nvidia-caps all check out -- and it hits
# some arms far more than others, which points at WHEN a task starts rather than
# what it runs. The likely mechanism is slice hand-off: an array task inherits a
# MIG slice whose previous occupant has not finished tearing its context down
# (these train scripts end in os._exit(0), which skips CUDA teardown entirely),
# so whoever is next in line on that slice sees it as busy for a few seconds.
# Waiting is the whole fix; the alternative is losing a 12 h allocation to a race
# that resolves itself in under a minute.
for ATTEMPT in 1 2 3 4 5; do
    # `|| PF=$?` is load-bearing twice over: `set -e` is on, so a bare failing
    # command would abort the script here and the retry below would never run;
    # and `|| true` would not do -- it makes $? always 0, so every attempt would
    # look like a success.
    PF=0
    $PY - <<'PREFLIGHT' || PF=$?
import sys, torch
ok = torch.cuda.is_available()
print(f"[gpu] torch {torch.__version__}  cuda_available={ok}  count={torch.cuda.device_count()}")
if not ok:
    sys.exit("no CUDA device visible inside the container")
try:
    torch.zeros(8, device="cuda").sum().item()          # forces context creation
    print(f"[gpu] context OK on {torch.cuda.get_device_name(0)}")
except Exception as e:
    sys.exit(f"cannot create a CUDA context -- {type(e).__name__}: {e}")
PREFLIGHT
    if [ "$PF" -eq 0 ]; then break; fi
    if [ $ATTEMPT -lt 5 ]; then
        echo "[gpu] preflight attempt $ATTEMPT failed (rc=$PF); retrying in $((ATTEMPT*30))s"
        sleep $((ATTEMPT * 30))
    fi
done
[ $PF -eq 0 ] || { echo "[gpu] preflight failed 5x over ~5 min; the slice is not"; \
                   echo "[gpu] transiently busy -- not starting the run"; exit $PF; }


# ⚠️ --array MUST match `wc -l < $JOBS`. It did not: the header said 1-40 while
# the list held 60 lines, so every submission silently dropped rows 41-60 (both
# dynaibrl_dynens arms).
JOBS=${JOBS:-cluster/jobs_threading_margin.txt}
[ -f "$JOBS" ] || { echo "no job list at JOBS=$JOBS"; exit 1; }
echo "[jobs] $JOBS"
JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
[ -n "$JOB" ] || { echo "task $SLURM_ARRAY_TASK_ID is past the end of $JOBS \
($(wc -l < "$JOBS") lines) -- --array is wider than the list"; exit 1; }
ARM=${JOB% *}; SEED=${JOB##* }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"

# save_dir's PARENT is the wandb project and the run name is the group
# (common_utils/py/run_cfg.py derives wb_exp/wb_run/wb_group from it as
# read-only properties -- --wb_* are NOT CLI args and argparse rejects them).
EXP_DIR=${EXP_DIR:-exps/simdist_threading_3arm}
SUFFIX=${SUFFIX:-}                 # e.g. SUFFIX=_v2 if a name would collide
OUT=$EXP_DIR/${ARM}${SUFFIX}_seed${SEED}
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5

# One head, horizon 5, diverse_v2 source. Shared by plain_h5 and both dyna arms
# so the planner and the actors reason over the SAME model.
STAGE1_PLAIN=exps/mimicgen/threading_simdist_diversev2_h5_seed1/latest.pt
# 5-head ensemble, a SEPARATE distillation run -- not the same weights with more
# heads. Its encoder, value head and policy head all differ, so an arm that swaps
# this in swaps all four channels the world model feeds (representation, prior,
# Q warm-start, imagined data) at once, not just the ensemble. Keep that in mind
# when reading dynaibrl_plain against dynaibrl_dynens: the difference is "these
# two stage-1s", not "ensemble averaging".
STAGE1_DYNENS=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
# Target-domain BC prior for the model-free arms, on sat_160000 (see above).
BC_PRIOR=exps/mimicgen/il_ft_target_s160_seed${SEED}/model0.pt
# SOURCE-domain IBRL expert; only ibrl_m001_qsrc uses it, for its critic alone.
# This is the same checkpoint stage-1 distilled from (source_agent in
# release/cfgs/mimicgen/threading_d0_state_simdist.yaml), so "the source critic"
# means the same thing here as everywhere else in the pipeline.
SRC_EXPERT=exps/mimicgen/threading_ibrl_source_seed1/model0.pt
# ⚠️ train_rl.py's dataclass DEFAULTS are critic 10x512 (no layer-norm) and actor
# 512/dropout 0.5. EVERY IBRL run in this project -- source experts, target
# experts, permmargin, and the Square/Coffee arms (which inherit them through
# --config_path) -- is critic 5x1024 + layer-norm and actor 1024/dropout 0.
# The Threading ibrl arms pass no --config_path, so without these five flags they
# would silently run an architecture used nowhere else, making them incomparable
# to the Square/Coffee arms AND to ibrl_m005_seed1 (which ibrl_m005anneal0 claims
# to reproduce). It also broke ibrl_m001_qsrc outright: the source expert's
# critic then matches no key, `skipped` is non-empty, and the load falls into the
# Fourier/PE repair path -> AttributeError: 'NoneType' has no attribute 'out_dim'.
#
# ⚠️⚠️ act_method: QAgentConfig defaults to "rl" (rl/q_agent.py:37) and train_rl.py
# does NOT flip it to "ibrl" just because --bc_policy was passed. Without the flag
# below the BC prior is loaded and printed ("Using BC policy ...") but never
# entered into the max-Q argmax: _act_ibrl is not called, prior_margin is dead
# code, and the arm is plain RL + preloaded demos wearing an IBRL name. The
# symptom is that two arms differing ONLY in prior_margin come out byte-identical,
# loss curve included. This is CLAUDE.md rule 8 again (5th time).
# The Square/Coffee arms escape it only because --config_path carries
# act_method: ibrl over from the target-expert template.
# The generated target-domain config (scripts/make_ibrl_m001_cfg.py). It carries
# train/eval_domain_shift = the TARGET the dyna arms run in, act_method=ibrl, the
# 5x1024 critic, sigma 0.1, episode_length 400 and strict_success -- i.e. every
# field that made the bare-CLI version incomparable. IBRL_ARCH below is kept as a
# belt-and-braces override; the values are identical to the cfg's.
IBRL_CFG=release/cfgs/mimicgen/threading_ibrl_m001.yaml
IBRL_ARCH="--q_agent.act_method ibrl \
  --q_agent.state_critic.num_q 5 --q_agent.state_critic.hidden_dim 1024 \
  --q_agent.state_critic.layer_norm 1 \
  --q_agent.state_actor.hidden_dim 1024 --q_agent.state_actor.dropout 0.0"

MARGIN_INIT=${MARGIN_INIT:-0.05}
MARGIN_FINAL=${MARGIN_FINAL:-0.0}
MARGIN_STEPS=${MARGIN_STEPS:-50000}
# The SGFT arms run a CONSTANT margin instead of the schedule above: the annealed
# 0.05 -> 0 version was measured to do badly with shaping on. init==final makes
# margin(t) = max(final, init*(1-t/steps)) constant for every t, so the schedule
# code path is exercised identically and only the value differs.
# 0.01 is the best constant measured on the workstation (single seed, 75-100k
# steady, policy-only): 0.01 -> 0.873, 0.05 -> 0.775, anneal 0.05->0.01 -> ~0.88,
# anneal 0.05->0.00 -> ~0.855.
# Renamed from SGFT_MARGIN, which is no longer accurate: the constant is used by
# the shaped arms, the _m001 arms and the model-free ibrl_m001 alike. The old
# name is still accepted so existing submissions keep working.
CONST_MARGIN=${CONST_MARGIN:-${SGFT_MARGIN:-0.01}}

for f in "$DEMOS" "$STAGE1_PLAIN"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_threading_margin.sh)"; exit 1; }
done
case $ARM in dynaibrl_dynens*)
    [ -f "$STAGE1_DYNENS" ] || { echo "MISSING asset: $STAGE1_DYNENS"; exit 1; } ;;
esac
case $ARM in
  ibrl_m001_qsrc|ibrl_m0_qsrc) NEED_BC=1
    [ -f "$IBRL_CFG" ] || { echo "MISSING $IBRL_CFG (python3 scripts/make_ibrl_m001_cfg.py)"; exit 1; }
    [ -f "$SRC_EXPERT" ] || { echo "MISSING source expert: $SRC_EXPERT"; exit 1; } ;;
  ibrl_*) NEED_BC=1
    [ -f "$IBRL_CFG" ] || { echo "MISSING $IBRL_CFG (python3 scripts/make_ibrl_m001_cfg.py)"; exit 1; } ;;
  *) NEED_BC=0 ;;
esac
if [ "$NEED_BC" = 1 ] && [ ! -f "$BC_PRIOR" ]; then
    echo "MISSING $BC_PRIOR -- run cluster/job_threading_bc_s160.sh first"; exit 1
fi

case $ARM in
  plain_h5)
    # collect_disagree_coef stays 0: geometric disagreement needs >1 dynamics head,
    # so the 0.5 the dynens arms use is undefined here. eval_disagree_coefs is left
    # at its "" default for the same reason.
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.0 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_plain)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $MARGIN_INIT --dyna.prior_margin_final $MARGIN_FINAL \
      --dyna.prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_plain_m001)
    # dynaibrl_plain with margin(t) CONSTANT at $CONST_MARGIN instead of the
    # 0.05 -> 0 anneal, shaping still off. Gives dynaibrl_plain_sgft the
    # single-variable partner it never had (see its comment), and separately asks
    # whether the constant is just the better schedule.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_plain_sgft)
    # ⚠️ Differs from dynaibrl_plain in TWO fields: sgft_coef AND the margin
    # schedule (constant CONST_MARGIN here, annealed there). The annealed version
    # of this arm performed badly, so the schedule was dropped for the shaped
    # arms. Consequence: this pair no longer isolates SGFT -- read the sgft arms
    # on their absolute score, not as a controlled ablation against the arm above.
    # To restore the single-variable pair, run with CONST_MARGIN unset and pass
    # --dyna.prior_margin_{init,final,steps} from MARGIN_* as the other arms do.
    #
    # Prior evidence on plain Dyna was a null (0.139 vs 0.124, n=10, ns), measured
    # under the MPPI-proposal metric on the pre-fix implementation, so it says
    # little here.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS --dyna.sgft_coef 1.0 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_dynens)
    # Same recipe as dynaibrl_plain, on the 5-head stage-1. This is what the
    # workstation runs used (dynaibrl_m005anneal*, steady ~0.88); the plain arms
    # in this batch are the deliberate single-head counterpart.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $MARGIN_INIT --dyna.prior_margin_final $MARGIN_FINAL \
      --dyna.prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_dynens_m001)
    # The 5-head counterpart of dynaibrl_plain_m001; single-variable partner of
    # dynaibrl_dynens_sgft.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dynaibrl_dynens_sgft)
    # ⚠️ Differs from dynaibrl_dynens in TWO fields -- see dynaibrl_plain_sgft.
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS --dyna.sgft_coef 1.0 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  ibrl_m005anneal0)
    # train_rl.py, NOT train_ibrl_simdist.py. ibrl_m005_seed1's cfg.yaml matches
    # train_rl.py on 55/55 top-level keys and train_ibrl_simdist.py on 34/55
    # (bc_policy, preload_*, freeze_bc_replay and the perturb fields live only in
    # train_rl.py) -- the model-free IBRL entry point is train_rl.py.
    #
    # Every flag below is one ibrl_m005_seed1 set away from train_rl.py's default,
    # so this reproduces that run except for: the demo standard (sat_160000, and
    # the matching s160 BC prior) and prior_margin_final 0.05 -> 0.0, which is the
    # anneal this batch is about. prior_margin_steps is train_rl.py's own default
    # of 100000, overridden here to $MARGIN_STEPS to match the dyna arms.
    # No world model anywhere in this arm.
    $PY train_rl.py --config_path $IBRL_CFG --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR $IBRL_ARCH \
      --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
      --num_eval_episode 100 --log_per_step 1000 \
      --prior_margin_init $MARGIN_INIT --prior_margin_final $MARGIN_FINAL \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  ibrl_m001)
    # ibrl_m005anneal0 with margin(t) CONSTANT at $CONST_MARGIN instead of the
    # 0.05 -> 0 anneal -- the model-free partner of dynaibrl_*_m001, so the whole
    # batch sits at one margin and "does the margin's value need a world model?"
    # is asked at the setting the dyna arms actually run at.
    #
    # ⚠️ Still TWO differences from any dyna arm, not one: no world model AND a
    # different prior (target-BC here, the frozen stage-2 distilled policy there).
    # It bounds how much of the effect needs a model; it does not isolate it.
    $PY train_rl.py --config_path $IBRL_CFG --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR $IBRL_ARCH \
      --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
      --num_eval_episode 100 --log_per_step 1000 \
      --prior_margin_init $CONST_MARGIN --prior_margin_final $CONST_MARGIN \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  ibrl_m0_qsrc)
    # Plain IBRL (NO margin at all) + the source expert's critic as Q's start.
    # Pairs with ibrl_m001_qsrc on one field (the margin) and asks the question
    # the margin exists to answer: the fixed-margin sweep put `margin 0 -> 0.000`
    # because an uncalibrated critic decides the argmax by noise (~5.8% action
    # deviation kills the episode). A critic warm-started from the source expert
    # is not uncalibrated -- it is calibrated to the WRONG domain. If the margin
    # was only ever a cold-start crutch, this arm should not collapse.
    #
    # ⚠️ That `margin 0 -> 0.000` datapoint predates the IBRL_ARCH fix above and
    # was a single seed, so it is context, not a baseline. The controlled
    # comparison is against ibrl_m001_qsrc in THIS batch.
    $PY train_rl.py --config_path $IBRL_CFG --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR $IBRL_ARCH \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
      --num_eval_episode 100 --log_per_step 1000 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  ibrl_m001_qsrc)
    # ibrl_m001 + the SOURCE expert's critic as Q's starting point. Differs from
    # ibrl_m001 in exactly ONE field (load_critic_only), so this is the controlled
    # test of "does warm-starting Q from the source expert help?".
    #
    # load_critic_only=1 (train_rl.py) keeps critic + critic_target from the
    # pretrained agent and RESTORES the freshly-initialised actor, so the RL actor
    # still cold-starts. Without it, load_pretrained_agent would also transfer the
    # actor and the arm would move two things at once -- which is what
    # threading_ibrl_simdist_srccritic_mppiactor_seed1 did (peak 0.97@170k but
    # oscillating, simdist.md 26); that run is NOT a clean precedent.
    #
    # Prior evidence is mixed and none of it is this measurement: tasks.md 41b has
    # the source expert critic LOSING to the distilled V-hat by 0.085 (z=2.8) as an
    # MPPI scorer, and the target-domain IQL-pretrained critic bought only speed,
    # not ceiling (~96%, matched). Both are different uses of a critic than IBRL's
    # argmax. Expect a null; the point is to stop guessing.
    $PY train_rl.py --config_path $IBRL_CFG --save_dir $OUT --seed $SEED \
      --bc_policy $BC_PRIOR $IBRL_ARCH \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
      --num_eval_episode 100 --log_per_step 1000 \
      --prior_margin_init $CONST_MARGIN --prior_margin_final $CONST_MARGIN \
      --prior_margin_steps $MARGIN_STEPS \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac
RC=$?
# NOT `rc=$?` after a $(date) -- that reports date's exit code and silently turns
# a truncated run into a green DONE.
echo "[$(date)] task $SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED rc=$RC"
grep -q "^100000: score/score" $OUT/train.log 2>/dev/null \
  || { echo "INCOMPLETE: $OUT never logged step 100000"; exit 1; }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
