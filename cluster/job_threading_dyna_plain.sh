#!/bin/bash
#SBATCH --job-name=thr_dsp
#SBATCH --array=1-30
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
# ^ --account=def-<pi> at submit time:
#       sbatch --account=def-<pi> cluster/job_threading_dyna_plain.sh
#
# Threading_D0, matched 100k-transition budget, into exps/simdist_threading_3arm
# so these land in the SAME wandb project as the existing 9 groups and overlay
# on one chart. Three arms x 10 seeds:
#
#   dyna        FAITHFUL Dyna/MBPO on the frozen dynens world model: plain RL on
#               real + imagined data, bc_coef = 0. No BC anchor, no source
#               anchor, no planner anywhere -- the policy acts alone.
#   dyna_sgft   the same + SGFT potential-based shaping (sgft_coef 1.0),
#               r' = r + gamma*V_hat(z') - V_hat(z) with V_hat FROZEN. Potential-
#               based, so the optimal policy is unchanged; it only densifies the
#               sparse 0/1 credit. Differs from `dyna` in exactly one field.
#   plain_h5    plain SimDist MPPI (SINGLE dynamics head) at horizon 5
#
# ⚠️ `dyna` here is NOT the old `dyna_h5`. Besides the demo standard (below) the
# implementation itself was corrected on 2026-08-31 -- see "four fixes" below.
#
# ----------------------------------------------------- four fixes (2026-08-31)
# The Dyna implementation had four defects; all are corrected, so these runs are
# NOT comparable to any pre-2026-08-31 dyna_* run.
#
# 1. `score/score` was the MPPI score. train_simdist_dyna.py set use_planner=0 in
#    __init__ but inherited the parent's _eval_mppi, which forces use_planner=1
#    for the duration of each eval. So the logged number was "current policy used
#    as an MPPI proposal" -- a quantity Dyna neither claims nor deploys. Measured
#    gap on a fixed smoke run: policy 0.25 against MPPI 0.65 at the same point.
#    Now the arm has no MPPI eval at all.
# 2. Real collection ran the policy at stddev=0, so every real transition sat at
#    exactly a=pi(s): the critic was pinned at ONE action per state and its shape
#    everywhere else was free extrapolation -- the direction DPG ascends. Now
#    `dyna.collect_std=0.1`. (The planning arms never had this: MPPI's CEM
#    sampling spreads their collected actions naturally.)
# 3. The synthetic buffer never wrapped (400/round x 100 rounds = 40k into a 100k
#    ring), so ~99% of the synthetic half of each late batch came from a policy
#    and dynamics head up to 100 rounds old.
# 4. imagine() decoded the latent it had just predicted and update() re-encoded
#    it. encode.decode is not the identity: measured on 4096 real target states,
#    ||encode(decode(z2)) - z2|| is 0.98x the one-step latent motion at stage 1
#    and 3.94x after 100k of adapt (the dynamics head trains while the DECODER
#    stays frozen at its stage-1 fit, so the corruption GROWS as the model
#    improves). Every synthetic next-state the critic saw was noise the size of
#    its signal. Both buffers now hold latents; nothing decodes.
#
# Two further corrections that are not bug fixes but change the numbers:
#   - `bc_coef` defaulted to 0.1. That is the TD3+BC regulariser, not part of
#     Dyna or MBPO, so the old "Dyna baseline" was really Dyna + a weak BC anchor
#     and bc_coef=0 had never been run. Default is now 0.
#   - `rollout_batch` was 400/round against 128,000 synthetic DRAWS per round
#     (1000 updates x 256 x (1-real_ratio)) -- a 320:1 deficit, and ~1000x below
#     MBPO's own rate (MBPO: 100k rollouts every 250 env steps = 400 per ENV
#     STEP). Now 4000/round with a 20k buffer, giving MBPO's ~6x reuse over data
#     at most 5 rounds old.
#
# ------------------------------------------------------- what the score means
# `score/score` is the POLICY score on all three arms in the sense each arm
# deploys: the Dyna arms have no planner, so it is the policy acting alone; the
# plain arm's deliverable IS the planner, so there it is the MPPI score, with
# `--eval_policy_only 1` logging the bare policy alongside as `score/policy_only`.
#
# ---------------------------------------------------------------- demo standard
# BOTH arms use the sat_160000 target demos (25 episodes each, one set per seed),
# i.e. demos collected from an UNSATURATED policy. That was the `_slow` suffix on
# the old arms; it is now the standing collection standard and is no longer a
# swept variable, so it is not in the run names. It is not free: docs/demo-
# collection.md 38 measured +0.139 from this switch alone, and the old
# dynens_h5 -> dynens_h5_slow pair (identical but for this path) moved round-0
# from 0.274 to 0.417.
#
# ==> Runs from this file are therefore NOT directly comparable to the workstation
#     dyna_bc10_seed{1,101}, which used sat_400000. Compare against dynens_h5_slow
#     / dynens_h5_slow_anchor, or re-run the old cells on sat_160000.
#
# ------------------------------------------------------------- the two stage-1s
# They are DIFFERENT checkpoints and that is the point of the plain arm:
#   dyna_bc10  -> threading_simdist_dynens_diversev2_h5_seed1  (5 dynamics heads)
#   plain_h5   -> threading_simdist_diversev2_h5_seed1         (1 head, same h=5,
#                                                               same diverse_v2)
# The existing plain_h3 arm is horizon 3, so plain-vs-dynens was confounded with
# horizon. This arm fixes that: identical source data, identical horizon, the
# only difference is the ensemble.
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
# afterwards. Override with WANDB_MODE=online sbatch --export=ALL,...,WANDB_MODE.
export WANDB_MODE=${WANDB_MODE:-offline}
# The key is NEVER hardcoded here -- this file is tracked by git. Three ways to
# supply it, checked in this order:
#   1. export it yourself:  WANDB_API_KEY=... sbatch --export=ALL,SIF,WANDB_API_KEY ...
#   2. a key file outside the repo:  chmod 600 ~/.config/wandb/key   (override
#      the path with WANDB_KEY_FILE=...)
#   3. `wandb login` once on a login node -- it writes ~/.netrc, and $HOME is
#      shared with the compute nodes and mounted inside the container by default.
# Note we do NOT pass the key via `apptainer --env` (that would put it in argv,
# where any user on the node can read it out of `ps`); APPTAINERENV_* forwards it
# through the environment instead.
WANDB_KEY_FILE=${WANDB_KEY_FILE:-$HOME/.config/wandb/key}
if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_KEY_FILE" ]; then
    WANDB_API_KEY=$(tr -d '[:space:]' < "$WANDB_KEY_FILE")
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
# online with no credentials at all dies inside wandb.init with
# "api_key not configured (no-tty)" and takes the whole task down with it, after
# the stage-1 ckpt is already loaded. Fall back rather than lose the run.
if [ "$WANDB_MODE" = "online" ] && [ -z "$WANDB_API_KEY" ] \
   && ! grep -qs 'api\.wandb\.ai' "$HOME/.netrc"; then
    echo "[wandb] no API key and no ~/.netrc entry -> falling back to WANDB_MODE=offline"
    echo "[wandb] sync afterwards from a login node:  wandb sync wandb/offline-run-*"
    export WANDB_MODE=offline
fi
USE_WB=${USE_WB:-1}     # USE_WB=0 -> train.log still records everything;
                        # replay later with scripts/wandb_import_logs.py

if [ -n "${SIF:-}" ]; then
    module purge; module load apptainer
    R=$(realpath "$PWD")
    MG_ASSETS=$R/deps/mimicgen/mimicgen/models/robosuite/assets
    MG_DEST=/usr/local/lib/python3.10/dist-packages/mimicgen/models/robosuite/assets
    MG_BIND=""
    [ -d "$MG_ASSETS" ] && MG_BIND="-B $MG_ASSETS:$MG_DEST"
    # Every --env below is load-bearing on fir; see job_square_baselines.sh for
    # the full diagnosis of each one (glibc/libGL, tensorboard/protobuf, CA bundle).
    PY="apptainer exec --nv -B $R:$R -B ${SLURM_TMPDIR:-/tmp}:/tmp $MG_BIND --pwd $R \
        --env PYTHONPATH=$R/.pylibs:$R --env MUJOCO_GL=$MUJOCO_GL --env WANDB_MODE=$WANDB_MODE \
        --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
        --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
        $SIF python"
else
    module purge; module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
    source ${VENV:-$HOME/venvs/ibrl}/bin/activate
    export PYTHONPATH=$PWD:$PYTHONPATH
    PY="python"
fi
echo "[env] $PY"

JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster/jobs_threading_dyna_plain.txt)
ARM=${JOB% *}; SEED=${JOB##* }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"

# save_dir's PARENT is the wandb project and the run name is the group
# (common_utils/py/run_cfg.py derives wb_exp/wb_run/wb_group from it as
# read-only properties -- --wb_* are NOT CLI args and argparse rejects them).
EXP_DIR=${EXP_DIR:-exps/simdist_threading_3arm}
SUFFIX=${SUFFIX:-}                 # e.g. SUFFIX=_s160 if a name would collide
OUT=$EXP_DIR/${ARM}${SUFFIX}_seed${SEED}
DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5

STAGE1_DYNENS=exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt
STAGE1_PLAIN=exps/mimicgen/threading_simdist_diversev2_h5_seed1/latest.pt

for f in "$DEMOS" "$STAGE1_DYNENS" "$STAGE1_PLAIN"; do
    [ -f "$f" ] || { echo "MISSING asset: $f  (run cluster/sync_threading_dyna_plain.sh)"; exit 1; }
done

# target_num_data 20 on BOTH arms. The sat_160000 sets hold 25 episodes, and the
# 3-arm comparison is defined on 20 -- the old plain_h3 left this null and so
# trained stage-2 on all 25, which is a second difference nobody wanted.
case $ARM in
  dyna)
    # Every knob at its (now faithful) default: bc_coef 0, no anchor, no planner.
    $PY train_simdist_dyna.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 0 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  dyna_sgft)
    # Differs from `dyna` in EXACTLY one field.
    $PY train_simdist_dyna.py --stage1_ckpt $STAGE1_DYNENS --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 0 \
      --dyna.sgft_coef 1.0 --use_wb $USE_WB --state_only_no_render 1 ;;
  plain_h5)
    # collect_disagree_coef stays 0: geometric disagreement needs >1 dynamics head,
    # so the 0.5 the dynens arms use is undefined here. eval_disagree_coefs is left
    # at its "" default for the same reason (a single head has nothing to sweep).
    $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_PLAIN --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.0 --original_mppi 1 \
      --use_wb $USE_WB --state_only_no_render 1 ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac
RC=$?
# NOT `rc=$?` after a $(date) -- that reports date's exit code and silently turns
# a truncated run into a green DONE. (Cost us a disk-full stop that went unnoticed.)
echo "[$(date)] task $SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED rc=$RC"
grep -q "^100000: score/score" $OUT/train.log 2>/dev/null \
  || { echo "INCOMPLETE: $OUT never logged step 100000"; exit 1; }
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
