#!/bin/bash
#SBATCH --job-name=tpa_6arm
#SBATCH --array=1-40
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
# ^ --account=def-<pi> at submit time.
#
# ThreePieceAssembly_D0, target = kp0.25, 300k transitions, 8 arms x 5 seeds.
# Lands in its own wandb project (exps/tpa_kp025_6arm) -- save_dir's PARENT is
# the project and the run name is the group.
#
#   tpa_ibrl          model-free IBRL, no margin, cold critic
#   tpa_ibrl_qsrc     + the SOURCE expert's critic as Q's start   ("+srcval")
#   tpa_ibrl_qsrc_m001  + constant prior margin 0.01
#   tpa_dynaibrl      Dyna-IBRL on the SINGLE-head stage-1, no margin
#   tpa_dynaibrl_m001 + constant margin 0.01
#   tpa_dynaibrl_dynens_m001  the same on the 5-head stage-1      ("+dynens")
#   tpa_dynaibrl_sgft         tpa_dynaibrl_m001 + SGFT shaping (only sgft_coef differs)
#   tpa_iql           offline IQL warm start, act_method=rl, no prior
#   tpa_rlpd          50/50 sampling from a 2000-source + 20-demo prior buffer
#
# The three dyna arms read cumulatively: each adds one field to the one above.
# iql/rlpd are the model-free baselines from job_{square,coffee}_baselines.sh;
# both need assets built by scripts/prep_tpa_baselines.sh first.
#
# --------------------------------------------------------------- the target
# kp0.25 = base_hole_size_scale 0.96 / base_friction_scale 2.0 /
# gripper_kp_scale 0.25. This is TPA's DEFAULT target
# (release/cfgs/mimicgen/tpa_d0_state_ibrl_target.yaml) and is already the
# eval_domain_shift of BOTH stage-1 checkpoints, so the model-free and dyna arms
# run in the same env without any shift flags. (kp05 is a different, easier
# target -- tpa_d0_state_ibrl_target_kp05.yaml.) docs/infra.md 13 is the rule.
#
# ------------------------------------------------------- TPA's IBRL prior
# ⚠️ NOT --bc_policy. TPA's IBRL prior is the SOURCE EXPERT AGENT, attached via
# sgft_source_agent + source_agent_use_hybrid=1 (both already in the config).
# CLAUDE.md rule 8: load_frozen_agent pins act_method="rl" and drops the prior
# unless source_agent_use_hybrid=1 -- confirm with "[SGFT] prior = source HYBRID"
# in train.log. Threading/Square/Coffee use a target-finetuned BC instead; this
# is a real cross-task difference in what "the prior" means.
#
# ---------------------------------------------------------- eval cadence
# Every 10k transitions, 50 episodes -- NOT the 5k/100 the other three tasks use.
# TPA's episode_length is 1000 (2.5-6x the others), and a 100-episode eval
# measured 1447s on the adapt runs: 60 of those would be 23h of eval alone. At
# 10k/50 the dyna arms come to ~9h and the model-free arms ~5h, inside the wall.
# Dyna calls summary() every round either way, so both families still land on the
# same wandb x-axis (thousands of transitions).
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

JOBS=${JOBS:-cluster/jobs_tpa_6arm.txt}
[ -f "$JOBS" ] || { echo "no job list at $JOBS"; exit 1; }
JOB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
[ -n "$JOB" ] || { echo "task $SLURM_ARRAY_TASK_ID past the end of $JOBS"; exit 1; }
ARM=${JOB% *}; SEED=${JOB##* }
echo "[jobs] $JOBS"
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED"

EXP_DIR=${EXP_DIR:-exps/tpa_kp025_6arm}
OUT=$EXP_DIR/${ARM}${SUFFIX:-}_seed${SEED}

CFG=release/cfgs/mimicgen/tpa_d0_state_ibrl_target.yaml
DEMOS=release/data/mimicgen/tpa_d0_target_expert_n50/dataset.hdf5
SRC_EXPERT=exps/mimicgen/tpa_source_expert_frozen/model0.pt
STAGE1_PLAIN=exps/mimicgen/tpa_simdist_plain_diversev1_h5_seed1/latest.pt
STAGE1_DYNENS=exps/mimicgen/tpa_simdist_diversev1_h5_seed1/latest.pt
CONST_MARGIN=${CONST_MARGIN:-0.01}
MARGIN_STEPS=${MARGIN_STEPS:-50000}

for f in "$CFG" "$DEMOS" "$SRC_EXPERT"; do
    [ -f "$f" ] || { echo "MISSING asset: $f"; exit 1; }
done
case $ARM in
  *dynens*)   [ -f "$STAGE1_DYNENS" ] || { echo "MISSING $STAGE1_DYNENS"; exit 1; } ;;
  *dynaibrl*) [ -f "$STAGE1_PLAIN" ]  || { echo "MISSING $STAGE1_PLAIN"; exit 1; } ;;
  tpa_iql)    [ -f exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt ] || {
                  echo "MISSING IQL warm start (bash scripts/prep_tpa_baselines.sh iql)"; exit 1; } ;;
  tpa_rlpd)   [ -f release/data/mimicgen/tpa_rlpd_prior_s${SEED}/dataset.hdf5 ] || {
                  echo "MISSING RLPD prior for seed $SEED (bash scripts/prep_tpa_baselines.sh rlpd)"; exit 1; } ;;
esac

# Budget overrides shared by the three model-free arms. The config itself carries
# the kp0.25 target, act_method=ibrl, the 5x1024 critic, the source-expert prior
# and episode_length 1000; only the budget and the demo set change here.
MF_COMMON="--config_path $CFG --save_dir $OUT --seed $SEED \
  --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
  --num_train_step 300000 --num_warm_up_episode 30 --count_warmup_transitions 1 \
  --num_eval_episode 50 --log_per_step 10000 \
  --use_wb $USE_WB --state_only_no_render 1"

# Shared by the three dyna arms.
DY_COMMON="--save_dir $OUT --seed $SEED \
  --target_demo_path $DEMOS --target_num_data 20 \
  --max_transitions 300000 --adapt_interval 1000 --eval_every_rounds 10 \
  --num_eval_episode 50 --num_eval_proc 10 --eval_policy_only 1 \
  --use_wb $USE_WB --state_only_no_render 1"

case $ARM in
  tpa_ibrl)
    $PY train_rl.py $MF_COMMON \
      --prior_margin_init 0.0 --prior_margin_final 0.0 ;;
  tpa_ibrl_qsrc)
    # load_critic_only: keep critic + critic_target from the source expert and
    # restore the freshly-initialised actor, so "+srcval" moves exactly one thing.
    # ⚠️ The older tpa_ibrl_tgt20_qsrc_seed1 used load_policy_only=0, which
    # transfers the ACTOR too -- that run (0.84 @1.5M vs cold 0.32) is therefore
    # not this arm.
    $PY train_rl.py $MF_COMMON \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 ;;
  tpa_ibrl_qsrc_m001)
    $PY train_rl.py $MF_COMMON \
      --load_pretrained_agent $SRC_EXPERT --load_policy_only 0 --load_critic_only 1 \
      --prior_margin_init $CONST_MARGIN --prior_margin_final $CONST_MARGIN \
      --prior_margin_steps $MARGIN_STEPS ;;
  tpa_iql)
    # act_method MUST be rl: IQL has no prior, and act_method=ibrl trips
    # `assert len(self.bc_policies) == 1` in _act_ibrl. The config's
    # sgft_source_agent is cleared for the same reason -- it IS a prior.
    # The warm start is one checkpoint shared by every seed
    # (scripts/prep_tpa_baselines.sh), remapped by iql_ckpt_to_agent.py because
    # --load_pretrained_agent keeps only already-matching keys and the raw
    # {"critic":..,"actor":..} nest would warm-start nothing.
    $PY train_rl.py $MF_COMMON \
      --q_agent.act_method rl --q_agent.bootstrap_method rl \
      --sgft_source_agent none \
      --load_pretrained_agent exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt \
      --load_policy_only 0 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 ;;
  tpa_rlpd)
    # prior buffer = $RLPD_SRC_EPS source rollouts + this seed's 20 target demos;
    # mix_rl_rate 0.5 (in the config) is what makes the sampling 50/50. Overrides
    # MF_COMMON's preload, so --preload_* appear twice and argparse takes the last.
    $PY train_rl.py $MF_COMMON \
      --q_agent.act_method rl --q_agent.bootstrap_method rl \
      --sgft_source_agent none \
      --preload_datapath release/data/mimicgen/tpa_rlpd_prior_s${SEED}/dataset.hdf5 \
      --preload_num_data 2020 --replay_buffer_size 4000 \
      --prior_margin_init 0.0 --prior_margin_final 0.0 ;;
  tpa_dynaibrl)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $DY_COMMON \
      --dyna.prior_margin_init 0.0 --dyna.prior_margin_final 0.0 \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  tpa_dynaibrl_m001)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $DY_COMMON \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  tpa_dynaibrl_dynens_m001)
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_DYNENS $DY_COMMON \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS ;;
  tpa_dynaibrl_sgft|tpa_dynaibrl_sgft_qreinit)
    # tpa_dynaibrl_m001 + SGFT potential shaping, ONE field changed
    # (--dyna.sgft_coef 1.0): same single-head stage-1, same constant margin 0.01,
    # same 300k. A single-variable pair, unlike Threading's dynaibrl_plain_sgft,
    # which also swapped the annealed schedule for a constant and so stopped
    # isolating SGFT.
    #
    # Shaping rewrites Q'(z,a) = Q(z,a) - coef*V_hat(z), and the code moves BOTH
    # the clamp bounds and the Q warm start with it (rl/simdist_dyna_ibrl.py:286,
    # 393) so step 0 stays a fixed point of the SHAPED backup. Getting either
    # wrong silently truncates the shaped Q.
    #
    # ⚠️ Prior evidence is a null, and a weak one: on plain Dyna sgft_coef=1.0
    # scored 0.139 vs 0.124 (n=10, +0.015, ns) -- under the MPPI-proposal metric
    # on the pre-fix implementation, so it transfers nothing to this arm. On
    # Threading the shaping "could not produce contact skills" and what was
    # missing was reaching, not credit assignment (memory
    # project_sgft_shaping_threading). TPA is the task where stage-3 is dominated
    # by rollout compounding, so a potential that reshapes credit over long
    # horizons is at least mechanistically plausible here.
    QREINIT_ARGS=()
    case $ARM in *_qreinit) QREINIT_ARGS=(--dyna.q_init_from_value 0) ;; esac
    $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN $DY_COMMON \
      --dyna.sgft_coef ${SGFT_COEF:-1.0} \
      --dyna.prior_margin_init $CONST_MARGIN --dyna.prior_margin_final $CONST_MARGIN \
      --dyna.prior_margin_steps $MARGIN_STEPS "${QREINIT_ARGS[@]}" ;;
  *) echo "unknown arm '$ARM'"; exit 1 ;;
esac
RC=$?
echo "[$(date)] task $SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED rc=$RC"
[ "$RC" = 0 ] || exit $RC
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE"
