#!/bin/bash
# Interactive smoke test for cluster/job_threading_margin.sh and its BC
# prerequisite. Runs EVERY command line from those two job files verbatim except
# for the budget knobs, which are knocked down to the minimum.
#
# The point is to prove the flags parse and the run reaches training inside the
# CONTAINER -- the three things that only bite there (mimicgen's missing XML
# assets, fir's CA bundle path, MUJOCO_GL) are all exercised. It is NOT a
# scientific run; every score it prints is meaningless.
#
# Usage, inside an salloc:
#   salloc --account=def-rhinehar --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 \
#          --cpus-per-task=12 --mem=24G --time=1:00:00
#   SIF=cluster/ibrl.sif bash cluster/smoke_threading_margin.sh
#
# Add ARMS="plain_h5 dyna" to test a subset. Each arm is capped by TIMEOUT
# (default 420s) -- hitting that cap is a PASS: it means the flags parsed and
# training started.
set -u
cd "$(dirname "$0")/.."

SEED=${SEED:-1}
TIMEOUT=${TIMEOUT:-420}
OUTROOT=${OUTROOT:-exps/_smoke_margin}
ARMS=${ARMS:-"bc_s160 plain_h5 dynaibrl_plain dynaibrl_plain_sgft ibrl_m005anneal0"}

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=offline          # a smoke test never touches the real project

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
echo "[env] MUJOCO_GL=$MUJOCO_GL  seed=$SEED  timeout=${TIMEOUT}s"

DEMOS=release/data/mimicgen/threading_d0_sat_160000_s${SEED}/dataset.hdf5
STAGE1_PLAIN=exps/mimicgen/threading_simdist_diversev2_h5_seed1/latest.pt
SRC_BC=exps/mimicgen/threading_source_mlp_bc_n2000_seed1/model0.pt
# The real ibrl arm uses il_ft_target_s160_seed*, which job_threading_bc_s160.sh
# builds. For the smoke test any existing prior proves the flags parse; fall back
# to the sat_400000 one if the s160 build has not run yet.
BC_PRIOR=exps/mimicgen/il_ft_target_s160_seed${SEED}/model0.pt
[ -f "$BC_PRIOR" ] || BC_PRIOR=exps/mimicgen/il_ft_target_seed${SEED}/model0.pt

MI=0.05; MF=0.0; MS=50000

echo "=== assets ==="
for f in "$DEMOS" "$STAGE1_PLAIN" "$SRC_BC" "$BC_PRIOR"; do
    [ -f "$f" ] && echo "  OK      $f" || echo "  MISSING $f"
done

mkdir -p $OUTROOT
PASS=0; FAIL=0

# A bad flag dies in seconds, during pyrallis parsing or the first asset load.
# Real training then runs until the timeout. So SURVIVAL TIME is the signal, not
# log scraping: robosuite prints "Got error: No module named 'robosuite_task_zoo'"
# on every import (a benign optional dependency) and mujoco's EGL context can
# raise from its own __del__ at interpreter shutdown, long after training worked.
# Both used to trip a keyword-matching check.
ALIVE_SEC=${ALIVE_SEC:-60}

run() {  # run <name> <cmd...>
    local name=$1; shift
    echo
    echo "======================================== $name"
    local t0=$SECONDS
    timeout $TIMEOUT $@ > $OUTROOT/$name.log 2>&1
    local rc=$? elapsed=$((SECONDS-t0))

    # Unambiguous flag errors, whenever they appear.
    if grep -qiE "unrecognized arguments|error: unrecognized|is not a valid|got an unexpected keyword" $OUTROOT/$name.log; then
        echo "  FAIL  the command line did not parse (${elapsed}s):"
        grep -iE "unrecognized arguments|error: unrecognized|is not a valid|got an unexpected keyword" \
            $OUTROOT/$name.log | head -3 | sed 's/^/        /'
        echo "        full log: $OUTROOT/$name.log"
        FAIL=$((FAIL+1)); return
    fi

    if [ $rc -eq 124 ]; then
        echo "  PASS  ran ${elapsed}s and was still training when the timeout fired"
        PASS=$((PASS+1))
    elif [ $rc -eq 0 ]; then
        echo "  PASS  completed in ${elapsed}s"
        PASS=$((PASS+1))
    elif [ $elapsed -lt $ALIVE_SEC ]; then
        echo "  FAIL  died after only ${elapsed}s (rc=$rc) -- parse or asset problem:"
        tail -12 $OUTROOT/$name.log | grep -vE "robosuite WARNING|ROBOMIMIC|macro file|task zoo|robosuite_task_zoo" \
            | tail -5 | sed 's/^/        /'
        echo "        full log: $OUTROOT/$name.log"
        FAIL=$((FAIL+1))
    elif grep -qE "in __del__|egl_context\.py" $OUTROOT/$name.log; then
        echo "  PASS  trained ${elapsed}s; the only traceback is a GL context teardown"
        echo "        at interpreter shutdown -- harmless, and MUJOCO_GL=osmesa"
        echo "        should prevent it entirely on the cluster."
        PASS=$((PASS+1))
    else
        echo "  FAIL  trained ${elapsed}s then crashed (rc=$rc):"
        tail -12 $OUTROOT/$name.log | grep -vE "robosuite WARNING|ROBOMIMIC|macro file|task zoo|robosuite_task_zoo" \
            | tail -5 | sed 's/^/        /'
        echo "        full log: $OUTROOT/$name.log"
        FAIL=$((FAIL+1))
    fi
}

for ARM in $ARMS; do
case $ARM in
  bc_s160)
    run bc_s160 $PY train_bc.py --save_dir $OUTROOT/bc --seed $SEED --task_name Threading_D0 \
      --dataset.path $DEMOS --dataset.num_data 20 --dataset.use_state 1 \
      --load_model $SRC_BC --num_epoch 1 --epoch_len 5 --lr 1e-4 --batch_size 256 \
      --grad_clip 5.0 --eval_period 1000 --num_eval_episode 2 \
      --eval_hand_init_perturb_joint_sigma 0.1 --use_wb 0 ;;
  plain_h5)
    run plain_h5 $PY train_simdist_adapt.py --stage1_ckpt $STAGE1_PLAIN \
      --save_dir $OUTROOT/plain --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 2000 --adapt_interval 1000 --eval_every_rounds 100 \
      --num_eval_episode 2 --num_eval_proc 1 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --collect_disagree_coef 0.0 --original_mppi 1 --use_wb 0 --state_only_no_render 1 ;;
  dynaibrl_plain)
    run dynaibrl_plain $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN \
      --save_dir $OUTROOT/dyna --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 2000 --adapt_interval 1000 --eval_every_rounds 100 \
      --num_eval_episode 2 --num_eval_proc 1 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $MI --dyna.prior_margin_final $MF \
      --dyna.prior_margin_steps $MS --use_wb 0 --state_only_no_render 1 ;;
  dynaibrl_plain_sgft)
    run dynaibrl_plain_sgft $PY train_simdist_dyna_ibrl.py --stage1_ckpt $STAGE1_PLAIN \
      --save_dir $OUTROOT/dynasgft --seed $SEED \
      --target_demo_path $DEMOS --target_num_data 20 \
      --max_transitions 2000 --adapt_interval 1000 --eval_every_rounds 100 \
      --num_eval_episode 2 --num_eval_proc 1 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
      --dyna.prior_margin_init $MI --dyna.prior_margin_final $MF \
      --dyna.prior_margin_steps $MS --dyna.sgft_coef 1.0 \
      --use_wb 0 --state_only_no_render 1 ;;
  ibrl_m005anneal0)
    run ibrl_m005anneal0 $PY train_rl.py --save_dir $OUTROOT/ibrl --seed $SEED \
      --task_name Threading_D0 --bc_policy $BC_PRIOR \
      --preload_datapath $DEMOS --preload_num_data 20 --freeze_bc_replay 1 \
      --num_train_step 300 --num_warm_up_episode 1 \
      --num_eval_episode 2 --mp_eval 1 --episode_length 400 --end_on_success 1 \
      --replay_buffer_size 1000 --mix_rl_rate 0.5 --num_critic_update 5 \
      --stddev_max 0.1 --use_state 1 --log_per_step 1000 \
      --train_hand_init_perturb_joint_sigma 0.1 \
      --eval_hand_init_perturb_joint_sigma 0.1 \
      --prior_margin_init $MI --prior_margin_final $MF --prior_margin_steps $MS \
      --use_wb 0 --state_only_no_render 1 ;;
  *) echo "unknown arm '$ARM'"; FAIL=$((FAIL+1)) ;;
esac
done

echo
echo "======================================== $PASS passed, $FAIL failed"
echo "logs in $OUTROOT/ -- delete it before the real run: rm -rf $OUTROOT"
[ $FAIL -eq 0 ]
