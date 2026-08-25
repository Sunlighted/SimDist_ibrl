#!/bin/bash
#SBATCH --job-name=sq_pack
#SBATCH --array=1-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/logs/%x_%A_%a.out
# ^ --account=def-<pi> must be supplied at submit time:
#       sbatch --account=def-<pi> cluster/job_square_baselines.sh
#
# PACKED variant of job_square_baselines.sh: TWO seeds per array task, run
# CONCURRENTLY on the one GPU the task holds. 21 jobs -> 11 array tasks, 20 runs
# in flight once 10 tasks are scheduled.
#
# Why this is safe here (measured, not assumed): with --state_only_no_render 1 a
# run uses only 1122 MiB of VRAM and puts exactly ONE process on the GPU.
# run_eval_mp is central-inference -- the 10 eval workers run MuJoCo on the CPU
# and ship observations back through a queue; only the main process touches the
# GPU. (The workers used to appear on the GPU solely because each built an EGL
# render context; --state_only_no_render removes that.) So 2 runs need ~2.2 GB
# and 2 CUDA contexts -- comfortable inside a 1g.10gb MIG slice.
#
# Resources are therefore driven by CPU and host RAM, not VRAM:
#   cpus 24  = 2 x (10 eval workers + 1 main)
#   mem  48G = 2 x 17 GB measured peak (peak is during eval), plus headroom
#   time 24h = 2 x the single-run budget; two runs share one GPU, and measured
#              throughput per GPU is flat, so packing does NOT speed a run up --
#              it buys job-slot efficiency, not wall clock.
#
# Square_D0 rotw1 TARGET-B: the model-free + Dyna baselines, mirroring
# exps/simdist_threading_3arm/{ibrl,iql,rlpd,dyna_h5,dyna_sgft}_seed*.
#
# Array index -> one line of cluster/jobs_square_baselines.txt ("<arm> <seed>").
# The default 21 tasks reproduce Threading's seed counts exactly:
#   ibrl / iql / rlpd  x 5 seeds (1,101,201,301,401)
#   dyna_h5 / dyna_sgft x 3 seeds (1,101,201)
# Bump to 10 seeds each by regenerating the job list (see the README) and
# changing --array.
#
# Resources: 12 CPUs because train_rl.py's mp_eval hard-codes num_proc=10
# (train_rl.py:565) -- fewer cores and every eval serialises. 32G covers the 10
# MuJoCo workers plus torch.
set -e
cd $SLURM_SUBMIT_DIR
mkdir -p cluster/logs

# Two ways to get the env; set SIF to use the container, otherwise the venv.
#   SIF=cluster/ibrl.sif sbatch --account=def-<pi> cluster/job_square_baselines.sh
# MIG-partitioned H100s (Compute Canada fir) cannot initialise an EGL device
# display at all -- `import mujoco` with MUJOCO_GL=egl dies at IMPORT time with
# "your EGL driver does not support the PLATFORM_DEVICE extension", before any of
# our code runs. osmesa lets the import succeed; combined with
# --state_only_no_render 1 below no GL context is ever created, so osmesa never
# actually renders anything. Verified: the two paths produce BIT-IDENTICAL
# trajectories (max abs diff 0.000e+00 over 60 steps).
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export WANDB_MODE=${WANDB_MODE:-offline}   # compute nodes have no internet;
                                           # `wandb sync wandb/offline-run-*` later
if [ -n "${SIF:-}" ]; then
    module purge; module load apptainer
    # --nv injects the host NVIDIA driver (needed for CUDA *and* for EGL).
    # -B $PWD and --pwd keep repo-relative paths in the cfgs valid; $SLURM_TMPDIR
    # is bound because robosuite/mujoco write caches there.
    # LD_LIBRARY_PATH is LOAD-BEARING on clusters whose host glibc is newer than
    # the image's. `--nv` injects the host GPU stack into /.singularity.d/libs,
    # and on fir the injected libGL.so.1 needs GLIBC_ABI_DT_RELR (glibc >= 2.36)
    # while the ubuntu 22.04 image has 2.35 -- so even `import cv2` dies with
    # "version `GLIBC_ABI_DT_RELR' not found". Putting the container's own libdir
    # FIRST makes its libGL win, while the NVIDIA vendor libs still resolve from
    # the injected dir. Verified on fir.
    # .pylibs shadows the image's tensorboard 2.21.0 with 2.20.0. robomimic
    # requires tensorboard with no upper bound, so the image got the newest, whose
    # summary_pb2.py is protoc-6-generated and needs google.protobuf.runtime_version
    # (protobuf >= 5.27) -- the image has 4.25.9, pinned there because wandb 0.16.4
    # requires protobuf<5. wandb.init() starts a TBWatcher that lazily imports it,
    # so every run died with "MailboxError: transport failed" at init. The
    # workstation has tensorboard 2.20.0, which is protoc-4-generated and fine.
    # Populate it once on a LOGIN node (compute nodes have no internet):
    #   apptainer exec -B $PWD:$PWD cluster/ibrl.sif \
    #     python -m pip install --no-deps --target=$PWD/.pylibs "tensorboard==2.20.0"
    # --no-deps is load-bearing: without it pip also drops numpy 2.x and protobuf
    # 7.x into the shadow dir, which then shadow the image's correct versions and
    # `import wandb` dies on the removed np.float_.
    R=$(realpath "$PWD")          # CC home/scratch paths are symlinks; bind the real one
    PY="apptainer exec --nv -B $R:$R -B ${SLURM_TMPDIR:-/tmp}:/tmp --pwd $R \
        --env PYTHONPATH=$R/.pylibs:$R --env MUJOCO_GL=$MUJOCO_GL --env WANDB_MODE=$WANDB_MODE \
        --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs \
        $SIF python"
else
    module purge; module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
    source ${VENV:-$HOME/venvs/ibrl}/bin/activate
    export PYTHONPATH=$PWD:$PYTHONPATH
    PY="python"
fi
echo "[env] $PY"

# Two lines of the job list per array task: 2i-1 and 2i.
L1=$(( 2 * SLURM_ARRAY_TASK_ID - 1 )); L2=$(( 2 * SLURM_ARRAY_TASK_ID ))
JOB1=$(sed -n "${L1}p" cluster/jobs_square_baselines.txt)
JOB2=$(sed -n "${L2}p" cluster/jobs_square_baselines.txt)   # empty on the last, odd task
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> lines $L1,$L2 = '$JOB1' '$JOB2'"

BASE=release/cfgs/mimicgen/square_d0_state_baseline_target_rotw1.yaml
STAGE1=exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/latest.pt

run_one() {
  local ARM=$1 SEED=$2
  [ -z "$ARM" ] && return 0
  local OUT=exps/square_rotw1_baselines/${ARM}_seed${SEED}
  local DEMOS=release/data/mimicgen/square_sat_600000_s$SEED/dataset.hdf5
  echo "[$(date +%H:%M)] start $ARM seed$SEED"
case $ARM in
  ibrl)
    # Prior = the source MLP BC, UNMODIFIED and shared across seeds. Threading's
    # recipe fine-tunes it on the seed's 20 target demos; measured on Square that
    # drops the prior 0.26 -> 0.00 at every fine-tune length tried (and
    # from-scratch on the demos is 0.00 too). See scripts/prep_square_baselines.sh
    # for the full table. This is a real cross-task difference in the baseline's
    # prior definition and must be stated when comparing the two IBRL arms.
    $PY train_rl.py --config_path $BASE --seed $SEED --save_dir $OUT \
      --bc_policy exps/mimicgen/square_source_mlp_bc_n1000_seed1/model0.pt \
      --preload_datapath $DEMOS --preload_num_data 20 \
      --num_warm_up_episode 10 --replay_buffer_size 1000 --state_only_no_render 1 ;;
  iql)
    # act_method MUST be rl: IQL has no BC prior and act_method=ibrl trips
    # `assert len(self.bc_policies) == 1`.
    $PY train_rl.py --config_path $BASE --seed $SEED --save_dir $OUT \
      --q_agent.act_method rl --q_agent.bootstrap_method rl \
      --load_pretrained_agent exps/mimicgen/square_iql_offline_source/agent_warmstart.pt \
      --preload_datapath $DEMOS --preload_num_data 20 \
      --num_warm_up_episode 10 --replay_buffer_size 1000 --state_only_no_render 1 ;;
  rlpd)
    # prior buffer = 2000 source rollouts + this seed's 20 target demos, and
    # mix_rl_rate 0.5 (in $BASE) is what makes the sampling 50/50.
    $PY train_rl.py --config_path $BASE --seed $SEED --save_dir $OUT \
      --q_agent.act_method rl --q_agent.bootstrap_method rl \
      --preload_datapath release/data/mimicgen/square_rlpd_prior_s${SEED}/dataset.hdf5 \
      --preload_num_data 2020 \
      --num_warm_up_episode 10 --replay_buffer_size 4000 --state_only_no_render 1 ;;
  dyna|dyna_h5)
    $PY train_simdist_dyna.py --stage1_ckpt $STAGE1 --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 0 --use_wb 1 \\
      --state_only_no_render 1 ;;
  dyna_sgft)
    $PY train_simdist_dyna.py --stage1_ckpt $STAGE1 --save_dir $OUT --seed $SEED \
      --target_demo_path $DEMOS \
      --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
      --num_eval_episode 100 --num_eval_proc 10 \
      --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 0 --use_wb 1 \
      --dyna.sgft_coef 1.0 --state_only_no_render 1 ;;
  *) echo "unknown arm '$ARM'"; return 1 ;;
esac
  echo "[$(date +%H:%M)] done $ARM seed$SEED"
}

run_one ${JOB1% *} ${JOB1##* } &
P1=$!
sleep 60                      # stagger: keep the two evals from colliding on start
run_one ${JOB2% *} ${JOB2##* } &
P2=$!
wait $P1; R1=$?
wait $P2; R2=$?
echo "[$(date)] task $SLURM_ARRAY_TASK_ID DONE (exit $R1 / $R2)"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ]
