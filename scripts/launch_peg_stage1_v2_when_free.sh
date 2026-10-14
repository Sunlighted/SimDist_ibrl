#!/bin/bash
# Wait until GPU 0 or GPU 1 is released by the running Threading stage-1 jobs, then
# launch the peg photoreal SimDist stage-1 (release/cfgs/mimicgen/peg_d0_pixel_simdist_r18_224_dynens_v2.yaml)
# on the GPU that freed up.
#
# Gates before launch (each one has bitten a run before):
#   - the watched PID has exited AND that GPU's memory is back under FREE_GPU_MB
#   - host RAM available >= NEED_RAM_GB (the seq buffer holds 3.26M x 39 KB ~= 127 GB
#     resident, and each Threading stage-1 holds 114 GB until it exits)
#
#   setsid bash scripts/launch_peg_stage1_v2_when_free.sh > logs/peg_stage1_v2_launcher.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
# PID -> GPU index of the jobs we wait for (measured 2026-09-25 with nvidia-smi)
declare -A WATCH=( [3301500]=0 [3301977]=1 )
NEED_RAM_GB="${NEED_RAM_GB:-150}"
FREE_GPU_MB="${FREE_GPU_MB:-8000}"
CFG=release/cfgs/mimicgen/peg_d0_pixel_simdist_r18_224_dynens_v2.yaml
SAVE=exps/mimicgen/peg_pixel_stage1_v2_seed1
stamp() { echo "=== [$(date '+%F %T')] $* ==="; }

stamp "waiting for one of: $(for p in "${!WATCH[@]}"; do echo -n "pid $p (gpu ${WATCH[$p]}) "; done)"
GPU=""
while [ -z "$GPU" ]; do
    for p in "${!WATCH[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
            g=${WATCH[$p]}
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -dc 0-9)
            if [ "$used" -lt "$FREE_GPU_MB" ]; then GPU=$g; stamp "pid $p exited; gpu $g now uses ${used} MiB"; break; fi
        fi
    done
    [ -z "$GPU" ] && sleep 120
done

while :; do
    avail=$(free -g | awk '/^Mem:/{print $7}')
    [ "$avail" -ge "$NEED_RAM_GB" ] && break
    stamp "RAM available ${avail} GB < ${NEED_RAM_GB} GB, waiting"; sleep 120
done
stamp "launching peg stage-1 v2 on gpu $GPU (RAM available ${avail} GB)"

source set_env.sh
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
export CUDA_HOME=/usr/local/cuda-12.8 PATH=$(dirname $PY):/usr/local/cuda-12.8/bin:$PATH
export WANDB_MODE=offline     # online wandb has deadlocked long runs (memory: gotcha-wandb-online-deadlock)
$PY -u train_simdist.py --config_path "$CFG" \
    --source_agent exps/mimicgen/peg_ibrl_source_expert_board_hard_preload1000_seed1/model0.pt \
    --save_dir "$SAVE" --use_wb 0
stamp "train_simdist exited rc=$?"
grep -E "^\[eval @" "$SAVE/train.log" | tail -5
