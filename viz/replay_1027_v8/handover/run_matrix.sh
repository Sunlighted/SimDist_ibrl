#!/bin/bash
# one job: CKTAG STATE INHAND(sim|real) GPU
cd "$(dirname "$0")/../../.."; source set_env.sh >/dev/null 2>&1
declare -A CK=( [old]=exps/sim2real/threading_robotiq_pixel_stage1_seed1/model0.pt [v8m0]=exps/sim2real/threading_robotiq_v8_pixel_stage1_seed1/model0.pt [v8latest]=exps/sim2real/threading_robotiq_v8_pixel_stage1_seed1/latest.pt )
tag=$1; st=$2; ih=$3; g=$4; b=$(basename $st .npz); n=${b#demo_}
steps=250; [[ $b == *_t0 ]] && steps=400
flag=""; [ "$ih" = real ] && flag=--real_inhand
out=viz/replay_1027_v8/handover/runs/${tag}_${n}_${ih}
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g ${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python} \
  real2sim/zed_capture/calib_20261015/real_1027/tools/sim_handover.py --ckpt ${CK[$tag]} --state $st --out $out.json \
  --video $out.mp4 --steps $steps $flag > $out.log 2>&1
