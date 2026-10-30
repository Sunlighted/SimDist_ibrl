#!/bin/bash
# usage: kin3.sh OUTDIR "dx_mm,dy_mm,dyaw_deg" [extra replay_kin.py args, e.g. --mode track --no_log --tripod_mass_scale 3]
# Replays the 6 converted 10-27 demos in the 10-30 cell with the wrist-refit layout + a tripod offset (README "10-27 data in the v8 sim").
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO"; source set_env.sh >/dev/null 2>&1
G=${GPU:-3}
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G PYTHONUNBUFFERED=1 \
  THREADING_ROBOTIQ_CELL_YAML=real2sim/zed_capture/calib_20261015/threading_real_cell_20261030.yaml
OUT=$1; OFF=$2; shift 2
mkdir -p "$OUT"
${PY:-/home/Chenyang/.conda/envs/ibrl/bin/python} "$HERE/replay_kin.py" \
  --dataset /user_data/Chenyang/real_stage2/threading_robotiq_1027_c1030/dataset.hdf5 --cell $THREADING_ROBOTIQ_CELL_YAML \
  --mode kin --grip action --gripper_cmd_delay 1 2 \
  --layout_json real2sim/zed_capture/calib_20261015/demo_layout/layout_demos_1027_wristrefit.json \
  --video "$OUT" --tripod_offset=$OFF "$@" > "$OUT/run.log" 2>&1
echo DONE rc=$? >> "$OUT/run.log"
