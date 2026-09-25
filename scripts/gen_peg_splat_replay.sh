#!/bin/bash
# PHOTOREAL stage-1 source replay for PegInsertion_D0, one shard per call (local,
# no SLURM). Recipe = release/cfgs/mimicgen/peg_d0_pixel_simdist_r18_224_dynens.yaml
# (EXPERT_HYBRID=1, sigma 0.05-0.2, p_expert 0.5, board +-10 mm/+-5 deg, zedview +
# wrist @224, JPEG q95) with the 2026-09-24 task (stiff walls, 30 mm, tilt <= 5 deg)
# and the fixed board splat (docs/real2sim.md §5a3). --store_sim_state 1 so the
# frames can be re-rendered later without re-simulating.
#
# Expert = IBRL arm A (exps/.../peg_ibrl_source_expert_board_hard_preload1000_seed1,
# independent n=100 = 0.96) + its 16 snapshots; prior = the gated MLP BC.
#
#   # one-time setup on a new machine: see the PRE-FLIGHT block below
#   BUNDLE=/user_data/Chenyang/peg_gen_bundle GPU=0 SHARDS="0 1 2 3" NUM_PROC=4 \
#     setsid bash scripts/gen_peg_splat_replay.sh > logs/gen_peg_splat_gpu0.log 2>&1 < /dev/null & disown
#   ... GPU=1 SHARDS="4 5 6 7" ...  (a second GPU, in parallel)
#   DRY_RUN=1 ... runs only the pre-flight checks. PY=<env python> if `python` is not on PATH.
#
# Sizing (measured 2026-09-24, 40-traj calibration): mean length 230, 50% success,
# 40.1 KB/transition, ~11 GB GPU per worker. Training caps the buffer at 3.3M
# transitions => 8 shards x 1800 traj ~= 3.3M ~= 133 GB. Re-derive
# source_replay_num_data from the REAL shards afterwards (num_data = cap/mean_len/n_shards).
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
PY="${PY:-$(command -v python)}"
export MUJOCO_GL=egl
# *_OVERRIDE: hosts whose EGL device order != nvidia-smi order (shannon: EGL id must
# come from a map, and robosuite asserts the EGL id is a substring of CVD)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE:-${GPU:?set GPU}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID_OVERRIDE:-$GPU}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# ninja (pip-installed next to python) must be on PATH or gsplat's JIT build fails
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"

BUNDLE="${BUNDLE:-/user_data/Chenyang/peg_gen_bundle}"
SRC="${SRC:-$BUNDLE/exps/mimicgen/peg_ibrl_source_expert_board_hard_preload1000_seed1}"
BC="${BC:-$BUNDLE/exps/mimicgen/peg_source_mlp_bc_expert_n1000_board_hard_seed1/model0.pt}"
OUT_ROOT="${OUT_ROOT:-/user_data/Chenyang/simdist_data/peg_d0_src_splat224_v2}"
SHARDS="${SHARDS:-0}"
NUM_TRAJ="${NUM_TRAJ:-1800}"        # per shard
NUM_PROC="${NUM_PROC:-4}"           # ~11 GB GPU + ~6 GB RAM + 1 core each
MIN_FREE_GB="${MIN_FREE_GB:-30}"    # stop before the shared /user_data fills up
GB_PER_SHARD=$($PY -c "print(int($NUM_TRAJ * 230 * 40.1e3 / 1e9) + 2)")

stamp() { echo "=== [$(date '+%F %T')] $* ==="; }
die() { echo "FATAL: $*"; exit 1; }
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc 0-9; }

# ---------------------------------------------------------------- PRE-FLIGHT
stamp "pre-flight (gpu=$GPU shards=[$SHARDS] traj/shard=$NUM_TRAJ proc=$NUM_PROC)"
PLY=real2sim/splats/scene_scan5_norobot_v2.ply
[ "$(stat -c %s "$PLY")" -gt 100000000 ] || die "$PLY is a git-lfs POINTER ($(stat -c %s "$PLY") B): run 'git lfs pull --include=$PLY'"
[ -f deps/GaussianRenderer/src/gaussian_renderer/core/util_gau.py ] || die "submodule missing: git submodule update --init deps/GaussianRenderer"
command -v ninja >/dev/null || die "ninja not on PATH (pip install --no-deps ninja)"
[ -s "$SRC/model0.pt" ] && [ -s "$SRC/cfg.yaml" ] || die "expert missing: $SRC"
[ "$(ls "$SRC"/snapshot_*.pt 2>/dev/null | wc -l)" -ge 16 ] || die "expected 16 snapshots in $SRC"
[ -s "$BC" ] && [ -s "$(dirname "$BC")/cfg.yaml" ] || die "BC prior missing: $BC"
$PY - <<'EOF' || die "python pre-flight failed (see above)"
import numpy, torch, gsplat
from gsplat.cuda._backend import _C          # builds/loads the CUDA extension ONCE, before workers fork
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__}: need 1.x (CLAUDE.md #35)"
import robosuite
from robosuite import load_controller_config
import env.peg_insertion
import real2sim.splat_obs as so
e = robosuite.make("PegInsertion_D0", robots="Panda", has_offscreen_renderer=False, use_camera_obs=False,
                   controller_configs=load_controller_config(default_controller="OSC_POSE"))
so._check_board_calib(e.sim.model, so.DEFAULT_CALIB)      # the 62-64 mm board-splat bug guard
print(f"[pre-flight] OK numpy {numpy.__version__} torch {torch.__version__} gsplat {gsplat.__version__} "
      f"insert_depth {e.insert_depth} tilt {e.success_max_tilt}")
EOF
mkdir -p "$OUT_ROOT"
NEED=$((GB_PER_SHARD * $(echo $SHARDS | wc -w)))
FREE=$(free_gb "$OUT_ROOT")
echo "disk: need ~${NEED} GB for these shards, free ${FREE} GB on $(df --output=target "$OUT_ROOT" | tail -1)"
[ "$FREE" -gt $((NEED + MIN_FREE_GB)) ] || die "not enough space (need ${NEED}+${MIN_FREE_GB} GB)"
[ "${DRY_RUN:-0}" = 1 ] && { stamp "DRY_RUN: pre-flight passed, not generating"; exit 0; }

# ---------------------------------------------------------------- SHARDS
for s in $SHARDS; do
    OUT="$OUT_ROOT/shard_$s"
    if [ -s "$OUT/dataset.hdf5" ]; then stamp "shard $s already done, skip"; continue; fi
    [ "$(free_gb "$OUT_ROOT")" -gt $((GB_PER_SHARD + MIN_FREE_GB)) ] || die "free space below ${MIN_FREE_GB} GB margin before shard $s"
    stamp "shard $s -> $OUT (seed $((1000 + s)))"
    $PY scripts/generate_simdist_data.py \
        --expert "$SRC/model0.pt" --checkpoints_glob "$SRC/snapshot_*.pt" \
        --expert_bc_policy "$BC" \
        --out_dir "$OUT" --num_traj "$NUM_TRAJ" --num_proc "$NUM_PROC" \
        --p_expert 0.5 --sigma_min 0.05 --sigma_max 0.2 --hand_init_sigma 0.1 \
        --domain source --expert_hybrid 1 --snapshot_hybrid 1 --save_values 1 \
        --splat 1 --board_pos_noise 0.01 --board_yaw_noise 5 \
        --rl_cameras "zedview+robot0_zedm_wrist" --image_size 224 --rl_image_size 224 \
        --jpeg_quality 95 --store_sim_state 1 --flush_mb 2000 --flush_every 500 \
        --seed $((1000 + s)) &
    GEN=$!
    # watchdog: kill the shard if the shared partition gets close to full
    while kill -0 $GEN 2>/dev/null; do
        if [ "$(free_gb "$OUT_ROOT")" -lt "$MIN_FREE_GB" ]; then
            echo "FATAL: free space < ${MIN_FREE_GB} GB during shard $s -- killing generator"
            pkill -P $GEN; kill $GEN; exit 1
        fi
        sleep 60
    done
    wait $GEN; rc=$?
    [ $rc -eq 0 ] && [ -s "$OUT/dataset.hdf5" ] || die "shard $s failed (rc=$rc)"
    $PY -c "
import h5py, numpy as np, os
f = h5py.File('$OUT/dataset.hdf5'); d = f['data']
L = np.array([d[k]['actions'].shape[0] for k in d]); S = np.mean([np.asarray(d[k]['rewards']).max() >= 1 for k in d])
gb = os.path.getsize('$OUT/dataset.hdf5') / 1e9
print(f'shard $s: {len(L)} traj, mean len {L.mean():.1f}, success {S:.3f}, {gb:.1f} GB, {gb*1e9/L.sum():.0f} B/transition')"
done
stamp "done"
