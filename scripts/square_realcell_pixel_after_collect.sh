#!/bin/bash
# Square pixel stage-1 data, step 2 of 2 (docs/infra.md §15i): once the 8
# agentview-only parts exist, add the CALIBRATED wrist camera by re-rendering
# from stored sim_state -- the same render-only pass Threading's
# 2cam_calibrated_v1 came from, with the same wrist calibration block.
#   setsid bash scripts/square_realcell_pixel_after_collect.sh > logs/square_realcell_pix_wrist.log 2>&1 < /dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
DEST=${DEST:-/user_data/Chenyang/ibrl_offload/release/data/mimicgen/square_realcell_pixel_v2}
EXPERT=${EXPERT:-exps/mimicgen/square_realcell_ibrl_seed1/model0.pt}
LIMIT=${LIMIT:-4500}
# Which GPUs the 8 re-render shards run on, one entry per shard. Point them at
# GPUs that are actually free: another tenant holding ~70 GB is what made 7 of
# 10 collection workers CUDA-OOM (§15n).
GPUS=${GPUS:-0,0,0,0,0,0,0,0}
# The collection chain logs to wait on (one per chain).
CHAIN_LOGS=${CHAIN_LOGS:-logs/square_pix_v2_gpu0.log}

stamp() { echo "[sq-pix $(date '+%F %T')] $*"; }
# Wait for BOTH chains to finish, not just for 8 files: a part*.hdf5 appears by
# mv at the end of a batch, but a salvaged (guard-killed) batch is short.
n_chains=$(echo $CHAIN_LOGS | wc -w)
until [ "$(grep -l 'chain on GPU' $CHAIN_LOGS 2>/dev/null | wc -l)" = "$n_chains" ]; do
  sleep 120
done
stamp "collection chains finished"
sha256sum -c --status "$DEST/cell_chain.sha256" || { stamp "ERROR: cell chain changed"; exit 1; }
# Every part must hold at least LIMIT episodes, or the re-render (which takes
# the first LIMIT) would silently read past a short shard. Open them -- file
# size is not evidence (the 49 GB unreadable 2cam/ lesson, §14e).
$PY - "$DEST" "$LIMIT" <<'PYEOF' || exit 1
import glob, h5py, os, sys
d, lim = sys.argv[1], int(sys.argv[2])
parts = sorted(glob.glob(os.path.join(d, "part*.hdf5")))
assert len(parts) == 8, f"expected 8 parts, found {len(parts)}"
for p in parts:
    with h5py.File(p, "r") as f:
        g = f["data"]; n = len(g)
        k = next(iter(g.keys()))
        assert "sim_state" in g[k] and "model_state" in g[k], p
        print(f"  {os.path.basename(p)}: {n} episodes")
        assert n >= lim, f"{p} has {n} < {lim}"
PYEOF
stamp "parts validated; re-rendering the calibrated wrist (limit $LIMIT/shard)"
$PY -u scripts/rerender_wrist_chunks.py --source "$DEST" --dest "$DEST/2cam_calibrated_v1" \
    --calibration "$DEST/wrist_calibration.json" --limit "$LIMIT" \
    --expert "$EXPERT" --gpus "$GPUS" || { stamp "ERROR: rerender failed"; exit 1; }
stamp "DONE -> $DEST/2cam_calibrated_v1"
