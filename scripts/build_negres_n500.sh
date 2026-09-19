#!/bin/bash
# Build the N=500 target demo set for ONE seed of cluster/job_threading_negres.sh
# (demo500 arm):
#
#   threading_d0_sat_160000_n500_s<SEED> = sat_160000_s<SEED> (its 25 demos, as
#                                          demo_0..24) + 475 NEW demos (demo_25..499)
#
# Why not simply `collect --num_demo 500 --seed SEED`: collect_target_demos.py uses
# episode seed = args.seed + attempt, and 500 accepts need ~500-2000 attempts, so
# seed 1's set would run through env seeds 101, 201, ... and contain, demo for
# demo, the SAME episodes as the other seeds' sets (policy and reset are both
# deterministic). The 5 "independent" draws would share most of their data.
# The new 475 therefore come from a disjoint env-seed block
# 1_000_000 + SEED*10_000 (10k >= the 2000-attempt cap).
#
# Keeping the original 25 in front makes the arms NESTED within a seed:
# --target_num_data 20 reads demo_0..19 (load_demos_to_sequence_buffer walks
# demo_{i} in index order), i.e. the plain_h5 draw is a subset of the demo500 draw.
#
# Collector and every shift knob are those of scripts/rebuild_demos.sh
# `threading160` (iron law #13: pass them all explicitly).
#
#   bash scripts/build_negres_n500.sh <seed> [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-} MUJOCO_GL=osmesa
SEED=$1; GPU=${2:-0}
CK=exps/mimicgen/threading_ibrl_newtgt_sig01_snap_seed1/snapshot_160000.pt
BASE=release/data/mimicgen/threading_d0_sat_160000_s$SEED
EXTRA=release/data/mimicgen/_negres_n475_s$SEED
OUT=release/data/mimicgen/threading_d0_sat_160000_n500_s$SEED
NEW=475
SEED0=$((1000000 + SEED * 10000))

[ -f "$CK" ] || { echo "MISSING ckpt $CK"; exit 1; }
[ -f "$BASE/dataset.hdf5" ] || { echo "MISSING $BASE"; exit 1; }
[ -f "$OUT/dataset.hdf5" ] && { echo "[skip] $OUT exists"; exit 0; }

if [ ! -f "$EXTRA/dataset.hdf5" ] || ! $PY -c "import h5py,sys; sys.exit(0 if len(h5py.File('$EXTRA/dataset.hdf5')['data'])>=$NEW else 1)"; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/collect_target_demos.py \
    --weight "$CK" --mode rl --shift_mode fixed \
    --hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0 \
    --friction_scale 0.5 --gripper_kp_scale 0.12 --mass_scale 3.0 \
    --ring_friction_scale 2.0 --ring_size_scale 0.85 --needle_size_scale 1.0 \
    --hand_init_perturb_joint_sigma 0.02 \
    --num_demo $NEW --max_demo_len 400 --max_episodes 5000 \
    --interactive 0 --record_video 0 --drop_images 1 \
    --seed $SEED0 --out_dir "$EXTRA" || { echo "collect failed"; exit 1; }
fi

$PY - "$BASE" "$EXTRA" "$OUT" "$NEW" <<'PYEOF'
import sys, os, json, h5py, numpy as np
base, extra, out, new = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
fb = h5py.File(f"{base}/dataset.hdf5", "r"); fe = h5py.File(f"{extra}/dataset.hdf5", "r")
nb, ne = len(fb["data"]), len(fe["data"])
assert ne >= new, f"extra set has only {ne} < {new} demos"
# the two env_cfg.json must describe the same collection (shift, sigma, collector)
cb = json.load(open(f"{base}/env_cfg.json")); ce = json.load(open(f"{extra}/env_cfg.json"))
for k in ("env_name", "domain_shift", "hand_init_perturb_joint_sigma", "init_distribution"):
    assert cb.get(k) == ce.get(k), f"env_cfg mismatch on {k}: {cb.get(k)} vs {ce.get(k)}"
assert os.path.basename(cb["source_weight"]) == os.path.basename(ce["source_weight"]), \
    (cb["source_weight"], ce["source_weight"])
seeds_b = {int(fb[f"data/demo_{i}"].attrs["seed"]) for i in range(nb)}
seeds_e = {int(fe[f"data/demo_{i}"].attrs["seed"]) for i in range(new)}
assert not (seeds_b & seeds_e), "env-seed overlap between base and extra"
os.makedirs(out, exist_ok=True)
with h5py.File(f"{out}/dataset.hdf5", "w") as fo:
    g = fo.create_group("data")
    j = 0
    for src, n in ((fb, nb), (fe, new)):
        for i in range(n):
            src.copy(src[f"data/demo_{i}"], g, name=f"demo_{j}"); j += 1
    g.attrs["num_demos"] = j
cfg = dict(cb); cfg["merged_from"] = {"first": f"{base} ({nb})", "rest": f"{extra} ({new})"}
json.dump(cfg, open(f"{out}/env_cfg.json", "w"), indent=2)
lens = [int(fe[f"data/demo_{i}"].attrs["num_samples"]) for i in range(new)]
print(f"[merge] {out}: {j} demos = {nb} base + {new} new | new len mean {np.mean(lens):.1f}")
PYEOF
