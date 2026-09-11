#!/bin/bash
# Launch the 100k-trajectory pixel stage-1 arms once the generation shards land.
#
# Verifies the ARTIFACT before spending ~40 GPU-hours (four rc=0 traps so far in
# this project): every shard must exist, be non-empty, carry JPEG-coded frames at
# the expected resolution, and sum to the transition count we sized the run from.
#
# Usage:  bash scripts/launch_pixel224_stage1.sh [--go]
#         (without --go it only reports; nothing is submitted)
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env_drac.sh >/dev/null 2>&1

GLOB="/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_*/dataset.hdf5"
WANT_SHARDS="${WANT_SHARDS:-8}"
SRC=exps/mimicgen/threading_ibrl_source_rnnprior_seed1/model0.pt

shopt -s nullglob
PARTS=( $GLOB )
shopt -u nullglob
echo "shards present: ${#PARTS[@]} / $WANT_SHARDS"
if [ "${#PARTS[@]}" -lt "$WANT_SHARDS" ]; then
    echo "NOT READY -- still generating. Re-run when all $WANT_SHARDS shards exist."
    exit 1
fi

# A shard file EXISTS from the moment merge_shards starts writing it, and it
# grows for minutes. `[ -s ]` and even a successful h5py open are both true
# mid-merge -- shard_8 was 7.1 GB against 40 GB for its siblings while still
# merging. So require the shard's JOB to be terminal before trusting the file.
for J in $(cat /scratch/yifanr/simdist/pix224_shards.jobids 2>/dev/null | awk '{print $1}'); do
    ST=$(sacct -j "$J" -n -o State -X 2>/dev/null | head -1 | tr -d ' ')
    case "$ST" in
        COMPLETED) ;;
        FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL)
            echo "FATAL: generation job $J ended $ST -- shard set is incomplete"; exit 1;;
        *) echo "NOT READY: generation job $J is $ST (file may be mid-merge)"; exit 1;;
    esac
done

python - "${PARTS[@]}" <<'PYV'
import h5py, sys, numpy as np
tot_tr = tot_ep = tot_ok = 0
EXPECT = 12500
for p in sys.argv[1:]:
    with h5py.File(p, "r") as f:
        d = list(f["data"].keys())
        n = len(d)
        # merge_shards sets num_demos ONLY after its final loop, so a mismatch
        # here means we are looking at a partially written file.
        att = int(f["data"].attrs.get("num_demos", -1))
        if att != n or n != EXPECT:
            raise SystemExit(f"FATAL: {p} has {n} demos, attr says {att}, "
                             f"expected {EXPECT} -- file is incomplete")
        lens = np.array([f[f"data/{k}/actions"].shape[0] for k in d])
        k0 = d[0]
        cams = [c for c in f[f"data/{k0}/obs"].keys() if c.endswith("_image")]
        ds = f[f"data/{k0}/obs/{cams[0]}"]
        codec = ds.attrs.get("codec", "raw")
        fshape = tuple(int(x) for x in ds.attrs.get("frame_shape", ()))
        ok = (codec == "jpeg" and fshape == (3, 224, 224) and len(cams) == 2)
        tot_ep += n; tot_tr += int(lens.sum()); tot_ok += int(ok)
        print(f"  {p.split('/')[-2]:>9}: {n:>6} traj {lens.sum():>9} trans "
              f"codec={codec} frame={fshape} cams={len(cams)} {'OK' if ok else '<== BAD'}")
print(f"\nTOTAL: {tot_ep} trajectories, {tot_tr} transitions, {tot_ok}/{len(sys.argv)-1} shards OK")
if tot_ok != len(sys.argv) - 1:
    raise SystemExit("FATAL: some shards are not JPEG@224x2cam -- do NOT train on this")
if tot_tr < 12_000_000:
    print(f"WARNING: only {tot_tr} transitions (expected ~18M for 100k traj)")
PYV
rc=$?
[ "$rc" -ne 0 ] && { echo "artifact check FAILED (rc=$rc)"; exit $rc; }

if [ "${1:-}" != "--go" ]; then
    echo; echo "artifact check PASSED. Re-run with --go to submit."
    exit 0
fi

# --mem=200G: the 8M-transition cap is ~151 GB resident at 18,935 B/transition.
# 11:59 chunks (b2) with self-requeue rather than 23:59 (b3) -- shorter tier
# queues faster and pixel_stage1.slrm chains via --dependency=afterany.
J1=$(CFG=release/cfgs/mimicgen/threading_d0_pixel_simdist_r18_224.yaml \
     SAVE=exps/mimicgen/pixel_r18_224_seed1 SRC="$SRC" REPLAY="$GLOB" \
     STEPS=150000 MAX_CHUNK=6 \
     sbatch --parsable --mem=200G --time=11:59:00 --job-name=r18_224 \
            scripts/slurm/pixel_stage1.slrm)
echo "r18_224 stage-1 -> job $J1"

J2=$(CFG=release/cfgs/mimicgen/threading_d0_pixel_simdist_vit96_data100k.yaml \
     SAVE=exps/mimicgen/pixel_vit96_data100k_seed1 SRC="$SRC" REPLAY="$GLOB" \
     STEPS=150000 MAX_CHUNK=3 \
     sbatch --parsable --mem=200G --time=11:59:00 --job-name=vit96_100k \
            scripts/slurm/pixel_stage1.slrm)
echo "vit96 data-scale control -> job $J2"
echo "$J1 r18_224"$'\n'"$J2 vit96_data100k" > /scratch/yifanr/simdist/stage1_224.jobids
