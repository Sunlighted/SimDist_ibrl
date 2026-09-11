#!/bin/bash
# Stage 2/3 (target adapt) for the 224 + ResNet-18 arm.
#
# Two modes:
#   probe : tiny run (3k transitions, EVAL_N=20) purely to MEASURE s/round, so the
#           real run's walltime comes from `sacct`, not a guess (iron law #16).
#           The 480 s/round figure in adapt.slrm was measured at vit@96; the
#           encoder is 16x more expensive at 224 and the eval is ~95% of a round,
#           so that number does NOT transfer.
#   real  : the faithful protocol -- 100k target transitions, eval every 1k,
#           100 games/point, strict, sigma=0.1 (docs/on-policy-adaptation.md:3-8).
#           Headline = mean over the 75-100k steady-state window, NOT the last point.
#
# Usage: bash scripts/launch_pixel224_adapt.sh probe|real [TIME]
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-probe}"
STAGE1=exps/mimicgen/pixel_r18_224_seed1/latest.pt
DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5
ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5

for f in "$STAGE1" "$DEMOS" "$ANCHOR"; do
    [ -s "$f" ] || { echo "NOT READY: missing $f"; exit 1; }
done

# the demos and the anchor must BOTH be 224, or the frozen encoder sees a
# resolution it was never distilled on (and nothing would crash -- it would just
# score badly, which is the worst kind of failure here).
source set_env_drac.sh >/dev/null 2>&1
python - "$DEMOS" "$ANCHOR" <<'PYV'
import h5py, sys
from rl.sequence_replay import read_frames_array
for p in sys.argv[1:]:
    with h5py.File(p, "r") as f:
        k = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[0]
        cams = [c for c in f[f"data/{k}/obs"].keys() if c.endswith("_image")]
        assert len(cams) == 2, f"{p}: expected 2 cameras, got {cams}"
        a = read_frames_array(f[f"data/{k}/obs/{cams[0]}"])
        print(f"  {p.split('/')[-2]}: {len(f['data'])} eps, frames {a.shape[1:]}")
        assert a.shape[1:] == (3, 224, 224), f"{p} is NOT 224: {a.shape[1:]}"
print("demos + anchor are both 224x2cam: OK")
PYV
[ $? -ne 0 ] && { echo "FATAL: asset check failed"; exit 1; }

if [ "$MODE" = "probe" ]; then
    T="${2:-2:59:00}"; MT=3000; EN=20; TAG=probe
else
    T="${2:-23:59:00}"; MT=100000; EN=100; TAG=seed1
fi

J=$(ARM=pixel STAGE1="$STAGE1" DEMOS="$DEMOS" ANCHOR="$ANCHOR" \
    SAVE="exps/mimicgen/pixel224_adapt_${TAG}" \
    NUM_DEMO=20 MAX_TRANS=$MT EVAL_N=$EN INTERVAL=1000 \
    sbatch --parsable --time="$T" --mem=96G --job-name="ad224_$TAG" \
           scripts/slurm/adapt.slrm)
echo "pixel224 adapt ($MODE) -> job $J   [max_trans=$MT eval_n=$EN time=$T]"
[ "$MODE" = "probe" ] && echo "When it ends:  sacct -j $J -o Elapsed,MaxRSS -X   => size the real run from s/round"
