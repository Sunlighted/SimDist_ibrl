#!/bin/bash
# Ship the assets for cluster/job_sqcof_dynaibrl.sh (Square + Coffee Dyna-IBRL).
#
# .gitignore excludes *.pt / *.hdf5 / exps/, so a fresh clone on the cluster is
# CODE ONLY. Everything below has to be rsync'd.
#
# The one derived asset is the bc_anchor state pool. _build_bc_anchor reads only
# `obs/state` from a few thousand episodes, so shipping the 5.5 GB / 7.7 GB
# source-diverse sets would be almost entirely waste. Build the extracts first:
#
#   python scripts/extract_anchor_pool.py \
#       release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5 \
#       release/data/mimicgen/square_anchor_pool_4k/dataset.hdf5 --episodes 4000
#   python scripts/extract_anchor_pool.py \
#       release/data/mimicgen/coffee_d0_source_diverse_v1/dataset.hdf5 \
#       release/data/mimicgen/coffee_anchor_pool_4k/dataset.hdf5 --episodes 4000
#
# Usage:
#   CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_sqcof_dynaibrl.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
R="rsync -avh --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"
SEEDS="1 101 201 301 401"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen $CC_ROOT/cluster"

echo "=== 1/4  target demos (25 episodes each; the job passes --target_num_data 20) ==="
for S in $SEEDS; do
  $R release/data/mimicgen/square_sat_400000_s$S     $T/release/data/mimicgen/
  $R release/data/mimicgen/coffee_g5x_sat400k_s$S    $T/release/data/mimicgen/
done

echo "=== 2/4  stage-1 world models (4 x ~28 MB) ==="
# cfg.yaml must come along: the adapt scripts read env_params out of it, and
# CLAUDE.md rule 17 -- the eval environment follows the checkpoint, so a missing
# cfg silently changes the task.
# 5-head and single-head at the SAME horizon 5 and the same source data, so
# plain-vs-ensemble is not confounded with horizon.
for D in square_simdist_rotw1_dynens_h5_seed1 square_simdist_rotw1_h5_seed1 \
         coffee_simdist_diversev1_h5_seed1 coffee_simdist_plain_h5_seed1; do
  $R --copy-links --include='latest.pt' --include='cfg.yaml' --exclude='*' \
     exps/mimicgen/$D $T/exps/mimicgen/
done

echo "=== 3/4  bc_anchor state pools (72 MB + 222 MB) ==="
# Only needed when the job runs with ANCHOR_COEF>0; cheap enough to ship anyway.
$R release/data/mimicgen/square_anchor_pool_4k  $T/release/data/mimicgen/
$R release/data/mimicgen/coffee_anchor_pool_4k  $T/release/data/mimicgen/

echo "=== 4/4  job files ==="
$R cluster/job_sqcof_dynaibrl.sh cluster/jobs_sqcof_dynaibrl.txt \
   cluster/jobs_sqcof_dynaibrl_round2.txt cluster/jobs_sqcof_dynaibrl_noanneal.txt \
   cluster/make_joblist.py \
   cluster/README_sqcof_dynaibrl.md $T/cluster/

echo
echo "Done. On the cluster:"
echo "  # round 1 -- the two dynens arms, 2 tasks x 10 seeds. --array is REQUIRED:"
echo "  # the header used to say 1-20 against a 40-line list and dropped all cof_*."
echo "  sbatch --account=def-<pi> --array=1-40 cluster/job_sqcof_dynaibrl.sh"
echo "  # round 2 -- plain arms + constant-margin (_m001) arms, seed-major:"
echo "  JOBS=cluster/jobs_sqcof_dynaibrl_round2.txt \\"
echo "    sbatch --account=def-<pi> --array=1-80 --export=ALL,JOBS \\"
echo "    cluster/job_sqcof_dynaibrl.sh"
echo "  # with the stage-2 anchor on (recommended for Coffee, see the README):"
echo "  ANCHOR_COEF=1.0 sbatch --account=def-<pi> --export=ALL,ANCHOR_COEF \\"
echo "      cluster/job_sqcof_dynaibrl.sh"
