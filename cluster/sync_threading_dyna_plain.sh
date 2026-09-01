#!/bin/bash
# Ship the assets for cluster/job_threading_dyna_plain.sh (~66 MB total).
#
# .gitignore excludes *.pt / *.hdf5 / exps/, so a fresh clone on the cluster is
# CODE ONLY. Nothing here is derived -- no local prep step is needed (unlike the
# Square baselines, which need scripts/prep_square_baselines.sh first).
#
# Usage:
#   CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_threading_dyna_plain.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
R="rsync -avh --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"
SEEDS="1 101 201 301 401 501 601 701 801 901"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen"

echo "=== 1/3  target demos: sat_160000 x 10 seeds (~2.1 MB each, 21 MB) ==="
# The unsaturated-collector sets (docs/demo-collection.md 38). 25 episodes each;
# the job passes --target_num_data 20.
for S in $SEEDS; do
  $R release/data/mimicgen/threading_d0_sat_160000_s$S $T/release/data/mimicgen/
done

echo "=== 2/3  stage-1 world models (45 MB) ==="
# cfg.yaml must come along: the adapt scripts read env_params out of it, and
# CLAUDE.md rule 17 -- the eval environment follows the checkpoint.
# 5-head ensemble, for the Dyna arm (29 MB)
$R --include='latest.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1 $T/exps/mimicgen/
# single head, same horizon 5 and same diverse_v2 source, for the plain arm (16 MB)
$R --include='latest.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/threading_simdist_diversev2_h5_seed1 $T/exps/mimicgen/

echo "=== 3/3  job files ==="
ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/cluster"
$R cluster/job_threading_dyna_plain.sh cluster/jobs_threading_dyna_plain.txt \
   $T/cluster/

echo
echo "Done. On the cluster:"
echo "  sbatch --account=def-<pi> cluster/job_threading_dyna_plain.sh"
echo "  # container route:"
echo "  SIF=cluster/ibrl.sif sbatch --account=def-<pi> --export=ALL,SIF \\"
echo "      cluster/job_threading_dyna_plain.sh"
