#!/bin/bash
# Ship the assets for cluster/job_threading_margin.sh and its BC prerequisite
# cluster/job_threading_bc_s160.sh (~40 MB total).
#
# .gitignore excludes *.pt / *.hdf5 / exps/, so a fresh clone on the cluster is
# CODE ONLY. Nothing here is derived on the cluster except the BC priors, which
# job_threading_bc_s160.sh builds there from the pieces synced below.
#
# Usage:
#   CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_threading_margin.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
R="rsync -avh --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"
SEEDS="1 101 201 301 401 501 601 701 801 901"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen $CC_ROOT/cluster"

echo "=== 1/4  target demos: sat_160000 x 10 seeds (~2.1 MB each, 21 MB) ==="
# The unsaturated-collector sets (docs/demo-collection.md 38). 25 episodes each;
# every arm passes --target_num_data 20 (--preload_num_data 20 on the model-free
# arm), and job_threading_bc_s160.sh fine-tunes the BC priors on the same files.
for S in $SEEDS; do
  $R release/data/mimicgen/threading_d0_sat_160000_s$S $T/release/data/mimicgen/
done

echo "=== 2/4  stage-1 world model: SINGLE head, horizon 5 (16 MB) ==="
# cfg.yaml must come along: the adapt scripts read env_params out of it, and
# CLAUDE.md rule 17 -- the eval environment follows the checkpoint.
# Only the single-head model. The 5-head dynens checkpoint is deliberately NOT
# synced: this batch holds the dynamics fixed across plain_h5 and both dyna arms
# so that "planner vs actor" is not confounded with "ensemble vs single head".
$R --include='latest.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/threading_simdist_diversev2_h5_seed1 $T/exps/mimicgen/

echo "=== 3/4  source BC policy the s160 priors fine-tune from (3 MB) ==="
# job_threading_bc_s160.sh starts every seed from this one checkpoint, exactly as
# the original il_ft_target_seed* runs did.
$R --include='model0.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/threading_source_mlp_bc_n2000_seed1 $T/exps/mimicgen/

echo "=== 4/4  job files ==="
$R cluster/job_threading_margin.sh cluster/jobs_threading_margin.txt \
   cluster/job_threading_bc_s160.sh cluster/jobs_threading_bc_s160.txt \
   cluster/README_threading_margin.md $T/cluster/

echo
echo "Done. On the cluster, in order:"
echo "  # 1. BC priors on sat_160000 -- REQUIRED by the ibrl_m005anneal0 arm."
echo "  sbatch --account=def-<pi> cluster/job_threading_bc_s160.sh"
echo "  # 2. the four arms, once the 10 il_ft_target_s160_seed*/model0.pt exist"
echo "  sbatch --account=def-<pi> cluster/job_threading_margin.sh"
echo
echo "  # container route (add --export=ALL,SIF to either):"
echo "  SIF=cluster/ibrl.sif sbatch --account=def-<pi> --export=ALL,SIF \\"
echo "      cluster/job_threading_margin.sh"
echo
echo "  # arms 1-30 need no BC prior; to start them while the priors train:"
echo "  sbatch --account=def-<pi> --array=1-30 cluster/job_threading_margin.sh"
