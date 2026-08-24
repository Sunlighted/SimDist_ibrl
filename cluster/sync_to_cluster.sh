#!/bin/bash
# Ship the Square baseline assets to a Compute Canada cluster.
#
# The repo's .gitignore excludes *.pt / *.hdf5 / *.npz / exps/, so a fresh
# `git clone` on the cluster gives you CODE ONLY -- every checkpoint and dataset
# below has to be rsync'd. Run `bash scripts/prep_square_baselines.sh` FIRST:
# it builds the derived assets locally so the 5.2 GB source diverse set never
# has to be shipped.
#
# Usage:
#   CC_HOST=narval.alliancecan.ca CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_to_cluster.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
R="rsync -avh --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"
SEEDS="1 101 201 301 401 501 601 701 801 901"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen"

echo "=== 1/4  target demos (10 x ~1.5 MB) ==="
for S in $SEEDS; do
  $R release/data/mimicgen/square_sat_600000_s$S $T/release/data/mimicgen/
done

echo "=== 2/4  RLPD prior sets (10 x ~25 MB) ==="
for S in $SEEDS; do
  $R release/data/mimicgen/square_rlpd_prior_s$S $T/release/data/mimicgen/
done

echo "=== 3/4  checkpoints ==="
# IBRL's prior: the source MLP BC, unmodified and shared by all seeds (8.5 MB).
# cfg.yaml must come along -- train_rl.py reads it to rebuild the policy.
$R --include='model0.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/square_source_mlp_bc_n1000_seed1 $T/exps/mimicgen/
# IQL warm start (shared by all seeds)
$R --include='agent_warmstart.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/square_iql_offline_source $T/exps/mimicgen/
# stage-1 world model for the Dyna arms (dynens, 5 heads) + its cfg
$R --include='latest.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1 $T/exps/mimicgen/

echo "=== 4/5  Apptainer image (12 GB, only if you built it) ==="
# Kept OUT of the repo tree by default: 12 GB on the repo filesystem for no
# reason, and .gitignore would have to cover it. Override with SIF=<path>.
SIF_LOCAL=${SIF:-/user_data/Chenyang/ibrl.sif}
if [ -f "$SIF_LOCAL" ]; then
  ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/cluster"
  $R "$SIF_LOCAL" $T/cluster/ibrl.sif
else
  echo "  $SIF_LOCAL not present -- skipping (using the venv route, step 3A)"
fi

echo "=== 5/5  robosuite/mimicgen asset check ==="
echo "  (assets come from the pip-installed deps/ submodules -- nothing to copy,"
echo "   but the MimicGen core demo file IS needed if you ever retrain a prior:"
echo "   release/data/mimicgen/core/square_d0/processed_data96.hdf5)"
echo
echo "SYNC DONE.  Next: ssh in and run cluster/cc_setup.sh once."
