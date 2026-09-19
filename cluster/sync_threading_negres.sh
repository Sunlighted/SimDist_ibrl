#!/bin/bash
# Ship the assets for cluster/job_threading_negres.sh.
# The N=500 demo sets are built LOCALLY (scripts/build_negres_n500.sh); rerun this
# script after they finish and it ships them too.
#
# .gitignore excludes *.pt / *.hdf5 / exps/, so the cluster clone is CODE ONLY.
# The `vcalib` arm also needs code that is NEW (train_simdist_adapt.py
# --value_calib + SimDistAgent.update_value_calib): either `git pull` on the
# cluster after pushing, or let step 3 rsync those two files.
#
# Usage:
#   CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_threading_negres.sh [--dry-run]
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
R="rsync -avh --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"
SEEDS="1 101 201 301 401"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen $CC_ROOT/cluster/logs"

echo "=== 1/3  target demos: sat_160000 x 5 seeds (~2.1 MB each) ==="
for S in $SEEDS; do
  $R release/data/mimicgen/threading_d0_sat_160000_s$S $T/release/data/mimicgen/
  # N=500 sets, built locally by scripts/build_negres_n500.sh (demo500 arm)
  if [ -f release/data/mimicgen/threading_d0_sat_160000_n500_s$S/dataset.hdf5 ]; then
    $R release/data/mimicgen/threading_d0_sat_160000_n500_s$S $T/release/data/mimicgen/
  else
    echo "  (n500_s$S not built yet -- rerun this script once it is)"
  fi
done

echo "=== 2/3  stage-1 world models: plain h5 (16 MB) + value-ensemble h5 (82 MB) ==="
# cfg.yaml must come along -- the eval env follows the checkpoint (iron law #17).
for D in threading_simdist_diversev2_h5_seed1 threading_simdist_vens_diversev2_h5_seed1; do
  $R --include='latest.pt' --include='cfg.yaml' --exclude='*' \
     exps/mimicgen/$D $T/exps/mimicgen/
done

echo "=== 3/3  job files + the two changed python files ==="
$R cluster/job_threading_negres.sh cluster/jobs_threading_negres.txt \
   cluster/jobs_threading_negres_demo500.txt cluster/jobs_threading_negres_vens.txt $T/cluster/
$R train_simdist_adapt.py $T/
$R rl/simdist_agent.py $T/rl/

echo
echo "Done. On the cluster (all arms must run on the SAME code -- each task logs [code]):"
echo "  WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres.txt \\"
echo "    sbatch --account=rrg-rhinehar --array=1-20 --export=ALL,JOBS cluster/job_threading_negres.sh"
echo "  # once the n500 sets are synced:"
echo "  WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres_demo500.txt \\"
echo "    sbatch --account=rrg-rhinehar --array=1-5 --export=ALL,JOBS cluster/job_threading_negres.sh"
echo "  # value ensemble (existing stage-1):"
echo "  WANDB_MODE=online SIF=cluster/ibrl.sif JOBS=cluster/jobs_threading_negres_vens.txt \\"
echo "    sbatch --account=rrg-rhinehar --array=1-5 --export=ALL,JOBS cluster/job_threading_negres.sh"
