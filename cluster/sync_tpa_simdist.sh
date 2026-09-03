#!/bin/bash
# Ship the assets for cluster/job_tpa_simdist.sh -- TPA_D0 SimDist stage-1.
#
# UNLIKE the Square/Threading syncs, the big source dataset MUST be shipped:
# stage-1 distillation itself runs on the cluster, so there is no local prep
# step that turns 10 GB into something small. Budget ~10.4 GB of transfer.
#
# .gitignore excludes *.pt / *.hdf5 / exps/, so a fresh clone on the cluster is
# CODE ONLY.
#
# Usage:
#   CC_HOST=<host> CC_USER=<user> CC_ROOT='~/scratch/ibrl' \
#     bash cluster/sync_tpa_simdist.sh [--dry-run]
#   RESUME=1 ... bash cluster/sync_tpa_simdist.sh    # also carry the local run's
#                                                    # progress (resume.pt, 84 MB)
set -e
cd "$(dirname "$0")/.."
: "${CC_HOST:?set CC_HOST}" ; : "${CC_USER:?set CC_USER}"
CC_ROOT=${CC_ROOT:-'~/scratch/ibrl'}
DRY=${1:-}
# -L: release/data/mimicgen/tpa_d0_source_diverse_v1 is a SYMLINK into
# /user_data; without -L rsync ships a dangling link and the job dies at load.
R="rsync -avhL --progress $DRY"
T="$CC_USER@$CC_HOST:$CC_ROOT"

ssh $CC_USER@$CC_HOST "mkdir -p $CC_ROOT/release/data/mimicgen $CC_ROOT/exps/mimicgen $CC_ROOT/cluster/logs"

echo "=== 1/4  source diverse replay (10.2 GB) -- the long one ==="
# ONLY dataset.hdf5 + sidecars. part0-5.hdf5 are the per-batch intermediates
# kept locally as a backup; they are already inside dataset.hdf5 (another
# 10.2 GB of pure redundancy -- do NOT ship them).
$R --include='dataset.hdf5' --include='env_cfg.json' --include='PROVENANCE.md' \
   --exclude='part*.hdf5' --exclude='_shard*' --exclude='*' \
   release/data/mimicgen/tpa_d0_source_diverse_v1 $T/release/data/mimicgen/

echo "=== 2/4  frozen source expert -- supplies the V_src labels (103 MB) ==="
# cfg.yaml must come along: load_frozen_agent rebuilds the agent from it
# (CLAUDE.md rule 17 -- env_params follow the checkpoint).
$R --include='model0.pt' --include='cfg.yaml' --include='PROVENANCE.md' --exclude='*' \
   exps/mimicgen/tpa_source_expert_frozen $T/exps/mimicgen/

echo "=== 3/4  the expert's BC prior -- REQUIRED, easy to forget (8 MB) ==="
# tpa_source_expert_frozen/cfg.yaml has
#   bc_policy: exps/mimicgen/tpa_source_rnn_bc_tp1_n1000_seed1/model0.pt
# and load_frozen_agent loads it while rebuilding the expert. Without this the
# job dies at startup, AFTER the 10 GB replay load. Confirmed in the run log:
#   Using BC policy exps/mimicgen/tpa_source_rnn_bc_tp1_n1000_seed1/model0.pt
$R --include='model0.pt' --include='cfg.yaml' --exclude='*' \
   exps/mimicgen/tpa_source_rnn_bc_tp1_n1000_seed1 $T/exps/mimicgen/

echo "=== 4/4  job + cfg ==="
$R cluster/job_tpa_simdist.sh $T/cluster/
$R release/cfgs/mimicgen/tpa_d0_state_simdist.yaml $T/release/cfgs/mimicgen/

if [ "${RESUME:-0}" = "1" ]; then
  echo "=== extra: local run's resume.pt (84 MB) ==="
  # Carries weights + Adam moments + global_step + RNG. The resume sanity check
  # compares source_replay_path, which is RELATIVE in the cfg, so it matches on
  # the cluster as long as the repo layout is the same.
  $R --include='resume.pt' --include='cfg.yaml' --exclude='*' \
     exps/mimicgen/tpa_simdist_diversev1_h5_seed1 $T/exps/mimicgen/
fi

echo
echo "Done. On the cluster:"
echo "  sbatch --account=def-<pi> cluster/job_tpa_simdist.sh"
echo "  # the job self-chains via --resume auto until distill_steps is reached"
