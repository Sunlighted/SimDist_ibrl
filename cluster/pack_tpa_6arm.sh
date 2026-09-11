#!/bin/bash
# Package what the TPA 6-arm batch needs and git cannot carry.
#
#   bash cluster/pack_tpa_6arm.sh [outdir]     # default /user_data/Chenyang/ship
#
# release/cfgs/mimicgen/tpa_d0_state_ibrl_target.yaml is tracked -> git pull.
# cluster/sync_tpa_simdist.sh already shipped tpa_source_expert_frozen and the
# source-diverse set, and job_tpa_simdist.sh TRAINED tpa_simdist_diversev1_h5_seed1
# on the cluster -- so on a cluster that ran stage-1 those two may already exist.
# They are included anyway (SKIP_HAVE=1 to drop them) because a 45 MB re-send is
# cheaper than a job that dies on MISSING after its queue wait.
#
# NOT included: the iql/rlpd prerequisites (IQL warm start, RLPD prior buffers).
# Those are built by scripts/prep_tpa_baselines.sh and shipped separately -- the
# first six arms do not touch them.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
mkdir -p "$OUT"

LIST=$(mktemp); MISSING=0
add() { if [ -e "$1" ]; then echo "$1" >> "$LIST"; else echo "  MISSING $1"; MISSING=1; fi; }

# the 20 target demos every arm preloads (--target_num_data / --preload_num_data 20)
add release/data/mimicgen/tpa_d0_target_expert_n50/dataset.hdf5
add release/data/mimicgen/tpa_d0_target_expert_n50/env_cfg.json
# the SINGLE-head stage-1: trained on the workstation, never synced
add exps/mimicgen/tpa_simdist_plain_diversev1_h5_seed1/latest.pt
add exps/mimicgen/tpa_simdist_plain_diversev1_h5_seed1/cfg.yaml
if [ "${SKIP_HAVE:-0}" != 1 ]; then
  add exps/mimicgen/tpa_simdist_diversev1_h5_seed1/latest.pt
  add exps/mimicgen/tpa_simdist_diversev1_h5_seed1/cfg.yaml
  add exps/mimicgen/tpa_source_expert_frozen/model0.pt
  add exps/mimicgen/tpa_source_expert_frozen/cfg.yaml
fi

[ "$MISSING" = 0 ] || { echo "preflight FAILED"; rm -f "$LIST"; exit 1; }

TAR=$OUT/tpa_6arm_assets.tar.gz
tar czhf "$TAR" -T "$LIST"      # -h: dereference, in case any latest.pt is a symlink
rm -f "$LIST"
echo "== $TAR  ($(du -h "$TAR" | cut -f1)) =="
echo "  sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
echo
echo "  scp $TAR chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/"
echo "  ssh ... 'cd /project/6095505/chenyang/SimDist_ibrl && git pull && tar xzf tpa_6arm_assets.tar.gz'"
