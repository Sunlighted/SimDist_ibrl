#!/bin/bash
# Package the TPA iql/rlpd prerequisites built by scripts/prep_tpa_baselines.sh.
#
#   bash cluster/pack_tpa_baselines.sh [outdir]        # all 10 seeds
#   SEEDS="1 101 201 301 401" bash cluster/pack_tpa_baselines.sh
#
# Contents:
#   exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt   52 MB, shared by all seeds
#   release/data/mimicgen/tpa_rlpd_prior_s<N>/dataset.hdf5    388 MB EACH
#
# ⚠️ 3.9 GB at 10 seeds. TPA episodes are 1000 steps (~6x Threading), so a
# 2000-episode RLPD prior is an order of magnitude bigger here than on the other
# tasks. gzip only buys 24% on this float32 HDF5 (388 -> 294 MB) for a lot of
# CPU, so the tar is written UNCOMPRESSED.
#
# For a transfer this size prefer rsync over scp+tar: it resumes a broken link,
# needs no 3.9 GB local duplicate, and skips seeds already on the far end. The
# script prints both; the tar exists only for consistency with the other bundles.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
SEEDS=${SEEDS:-"1 101 201 301 401 501 601 701 801 901"}
mkdir -p "$OUT"

LIST=$(mktemp); MISSING=0
add() { if [ -e "$1" ]; then echo "$1" >> "$LIST"; else echo "  MISSING $1"; MISSING=1; fi; }

add exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt
for S in $SEEDS; do
  add "release/data/mimicgen/tpa_rlpd_prior_s$S/dataset.hdf5"
  # env_cfg.json is optional in a merged prior set; include it when present
  [ -f "release/data/mimicgen/tpa_rlpd_prior_s$S/env_cfg.json" ] && \
    echo "release/data/mimicgen/tpa_rlpd_prior_s$S/env_cfg.json" >> "$LIST"
done

if [ "$MISSING" != 0 ]; then
  echo; echo "Build them first:  bash scripts/prep_tpa_baselines.sh"
  rm -f "$LIST"; exit 1
fi

MAN=$OUT/tpa_baselines.manifest.txt
: > "$MAN"
while read -r f; do
  printf '%s  %s\n' "$(sha256sum < "$f" | cut -c1-16)" "$f" >> "$MAN"
done < "$LIST"

TAR=$OUT/tpa_baselines_assets.tar
tar chf "$TAR" -T "$LIST"          # no -z: see the header
rm -f "$LIST"
echo "== $TAR  ($(du -h "$TAR" | cut -f1)) =="
echo "  manifest: $MAN  ($(grep -c '' "$MAN") files)"
echo
echo "RECOMMENDED -- rsync (resumable, skips what is already there):"
echo "  rsync -avL --partial --progress \\"
echo "    exps/mimicgen/tpa_iql_offline_source \\"
echo "    chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/exps/mimicgen/"
echo "  rsync -avL --partial --progress \\"
echo "    release/data/mimicgen/tpa_rlpd_prior_s* \\"
echo "    chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/release/data/mimicgen/"
echo
echo "or the tar (untar AT the repo root):"
echo "  scp $TAR chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/"
echo "  ssh ... 'cd /project/6095505/chenyang/SimDist_ibrl && tar xf tpa_baselines_assets.tar'"
echo
echo "Verify on the far end:"
echo "  ls release/data/mimicgen/tpa_rlpd_prior_s*/dataset.hdf5 | wc -l    # expect $(echo $SEEDS | wc -w)"
echo "  ls -l exps/mimicgen/tpa_iql_offline_source/agent_warmstart.pt      # 51822401 bytes"
