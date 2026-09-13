#!/bin/bash
# Re-ship release/data to the cluster after commit da6f993 ("pixel") replaced it
# with a symlink to one machine's scratch (/scratch/yifanr/simdist/data). Pulling
# that commit made git DELETE the real directory on every other machine -- the
# contents are gitignored, so git treated them as removable. Both the workstation
# and the cluster lost it.
#
#   bash cluster/pack_data_restore.sh [outdir]    # default /user_data/Chenyang/ship
#
# Two groups, because the sizes differ by 10x:
#   small.tar  demos, RLPD priors, anchor-pool extracts   ~5 GB
#   the source-diverse sets (threading 4.1 GB + coffee 7.2 GB) are NOT tarred:
#   they already live on /user_data and a tar would just duplicate them on a
#   filesystem that is 97% full. rsync them straight from there (printed below).
#
# tpa_d0_source_diverse_v1 (20 GB) and core/ (45 GB) are deliberately excluded:
# nothing in the queued jobs reads them. TPA's prep artefacts (rlpd priors, IQL
# warm start) are already built and ARE included.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
DATA=release/data/mimicgen
mkdir -p "$OUT"

LIST=$(mktemp)
add() { [ -e "$DATA/$1" ] && echo "$DATA/$1" >> "$LIST" || echo "  (skip, absent) $1"; }

for s in 1 101 201 301 401 501 601 701 801 901; do
  for p in threading_d0_sat_160000 square_sat_400000 square_sat_600000 \
           coffee_g5x_sat400k square_rlpd_prior coffee_rlpd_prior tpa_rlpd_prior; do
    [ -e "$DATA/${p}_s${s}" ] && echo "$DATA/${p}_s${s}" >> "$LIST"
  done
done
for p in threading_anchor_pool_4k threading_anchor_pool_ex4k \
         square_anchor_pool_4k coffee_anchor_pool_4k \
         tpa_d0_target_expert_n50 tpa_source_diverse_5k.npz; do
  add "$p"
done

n=$(grep -c '' "$LIST")
TAR=$OUT/data_restore_small.tar
# -h dereferences the /user_data symlinks; no -z, these are float32 HDF5 (24%)
tar chf "$TAR" -T "$LIST"
rm -f "$LIST"
echo "== $TAR  ($(du -h "$TAR" | cut -f1), $n dirs) =="
echo "  sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
echo
echo "Ship the small bundle:"
echo "  scp $TAR chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/"
echo "  ssh ... 'cd /project/6095505/chenyang/SimDist_ibrl && tar xf data_restore_small.tar'"
echo
echo "Ship the two source-diverse sets by rsync (they are already on /user_data;"
echo "resumable, and no local duplicate on a 97%-full root fs):"
for d in threading_d0_source_diverse_v2 coffee_d0_source_diverse_v1; do
  t=$(readlink -f "$DATA/$d" 2>/dev/null || echo "$DATA/$d")
  echo "  rsync -avL --partial --progress $t \\"
  echo "    chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/release/data/mimicgen/"
done
