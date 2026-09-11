#!/bin/bash
# Package the anchor-loss ablation's anchor pool.
#
#   bash cluster/pack_anchor_abl.sh [outdir]      # default /user_data/Chenyang/ship
#
# threading_anchor_pool_ex4k (159 MB) is a 4000-episode extract of the 4.3 GB
# threading_d0_source_diverse_v2 carrying everything the anchor knobs read:
#   obs/state      always
#   actions        bc_anchor_data_actions=1  (the regression target)
#   expert_flags   bc_anchor_expert_only=1   (asserted to exist)
#   success attr   bc_anchor_success_only=1  (defaults to False -> would silently
#                                             drop every episode if missing)
# Rebuild with:
#   python scripts/extract_anchor_pool.py <src> <dst> --episodes 4000 \
#       --with_actions 1 --with_expert_flags 1
#
# ⚠️ NOT interchangeable with threading_anchor_pool_4k (115 MB, state only) --
# that one cannot serve the anch_es_* arms. The job script points every anchor
# arm at THIS file so the state distribution is identical across the table.
#
# The 100k target demos (threading_d0_sat_160000_s*) and the dynens stage-1 are
# already on the cluster from earlier batches; this bundle is only the pool.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
mkdir -p "$OUT"
SRC=release/data/mimicgen/threading_anchor_pool_ex4k/dataset.hdf5
[ -f "$SRC" ] || { echo "MISSING $SRC"; echo "build it with scripts/extract_anchor_pool.py"; exit 1; }

TAR=$OUT/threading_anchor_pool_ex4k.tar.gz
tar czhf "$TAR" "$SRC"
echo "== $TAR  ($(du -h "$TAR" | cut -f1)) =="
echo "  sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
echo
echo "  scp $TAR chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/"
echo "  ssh ... 'cd /project/6095505/chenyang/SimDist_ibrl && tar xzf $(basename "$TAR")'"
echo
echo "Verify on the far end:"
echo "  python3 -c \"import h5py;f=h5py.File('$SRC');g=f['data'];k=sorted(g)[0];\\"
echo "print(dict(g.attrs));print(list(g[k].keys()),dict(g[k].attrs))\""
echo "  # expect with_actions=1 with_expert_flags=1, keys incl. actions+expert_flags"
