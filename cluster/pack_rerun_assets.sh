#!/bin/bash
# Package the assets the 2026-09-06 rerun needs that git CANNOT carry.
#
#   bash cluster/pack_rerun_assets.sh [outdir]     # default /user_data/Chenyang/ship
#
# Code, job scripts, job lists and release/cfgs/**.yaml are all tracked -- those
# go over with `git pull` on the cluster. Only these two groups are gitignored:
#
#   1. exps/mimicgen/coffee_s1g5x_{dynens,plain}   the CORRECT G5x stage-1.
#      Their latest.pt is a symlink to coffee_simdist_*_h5_seed1/latest.pt (same
#      weights -- stage-1 is distilled in the source domain); what differs is
#      cfg.yaml's eval_domain_shift. tar -h follows the link so the far end gets
#      a real file and does not need the link target to exist.
#   2. release/data/mimicgen/{square_sat_400000,coffee_g5x_sat400k}_s{501..901}
#      Seeds >= 501 were never shipped: sync_sqcof_dynaibrl.sh's SEEDS list still
#      said "1 101 201 301 401" while the job lists had held 10 seeds for a while.
#
# NOT included (ship separately if you have not already):
#   ibrl_m001_priors.tar.gz  -- 30 BC priors + 3 source experts, 502 MB,
#                               built by cluster/pack_ibrl_priors.sh
#   *_anchor_pool_4k         -- only needed with ANCHOR_COEF > 0 (default 0)
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
SEEDS=${SEEDS:-"501 601 701 801 901"}
mkdir -p "$OUT"

LIST=$(mktemp); MISSING=0
add() { if [ -e "$1" ]; then echo "$1" >> "$LIST"; else echo "  MISSING $1"; MISSING=1; fi; }

for D in coffee_s1g5x_dynens coffee_s1g5x_plain; do
  add "exps/mimicgen/$D/cfg.yaml"
  add "exps/mimicgen/$D/latest.pt"
done
for S in $SEEDS; do
  add "release/data/mimicgen/square_sat_400000_s$S"
  add "release/data/mimicgen/coffee_g5x_sat400k_s$S"
done

[ "$MISSING" = 0 ] || { echo "preflight FAILED"; rm -f "$LIST"; exit 1; }

TAR=$OUT/rerun_assets.tar.gz
MAN=$OUT/rerun_assets.manifest.txt
: > "$MAN"
while read -r f; do
  if [ -d "$f" ]; then printf 'dir   %s\n' "$f" >> "$MAN"
  # `sha256sum -L` is BSD-only; on GNU coreutils redirect instead, which reads
  # through the symlink (coffee_s1g5x_*/latest.pt).
  else printf '%s  %s\n' "$(sha256sum < "$f" | cut -c1-16)" "$f" >> "$MAN"; fi
done < "$LIST"

# -h: dereference symlinks (coffee_s1g5x_*/latest.pt), so the tarball is portable
tar czhf "$TAR" -T "$LIST"
rm -f "$LIST"

echo "== $TAR  ($(du -h "$TAR" | cut -f1)) =="
echo "  sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
echo "  manifest: $MAN"
echo
echo "Ship (paths inside are repo-relative -- untar AT the repo root):"
echo "  scp $TAR chenyc@nibi.alliancecan.ca:/project/6095505/chenyang/SimDist_ibrl/"
echo "  ssh chenyc@nibi.alliancecan.ca 'cd /project/6095505/chenyang/SimDist_ibrl && \\"
echo "      git pull && tar xzf rerun_assets.tar.gz'"
echo
echo "Then verify on the far end (the cfgs themselves come over via git):"
echo "  cd /project/6095505/chenyang/SimDist_ibrl"
echo "  ls -lL exps/mimicgen/coffee_s1g5x_*/latest.pt        # two real files, 29M + 16M"
echo "  grep -A2 'friction_scale' exps/mimicgen/coffee_s1g5x_dynens/cfg.yaml | head -3   # 0.2, not 0.5"
echo "  ls -d release/data/mimicgen/{square_sat_400000,coffee_g5x_sat400k}_s{501,601,701,801,901} | wc -l   # 10"
echo "  python3 scripts/audit_runs.py exps/coffee_g5x_3arm/*/   # which finished runs are stale"
