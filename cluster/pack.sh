#!/bin/bash
# Package the Square-baseline assets for transfer to the cluster.
#
# Two artifacts, shipped separately on purpose:
#   1. square_baselines_assets.tar.gz  (~1 GB)  -- data + checkpoints
#   2. cluster/ibrl.sif                (12 GB)  -- already a single file; taring
#      it just burns time and disk for ~0% compression.
#
# A manifest with sha256 goes inside the tarball AND next to it, so the far end
# can verify without unpacking. Everything below is excluded from git
# (.gitignore drops *.pt / *.hdf5 / exps/), which is exactly why it has to be
# shipped out of band.
#
# Usage:  bash cluster/pack.sh [outdir]        (default /user_data/Chenyang/ship)
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
mkdir -p "$OUT"
SEEDS="1 101 201 301 401 501 601 701 801 901"

# Build the file list. Checkpoints ship with their cfg.yaml because the loaders
# read it from the ckpt's own directory:
#   train_rl.py    -> dirname(bc_policy)/cfg.yaml       (rebuilds the BC prior)
#   train_simdist_dyna -> dirname(stage1_ckpt)/cfg.yaml (rebuilds the world model
#                                                        AND the env/domain shift)
# agent_warmstart.pt needs no cfg -- it is loaded as a plain state_dict.
LIST=$(mktemp)
for S in $SEEDS; do
  echo "release/data/mimicgen/square_sat_600000_s$S"    >> $LIST
  echo "release/data/mimicgen/square_rlpd_prior_s$S"    >> $LIST
done
for f in exps/mimicgen/square_source_mlp_bc_n1000_seed1/model0.pt \
         exps/mimicgen/square_source_mlp_bc_n1000_seed1/cfg.yaml \
         exps/mimicgen/square_iql_offline_source/agent_warmstart.pt \
         exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/latest.pt \
         exps/mimicgen/square_simdist_rotw1_dynens_h5_seed1/cfg.yaml ; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
  echo "$f" >> $LIST
done

echo "=== manifest ==="
MAN="$OUT/MANIFEST.sha256"
: > "$MAN"
while read -r p; do
  if [ -d "$p" ]; then find "$p" -type f -exec sha256sum {} \; >> "$MAN"
  else sha256sum "$p" >> "$MAN"; fi
done < $LIST
echo "  $(wc -l < "$MAN") files"

cp "$MAN" ./MANIFEST.sha256
# `videos/` holds the collection-time mp4 recordings -- not needed to train.
tar czf "$OUT/square_baselines_assets.tar.gz" --exclude=videos -T $LIST MANIFEST.sha256
rm -f ./MANIFEST.sha256 $LIST

sha256sum "$OUT/square_baselines_assets.tar.gz" > "$OUT/square_baselines_assets.tar.gz.sha256"
SIF_LOCAL=${SIF:-/user_data/Chenyang/ibrl.sif}
[ -f "$SIF_LOCAL" ] && sha256sum "$SIF_LOCAL" > "$OUT/ibrl.sif.sha256"

echo
echo "=== packed ==="
ls -lh "$OUT" | tail -n +2 | awk '{printf "  %-42s %s\n", $9, $5}'
echo
echo "Ship both, then on the cluster:"
echo "  tar xzf square_baselines_assets.tar.gz -C \$CC_ROOT"
echo "  cd \$CC_ROOT && sha256sum -c MANIFEST.sha256 --quiet && echo 'all files verified'"
