#!/bin/bash
# Package the model-free `ibrl_m001` prerequisites for transfer to the cluster.
#
#   bash scripts/train_target_bc_priors.sh     # first: train them locally
#   bash cluster/pack_ibrl_priors.sh [outdir]  # default /user_data/Chenyang/ship
#
# Contents (~260 MB, 30 priors x ~8.5 MB):
#   exps/mimicgen/il_ft_target_s160_seed*          Threading  (ibrl_m005anneal0 + ibrl_m001)
#   exps/mimicgen/square_il_ft_target_rotw1_seed*  Square     (sq_ibrl_m001)
#   exps/mimicgen/coffee_il_ft_target_g5x_seed*    Coffee     (cof_ibrl_m001)
#   release/cfgs/mimicgen/{square_rotw1,coffee_g5x}_ibrl_m001.yaml
#
# ⚠️ Each prior ships with its cfg.yaml, not just model0.pt: train_rl.py rebuilds
# the BC prior by reading dirname(bc_policy)/cfg.yaml. A tarball of bare .pt
# files fails at load time, after the queue wait.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-/user_data/Chenyang/ship}
SEEDS=${SEEDS:-"1 101 201 301 401 501 601 701 801 901"}
mkdir -p "$OUT"

LIST=$(mktemp); MISSING=0
add() { if [ -e "$1" ]; then echo "$1" >> "$LIST"; else echo "  MISSING $1"; MISSING=1; fi; }

for S in $SEEDS; do
  for D in exps/mimicgen/il_ft_target_s160_seed$S \
           exps/mimicgen/square_il_ft_target_rotw1_seed$S \
           exps/mimicgen/coffee_il_ft_target_g5x_seed$S; do
    add "$D/model0.pt"
    add "$D/cfg.yaml"
  done
done
add release/cfgs/mimicgen/square_rotw1_ibrl_m001.yaml
add release/cfgs/mimicgen/coffee_g5x_ibrl_m001.yaml

if [ "$MISSING" != 0 ]; then
  echo
  echo "Some priors are not trained yet. Run:"
  echo "  setsid bash scripts/train_target_bc_priors.sh > bc_priors.log 2>&1 </dev/null & disown"
  echo "or pack a subset with SEEDS='1 101' bash cluster/pack_ibrl_priors.sh"
  rm -f "$LIST"; exit 1
fi

TAR=$OUT/ibrl_m001_priors.tar.gz
MAN=$OUT/ibrl_m001_priors.manifest.txt
echo "== manifest =="
: > "$MAN"
while read -r f; do
  printf '%s  %s\n' "$(sha256sum "$f" | cut -c1-16)" "$f" >> "$MAN"
done < "$LIST"
wc -l < "$MAN" | xargs echo "  files:"

cp "$MAN" .ibrl_m001_priors.manifest.txt
tar czf "$TAR" -T "$LIST" .ibrl_m001_priors.manifest.txt
rm -f .ibrl_m001_priors.manifest.txt "$LIST"

echo "== $TAR  ($(du -h "$TAR" | cut -f1)) =="
echo "  sha256: $(sha256sum "$TAR" | cut -d' ' -f1)"
echo
echo "Ship and unpack (paths inside are repo-relative, so untar AT the repo root):"
echo "  scp $TAR <user>@<host>:~/scratch/ibrl/"
echo "  ssh <user>@<host> 'cd ~/scratch/ibrl && tar xzf ibrl_m001_priors.tar.gz'"
echo
echo "Then verify on the far end before submitting:"
echo "  cd ~/scratch/ibrl && sha256sum -c <(awk '{print \$1\"  \"\$2}' .ibrl_m001_priors.manifest.txt) 2>/dev/null | grep -c OK"
echo "  # (the manifest holds 16-char prefixes; simplest real check is a file count)"
echo "  ls -d exps/mimicgen/{il_ft_target_s160,square_il_ft_target_rotw1,coffee_il_ft_target_g5x}_seed* | wc -l   # expect 30"
