#!/bin/bash
# Build the Apptainer image. Run from the repo root; needs cluster/ibrl.def and
# the deps/ submodules populated (git submodule update --init --recursive).
#
# ~20 min, ~7 GB output. Uses --fakeroot (verified working here: the user has a
# /etc/subuid mapping, so no sudo is needed).
set -e
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/user_data/Chenyang/apptainer_tmp}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/user_data/Chenyang/apptainer_cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$(dirname "$0")/.."
for d in deps/robosuite deps/robomimic deps/mimicgen; do
  [ -f $d/setup.py ] || [ -f $d/pyproject.toml ] || {
    echo "ERROR: $d looks empty -- run: git submodule update --init --recursive"; exit 1; }
done
# Default output lives OUTSIDE the repo: 12 GB does not belong on the repo
# filesystem, and it must never be committed.
OUT=${1:-/user_data/Chenyang/ibrl.sif}
apptainer build --fakeroot "$OUT" cluster/ibrl.def
echo
ls -lh "$OUT"
echo "--- smoke test ---"
apptainer exec "$OUT" python -c "
import torch, robosuite, mujoco
print(' torch', torch.__version__, '| arch', torch.cuda.get_arch_list())
print(' mujoco', mujoco.__version__, '| robosuite', robosuite.__version__)"
