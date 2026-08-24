#!/bin/bash
# One-time environment build on a Compute Canada / Alliance cluster.
# RUN THIS ON A LOGIN NODE -- compute nodes have no internet.
#
# We build a plain venv rather than using CC's --no-index wheelhouse, because
# robosuite / robomimic / mimicgen are not in it and have to come from the
# vendored submodules anyway. Mirrors docs/setup.md 1 with two changes:
#   * torch from CC's wheelhouse (their build matches the cluster's CUDA);
#     the cu128 pin in setup.md is Blackwell-specific and wrong here.
#   * no conda -- CC discourages it and venv is enough.
#
# Usage (from the repo root on the cluster):
#   bash cluster/cc_setup.sh
set -e
cd "$(dirname "$0")/.."

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2 mujoco/3.1.6 2>/dev/null \
  || module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2

VENV=${VENV:-$HOME/venvs/ibrl}
[ -d $VENV ] || python -m venv $VENV
source $VENV/bin/activate
python -m pip install --upgrade pip

# torch from CC's wheelhouse (matches the cluster CUDA); --no-index is required
# for their prebuilt wheels.
python -m pip install --no-index torch torchvision

# robosuite needs mujoco 2.3.2 (MimicGen requirement) and numpy<2 (wandb 0.16.4
# uses the removed np.float_). Both pins are load-bearing -- see docs/setup.md.
python -m pip install "numpy<2" "mujoco==2.3.2"
python -m pip install -e deps/robosuite -e deps/robomimic -e deps/mimicgen
python -m pip install Cython==0.29.36 einops==0.7.0 fasteners==0.19 pyrallis==0.3.1 \
  tabulate==0.9.0 typing-inspect==0.9.0 wandb==0.16.4 cloudpickle "gym==0.26.2" \
  "huggingface_hub==0.23.4" h5py

# C++ extension (common_utils/rela). Needs a compiler -- hence gcc above.
( cd common_utils && make -j8 )

echo
echo "=== smoke test ==="
export PYTHONPATH=$PWD:$PYTHONPATH MUJOCO_GL=egl
python -c "
import torch, robosuite, mujoco
from env.robosuite_wrapper import STATE_KEYS
print(' torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
print(' mujoco', mujoco.__version__, '| robosuite', robosuite.__version__)
"
echo
echo "SETUP DONE. Note MUJOCO_GL=egl is REQUIRED: train_rl.py does not expose"
echo "state_only_no_render, so robosuite builds an offscreen GL context even for"
echo "state-only runs. Verify EGL works on a GPU node before submitting 40 jobs:"
echo "  salloc --account=\$CC_ACCOUNT --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=0:20:00"
echo "  source $VENV/bin/activate && MUJOCO_GL=egl python -c \\"
echo "    'from env.robosuite_wrapper import PixelRobosuite as P;"
echo "     e=P(env_name=\"Square_D0\",robots=[\"Panda\"],use_state=True,rl_cameras=[],camera_names=[]);"
echo "     print(e.reset()[0][\"state\"].shape)'"
