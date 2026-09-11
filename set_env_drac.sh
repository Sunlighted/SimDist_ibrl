# Environment for the Alliance/DRAC SLURM cluster (klogin*/gpubase_*).
#
# The checked-in set_env.sh targets the original workstation: it assumes a conda
# env and exports MUJOCO_PY_MUJOCO_PATH for mujoco-py/metaworld, which this
# project deliberately does not install (docs/setup.md "Why diverge"). It also
# does NOT set MUJOCO_GL, despite CLAUDE.md's claim -- every entry point sets it
# itself (evaluate/eval.py:102, train_simdist.py:592). We set it here so ad-hoc
# scripts and interactive sessions get offscreen rendering too.
#
# Usage:  cd <repo> && source set_env_drac.sh

module load python/3.10.13 >/dev/null 2>&1
source ~/venvs/ibrl/bin/activate

# --- CUDA -----------------------------------------------------------------
# 12.2 matches what the DRAC torch 2.6.0 wheel reports (torch.version.cuda).
# Needed to BUILD common_utils/rela: find_package(Torch) chains into Caffe2's
# config, which errors "Caffe2 version uses CUDA but I cannot find the CUDA
# libraries" without it. `module load cuda/12.2` does NOT work standalone --
# the modulefile sits under the MPI/gcc12/openmpi4 layer, so loading it without
# gcc+openmpi silently no-ops (exactly like protobuf below). Point at the
# cudacore install directly, which is what EBROOTCUDA resolves to anyway.
_CU=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
if [ -d "$_CU" ]; then
  export CUDA_HOME=$_CU CUDA_TOOLKIT_ROOT_DIR=$_CU CUDAToolkit_ROOT=$_CU
  export PATH=$_CU/bin:$PATH
  export CMAKE_PREFIX_PATH=$_CU:${CMAKE_PREFIX_PATH:-}
  export LD_LIBRARY_PATH=$_CU/lib64:${LD_LIBRARY_PATH:-}
fi

export PYTHONPATH=$PWD:${PYTHONPATH:-}

# --- opencv: module-only on this cluster ---------------------------------
# There is no opencv wheel in the Compute Canada wheelhouse (only a dummy
# `opencv-noinstall` package that errors on purpose), and PyPI manylinux wheels
# are not compatible with the DRAC python, so `pip install opencv-python` builds
# from source and fails. cv2 therefore comes from the Lmod module.
# `module load opencv/...` alone does NOT reach into a --no-download virtualenv
# (it relies on a sitecustomize hook the venv does not inherit), so inject the
# path explicitly. NOTE: only opencv <= 4.10.0 ships python3.10 bindings; 4.11+
# start at python3.11. Keep this version aligned with the venv's python.
_OCV=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/opencv/4.10.0/lib/python3.10/site-packages
[ -d "$_OCV" ] && export PYTHONPATH=$PYTHONPATH:$_OCV

# --- mujoco 2.3.2 -------------------------------------------------------
# How it got installed (see docs/setup.md DRAC section): `avail_wheels mujoco`
# is EMPTY -- there is no Compute Canada wheel at any version -- and DRAC's pip
# is patched to reject manylinux tags ("not a supported wheel on this
# platform"), so neither `pip install mujoco` nor the pre-download flow works.
# Building from source fails too: mujoco 2.3.2 + gcc 12 gives an undefined
# `absl::lts_20230125::container_internal::kEmptyGroup`. The route that works is
# unzipping the official PyPI manylinux wheel straight into site-packages; it
# bundles its own libmujoco.so, so no MUJOCO_PATH is needed at runtime, and it
# brings its own dist-info so pip still sees mujoco as installed.
#
# 2.3.2 exactly (not the 2.3.6 Lmod module) is deliberate: this project's
# results turn on contact physics at a ~1.8 mm tolerance window, so a patch
# version is not a safe substitution when the state-mode control arm has to be
# comparable to the reference numbers.
#
# MUJOCO_GL=egl above is REQUIRED, not just preferred: mujoco's __init__ pulls
# in the glfw backend otherwise, and the venv's glfw has no GLFW3 shared
# library ("Failed to load GLFW3 shared library"). EGL sidesteps it entirely.

# Offscreen GPU rendering. If MuJoCo falls back to OSMesa (software GL) this is
# ~10x slower and silently wrecks every pixel-mode timing estimate --
# scripts/bench_render.py is the check.
export MUJOCO_GL=egl

# --- protobuf (for the common_utils/rela C++ extension) -----------------
# `find_package(Torch)` chains into Caffe2's config, which does
# `find_package(Protobuf)` -- the DRAC torch wheel bundles no libprotobuf, so
# `make` in common_utils/ dies with "Caffe2 version uses protobuf but the
# protobuf library cannot be found". The protobuf/24.4 module supplies it, but
# it lives under the gcccore compiler layer and `module load protobuf` alone
# silently no-ops, so point at it explicitly. Needed at BUILD time
# (CMAKE_PREFIX_PATH) and at RUN time (LD_LIBRARY_PATH, since rela links it).
# Abseil is required TOO: the only protobuf in StdEnv/2023 is 24.4 (i.e. 4.24),
# whose CMake targets declare a link interface on `absl::absl_check`. Without
# abseil on CMAKE_PREFIX_PATH the generate step fails with
# 'The link interface of target "protobuf::libprotobuf" contains absl::absl_check
# but the target was not found'. (protobuf 3.x, which Caffe2 was written
# against, only exists in the older StdEnv trees.)
_B=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3
_PROTO=$_B/Compiler/gcccore/protobuf/24.4
_ABSL=$_B/Compiler/gcccore/abseil/20230125.3
if [ -d "$_PROTO" ]; then
  export CMAKE_PREFIX_PATH=$_PROTO:$_ABSL:${CMAKE_PREFIX_PATH:-}
  export LD_LIBRARY_PATH=$_PROTO/lib:$_PROTO/lib64:$_ABSL/lib:$_ABSL/lib64:${LD_LIBRARY_PATH:-}
fi

# --- wandb -----------------------------------------------------------------
# Offline by default: compute nodes have no egress, and an online run blocks on
# auth. Runs are recorded locally and can be pushed later with `wandb sync`.
# NOTE the version tension: the repo pins wandb==0.16.4 while the cluster
# guidance is 0.18.0 (0.19.x hits an Alliance bug, wandb#8966). 0.16.4 predates
# that bug, so it is left in place; if logging misbehaves, move to 0.18.0 rather
# than the wheelhouse default.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-/scratch/yifanr/simdist/wandb}
mkdir -p "$WANDB_DIR" 2>/dev/null || true

# make multi-process eval work (mirrors set_env.sh)
export OMP_NUM_THREADS=1

# 10 eval workers each ship uint8 frames through mp.Queue, one shared-memory fd
# per tensor -- the default soft limit runs out and surfaces as an opaque
# "too many open files".
ulimit -n 65536 2>/dev/null || true

echo "ibrl env: $(python --version 2>&1), torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "  MUJOCO_GL=$MUJOCO_GL  PYTHONPATH=$PWD"
echo "  data -> $(readlink -f release/data 2>/dev/null)"
