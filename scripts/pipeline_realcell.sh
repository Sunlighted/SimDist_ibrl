#!/bin/bash
# Real-cell Threading pipeline, from merged MimicGen demos to IBRL:
#   convert -> strict-replay audit -> MLP BC prior -> IBRL (sim2real_cell)
#
# Every stage checks its own output before the next starts, because each
# downstream stage fails SILENTLY on a bad input (an empty processed hdf5 loads 0
# episodes; a half-trained model0.pt is a valid-looking prior).
#
# Launch detached (iron law #11):
#   setsid bash scripts/pipeline_realcell.sh > exps/mimicgen/pipeline_realcell.log 2>&1 < /dev/null & disown
set -euo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
# WANDB offline for BOTH trainers: online wandb has deadlocked long runs here
# (memory: gotcha-wandb-online-deadlock). Sync afterwards with `wandb sync`.
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline

GEN=release/data/mimicgen/generated/threading_realcell
DEMO=$GEN/demo.hdf5
PROC=$GEN/processed_data96.hdf5
BC_DIR=exps/mimicgen/threading_realcell_mlp_bc_seed1
RL_DIR=exps/mimicgen/threading_realcell_ibrl_seed1
BC_CFG=release/cfgs/sim2real/threading_realcell_state_mlp_bc.yaml
RL_CFG=release/cfgs/sim2real/threading_realcell_state_ibrl.yaml

stamp() { echo "[pipeline $(date '+%F %T')] $*"; }
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}

# ---------------------------------------------------------------- 1. convert
if [ -f "$PROC" ]; then
  N=$($PY -c "import h5py;print(len(h5py.File('$PROC','r')['data']))" 2>/dev/null || echo 0)
  # memory: a crashed conversion leaves an EMPTY processed file that would be
  # silently reused; treat anything short of the full set as absent
  [ "$N" -ge 1000 ] || { stamp "stale $PROC ($N demos) -> removing"; rm -f "$PROC"; }
fi
if [ ! -f "$PROC" ]; then
  stamp "converting $DEMO"
  $PY -u tools/dataset_states_to_obs.py --dataset "$DEMO" --output_folder "$GEN" --done_mode 2
fi
N=$($PY -c "import h5py;print(len(h5py.File('$PROC','r')['data']))")
[ "$N" -ge 1000 ] || { stamp "ERROR: $PROC has $N demos"; exit 1; }
stamp "converted: $N demos -> $PROC"

# ---------------------------------------------------------- 2. strict audit
stamp "strict-success replay audit (first 100 demos)"
$PY scripts/check_demos_strict.py --demo "$DEMO" --n 100 2>&1 | grep -E "^demos="

# ------------------------------------------------------------- 3. flow BC
if [ ! -f "$BC_DIR/.done" ]; then
  GPU=$(pick_gpu); stamp "training MLP BC on GPU $GPU -> $BC_DIR"
  # train_bc.py ends with `assert False` and exits 1 even on success (memory:
  # gotcha-train-bc-exits-nonzero), so judge it by its artefacts, not its rc
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_bc.py --config_path "$BC_CFG" --save_dir "$BC_DIR" || true
  [ -f "$BC_DIR/model0.pt" ] || { stamp "ERROR: no $BC_DIR/model0.pt"; exit 1; }
  grep -q "final eval | source" "$BC_DIR/train.log" 2>/dev/null \
    || { stamp "ERROR: BC did not reach its final eval (see $BC_DIR/train.log)"; exit 1; }
  touch "$BC_DIR/.done"
fi
grep -E "final eval \| source" "$BC_DIR/train.log" | tail -1 || true
# iron law #14: TopkSaver rewrites model0.pt -- pin exactly which prior IBRL used
sha256sum "$BC_DIR/model0.pt" | tee "$BC_DIR/model0.sha256"

# ------------------------------------------------------------------ 4. IBRL
GPU=$(pick_gpu); stamp "training IBRL on GPU $GPU -> $RL_DIR"
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_rl.py \
  --config_path "$RL_CFG" --save_dir "$RL_DIR"
stamp "IBRL finished"
