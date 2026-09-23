#!/bin/bash
# CONTROL ARM for "does the +-5 deg tripod-yaw jitter cost the RL expert anything?"
#
# Paired with the live arm: the cell differs in layout_tripod_z_rot_deg and
# nothing else (asserted before launch), the demos were generated from that cell,
# and the prior + IBRL are rebuilt on them. Comparing against the live arm's IBRL
# therefore attributes any gap to the jitter alone.
#
# The demos already exist -- they are the batch generated before the jitter was
# added, archived rather than deleted. Only the interrupted conversion is redone.
#
#   setsid bash scripts/queue_realcell_v2_fixedyaw.sh > logs/queue_fixedyaw.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline
# The demos' env_name is Threading_RealCell, which re-reads the cell yaml on EVERY
# reset -- so the conversion and the BC eval must be pointed at the CONTROL cell or
# they would silently run the live (+-5) one.
export THREADING_REAL_CELL_YAML=/home/Chenyang/ibrl/release/cfgs/sim2real/threading_real_cell_fixedyaw.yaml
GPU=${GPU:-1}

GEN=release/data/mimicgen/generated/threading_realcell_fixedyaw
DEMO=$GEN/demo.hdf5
PROC=$GEN/processed_data96.hdf5
BC_DIR=exps/mimicgen/threading_realcell_fixedyaw_mlp_bc_seed1
RL_DIR=exps/mimicgen/threading_realcell_fixedyaw_ibrl_seed1
say() { echo "[fixedyaw $(date '+%F %T')] $*"; }
die() { say "ABORT: $*"; exit 1; }

$PY - <<'PY' || die "the control cell differs from the live one in more than the yaw"
import dataclasses, sys
sys.path.insert(0, "/home/Chenyang/ibrl")
from env.sim2real import load_sim2real
a = dataclasses.asdict(load_sim2real("release/cfgs/sim2real/threading_real_cell.yaml"))
b = dataclasses.asdict(load_sim2real("release/cfgs/sim2real/threading_real_cell_fixedyaw.yaml"))
d = [k for k in a if a[k] != b[k]]
assert d == ["layout_tripod_z_rot_deg"], d
print(f"[fixedyaw] A/B isolates one knob: {a[d[0]]} vs {b[d[0]]}")
PY

n=$($PY -c "import h5py;print(len(h5py.File('$DEMO','r')['data']))") || die "cannot read $DEMO"
[ "$n" -ge 1000 ] || die "$DEMO has $n demos"
say "demos: $n (generated before the jitter was added, same cell otherwise)"

if [ ! -f "$PROC" ]; then
  say "converting -> $PROC"
  $PY -u tools/dataset_states_to_obs.py --dataset "$DEMO" --output_folder "$GEN" --done_mode 2 \
    || die "conversion failed"
fi
n=$($PY -c "import h5py;print(len(h5py.File('$PROC','r')['data']))") || die "cannot read $PROC"
[ "$n" -ge 1000 ] || die "$PROC has $n demos"
say "converted: $n"

if [ ! -f "$BC_DIR/model0.pt" ]; then
  say "MLP BC prior on GPU $GPU -> $BC_DIR"
  # train_bc.py exits 1 even on success (memory: gotcha-train-bc-exits-nonzero)
  CUDA_VISIBLE_DEVICES=$GPU $PY -u train_bc.py \
    --config_path release/cfgs/sim2real/threading_realcell_fixedyaw_state_mlp_bc.yaml \
    --save_dir "$BC_DIR" || true
fi
[ -f "$BC_DIR/model0.pt" ] || die "no $BC_DIR/model0.pt"
grep -q "final eval | source" "$BC_DIR/train.log" || die "BC never reached its final eval"
say "BC done: $(grep 'final eval | source' "$BC_DIR/train.log" | tail -1 | grep -o 'score = [0-9.]*')"

say "IBRL on GPU $GPU -> $RL_DIR"
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_rl.py \
  --config_path release/cfgs/sim2real/threading_realcell_fixedyaw_state_ibrl.yaml \
  --save_dir "$RL_DIR" --snapshot_period 25000 || die "IBRL failed"
say "control arm finished"
