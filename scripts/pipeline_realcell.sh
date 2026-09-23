#!/bin/bash
# Real-cell pipeline, from MimicGen generation to IBRL:
#   generate -> convert -> (strict audit) -> MLP BC prior -> IBRL (sim2real_cell)
#
# Every stage checks its own output before the next starts, because each
# downstream stage fails SILENTLY on a bad input (an empty processed hdf5 loads 0
# episodes; a half-trained model0.pt is a valid-looking prior).
#
# Launch detached (iron law #11):
#   setsid bash scripts/pipeline_realcell.sh > exps/mimicgen/pipeline_realcell.log 2>&1 < /dev/null & disown
#   TASK=square setsid bash scripts/pipeline_realcell.sh > exps/mimicgen/pipeline_realcell_square.log 2>&1 < /dev/null & disown
set -euo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
# WANDB offline for BOTH trainers: online wandb has deadlocked long runs here
# (memory: gotcha-wandb-online-deadlock). Sync afterwards with `wandb sync`.
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline

# ---- task selection ---------------------------------------------------------
# TASK picks the whole column below. Everything downstream of `generate` is
# task-agnostic; only the strict-replay audit is not (strict success is a
# Threading/Coffee gate, docs/infra.md §5c, and the Square line runs without it
# -- auditing Square against it would report a meaningless 0/100).
TASK=${TASK:-threading}
case "$TASK" in
  threading)
    GEN=release/data/mimicgen/generated/threading_realcell
    NAME=threading_realcell
    STRICT_AUDIT=1 ;;
  square)
    # rotw1 (120-180 deg nut window); the env class Square_RealCell_RotW1
    # carries it, so it is not repeated in any config.
    GEN=release/data/mimicgen/generated/square_realcell_rotw1
    NAME=square_realcell
    STRICT_AUDIT=0 ;;
  *) echo "unknown TASK=$TASK (threading|square)"; exit 1 ;;
esac
DEMO=$GEN/demo.hdf5
PROC=$GEN/processed_data96.hdf5
BC_DIR=exps/mimicgen/${NAME}_mlp_bc_seed1
RL_DIR=exps/mimicgen/${NAME}_ibrl_seed1
BC_CFG=release/cfgs/sim2real/${NAME}_state_mlp_bc.yaml
RL_CFG=release/cfgs/sim2real/${NAME}_state_ibrl.yaml

# IBRL snapshot cadence (train_rl.py --snapshot_period): a quality spectrum of
# policies for SimDist data generation. 0 disables.
SNAP=${SNAP:-25000}
CELL=release/cfgs/sim2real/${TASK}_real_cell.yaml
NSHARD=${NSHARD:-10}
PER_SHARD=${PER_SHARD:-100}

stamp() { echo "[pipeline $(date '+%F %T')] $*"; }
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}

# ------------------------------------------------------------- 0. generate
# The cell yaml is re-read by Threading_RealCell on EVERY reset, so it must not
# be edited while this pipeline runs. Pin what the data was generated on.
if [ ! -f "$DEMO" ]; then
  mkdir -p "$GEN"
  # Snapshot the cell AND any file it `inherit`s -- square_real_cell.yaml is a
  # 2-line override of threading_real_cell.yaml, so hashing only the child
  # would let a camera re-calibration land mid-generation unrecorded.
  cell_files() { $PY -c "
import os, sys, yaml
p = os.path.abspath(sys.argv[1])
while p:
    print(p)
    par = (yaml.safe_load(open(p)) or {}).get('inherit')
    p = os.path.join(os.path.dirname(p), par) if par else None
" "$1"; }
  cell_files "$CELL" | xargs sha256sum > "$GEN/cell_snapshot.sha256"
  for f in $(cell_files "$CELL"); do cp "$f" "$GEN/$(basename "$f").snapshot"; done
  git rev-parse HEAD > "$GEN/git_head.txt"; git diff > "$GEN/git_diff.patch"
  stamp "generating $NSHARD shards x $PER_SHARD successes for TASK=$TASK"
  pids=()
  for i in $(seq 0 $((NSHARD - 1))); do
    $PY -u scripts/gen_realcell_demos.py --task "$TASK" --num "$PER_SHARD" --shard "$i" \
      > "$GEN/shard_$(printf %02d $i).log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { stamp "ERROR: a generation shard failed"; exit 1; }; done
  $PY scripts/merge_mimicgen_shards.py --root "$GEN" --out "$DEMO"
  sha256sum -c --status "$GEN/cell_snapshot.sha256" \
    || { stamp "ERROR: the cell (or a file it inherits) changed during generation"; exit 1; }
fi
N=$($PY -c "import h5py;print(len(h5py.File('$DEMO','r')['data']))")
[ "$N" -ge $((NSHARD * PER_SHARD)) ] || { stamp "ERROR: $DEMO has $N demos"; exit 1; }
stamp "generated: $N demos -> $DEMO"

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
if [ "$STRICT_AUDIT" = "1" ]; then
  stamp "strict-success replay audit (first 100 demos)"
  $PY scripts/check_demos_strict.py --demo "$DEMO" --n 100 2>&1 | grep -E "^demos="
else
  stamp "skipping strict audit (not defined for TASK=$TASK)"
fi

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
  --config_path "$RL_CFG" --save_dir "$RL_DIR" --snapshot_period "$SNAP"
stamp "IBRL finished"
