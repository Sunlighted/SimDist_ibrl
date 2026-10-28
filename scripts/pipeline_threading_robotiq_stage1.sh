#!/bin/bash
# TRAIN_PLAN.md (calib_20261015) stage 3 gate -> stage 4 on the exec_obs line (2026-10-25):
#   1. gate: independent 300-episode evals of the exec_obs IBRL snapshots (both seeds, 400k, 500k, ...) until one
#      reaches >= GATE; the first step at which a seed passes wins (the better seed at that step);
#   2. freeze: copy that snapshot (expert.pt), the earlier snapshots of the same run (the sub-optimal policies of
#      the data mix) and cfg.yaml into FROZEN, with sha256 sums (CLAUDE.md #14: never collect from model0.pt);
#   3. pixel data: scripts/collect_threading_realcell_batches.sh, both cameras, 8 x 4500 trajectories;
#   4. SimDist stage 1 (threading_robotiq_pixel_simdist_r18_224.yaml), the replay cap from the real shards (#33);
#   5. stage-4 gate: independent 300-episode MPPI eval of the stage-1 model0.pt.
# Every step skips itself when its product exists, so a re-launch resumes.
#   setsid bash scripts/pipeline_threading_robotiq_stage1.sh > logs/pipeline_robotiq_stage1.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
source set_env.sh
GPU="${GPU:-3}"
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=offline
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
RUNS="${RUNS:-exps/sim2real/ibrl_robotiq_v5_2M_fr_chunk1_execobs_seed1 exps/sim2real/ibrl_robotiq_v5_2M_fr_chunk1_execobs_seed2}"
STEPS="${STEPS:-400000 500000 600000 700000 800000 900000 1000000}"
GATE="${GATE:-0.9}"
FROZEN="${FROZEN:-exps/sim2real/threading_robotiq_execobs_expert}"
CELL="${CELL:-real2sim/zed_capture/calib_20261015/threading_real_cell_20261015.yaml}"
DEST="${DEST:-/user_data/Chenyang/simdist_data/threading_robotiq_pixel_execobs_v1}"
PER_SHARD="${PER_SHARD:-4500}"
CFG=release/cfgs/sim2real/threading_robotiq_pixel_simdist_r18_224.yaml
S1="${S1:-exps/sim2real/threading_robotiq_pixel_stage1_seed1}"
RAM_BUDGET_GB="${RAM_BUDGET_GB:-170}"      # resident replay; the machine has 377 GB, shared with the IBRL runs
stamp() { echo "[pipe $(date -u '+%F %T')] $*"; }
score_of() { grep -E '^score: ' "$1" 2>/dev/null | tail -1 | sed -E 's/^score: ([0-9.]+).*/\1/'; }

# ---- 1. gate ------------------------------------------------------------------------------------------------------
if [ ! -s "$FROZEN/expert.pt" ]; then
  best=""; best_s=0
  for step in $STEPS; do
    for r in $RUNS; do         # wait for the snapshot, and for it to be fully written
      until [ -s "$r/snapshot_${step}.pt" ] && [ $(( $(date +%s) - $(stat -c %Y "$r/snapshot_${step}.pt") )) -gt 60 ]; do
        sleep 300
      done
    done
    stamp "gate @ ${step}: independent 300-episode evals"
    for r in $RUNS; do
      log="$r/gate_eval300_${step}.log"
      [ -n "$(score_of "$log")" ] && continue
      $PY evaluate/eval.py --weight "$r/snapshot_${step}.pt" --mode rl --num_game 300 --mp 10 --strict_success 1 \
          --hand_init_perturb_joint_sigma 0.1 > "$log" 2>&1 &
    done
    wait
    for r in $RUNS; do
      s=$(score_of "$r/gate_eval300_${step}.log"); stamp "  $r @ ${step}: ${s:-FAILED} | $(grep -m1 'EVAL ENV' "$r/gate_eval300_${step}.log")"
      if [ -n "$s" ] && awk -v a="$s" -v b="$best_s" 'BEGIN{exit !(a>b)}'; then best_s=$s; best="$r/snapshot_${step}.pt"; fi
    done
    if awk -v a="$best_s" -v g="$GATE" 'BEGIN{exit !(a>=g)}'; then break; fi
    best=""; best_s=0
  done
  [ -n "$best" ] || { stamp "GATE FAILED: no snapshot in [$STEPS] reached $GATE -- stopping"; exit 1; }
  stamp "gate passed: $best = $best_s"
  # ---- 2. freeze ----------------------------------------------------------------------------------------------------
  run=$(dirname "$best"); bstep=$(basename "$best" .pt | sed 's/snapshot_//')
  mkdir -p "$FROZEN"
  cp "$run/cfg.yaml" "$FROZEN/cfg.yaml"
  for f in "$run"/snapshot_*.pt; do
    st=$(basename "$f" .pt | sed 's/snapshot_//'); [ "$st" -lt "$bstep" ] && cp "$f" "$FROZEN/"
  done
  cp "$best" "$FROZEN/expert.pt.tmp" && mv "$FROZEN/expert.pt.tmp" "$FROZEN/expert.pt"
  (cd "$FROZEN" && sha256sum cfg.yaml expert.pt snapshot_*.pt > SHA256SUMS)
  printf 'expert = %s (independent 300-episode strict eval %s, gate %s)\nsub-optimal = the earlier snapshots of %s\nfrozen %s, git %s\n' \
    "$best" "$best_s" "$GATE" "$run" "$(date -u '+%F %T')" "$(git rev-parse --short HEAD)" > "$FROZEN/PROVENANCE.txt"
  for r in $RUNS; do grep -H -E '^score: ' "$r"/gate_eval300_*.log >> "$FROZEN/PROVENANCE.txt"; done
fi
(cd "$FROZEN" && sha256sum -c --quiet SHA256SUMS) || { stamp "ERROR: $FROZEN changed since it was frozen"; exit 1; }
stamp "expert: $(head -1 "$FROZEN/PROVENANCE.txt")"

# ---- 3. pixel data ------------------------------------------------------------------------------------------------
if [ "$(ls "$DEST"/part*.hdf5 2>/dev/null | wc -l)" -lt 8 ]; then
  stamp "collecting pixel data -> $DEST"
  GPU="$GPU" NUM_PROC="${NUM_PROC:-16}" PER_WORKER_MIB="${PER_WORKER_MIB:-3300}" GUARD_FRAC=0.75 \
    BATCH_TRAJ="$PER_SHARD" BATCH_IDS="1 2 3 4 5 6 7 8" SEED_BASE=100 \
    RUN="$FROZEN" EXPERT="$FROZEN/expert.pt" CELL="$CELL" DEST="$DEST" \
    CAMS=agentview+robot0_eye_in_hand RL_IMAGE_SIZE=224 JPEG_Q=90 STORE_STATE=1 LOG_TAG=robotiq_pix_v1 \
    bash scripts/collect_threading_realcell_batches.sh
fi
# open every part (file size is not evidence, docs/infra.md §14e) and size the replay from the real shards (#33)
read -r nper cap gb <<< "$($PY - "$DEST" "$PER_SHARD" "$RAM_BUDGET_GB" <<'PYEOF'
import glob, h5py, os, sys
d, per, budget = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
parts = sorted(glob.glob(os.path.join(d, "part*.hdf5")))
assert len(parts) == 8, f"expected 8 parts, found {len(parts)}"
lens, nbytes, ntr = [], 0, 0
for p in parts:
    with h5py.File(p, "r") as f:
        g = f["data"]; keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))
        assert len(keys) >= per, f"{p} has {len(keys)} < {per}"
        for k in keys[:per]:
            e = g[k]
            assert all(x in e["obs"] for x in ("agentview_image", "robot0_eye_in_hand_image", "state")), (p, k)
            assert e["obs/state"].shape[1] == 64, (p, k, e["obs/state"].shape)
            assert "sim_state" in e and "model_state" in e, (p, k)
            lens.append(e["actions"].shape[0])
        nbytes += os.path.getsize(p)                  # whole file / all its transitions: an upper bound on the
        ntr += sum(g[k]["actions"].shape[0] for k in keys)   # resident JPEG bytes (sim_state is not loaded)
bpt = nbytes / ntr
tot = sum(lens)
if tot * bpt / 1e9 > budget:                          # keep every shard, fewer episodes each
    per = int(per * budget * 1e9 / (tot * bpt))
    tot = int(tot * per / int(sys.argv[2]))
print(f"[pipe] {len(lens)} episodes, mean length {sum(lens)/len(lens):.1f}, {bpt:.0f} B/transition", file=sys.stderr)
print(per, int(tot * 1.02) + 1000, f"{tot * bpt / 1e9:.0f}")
PYEOF
)" || { stamp "ERROR: shard validation failed"; exit 1; }
stamp "replay: $nper episodes/shard, cap $cap transitions, ~${gb} GB resident"

# ---- 4. SimDist stage 1 -------------------------------------------------------------------------------------------
if ! grep -q "distill step 150000\|^150000: score" "$S1/train.log" 2>/dev/null; then
  stamp "stage 1 -> $S1"
  $PY train_simdist.py --config_path "$CFG" --source_agent "$FROZEN/expert.pt" --save_dir "$S1" \
      --source_replay_path "$DEST/part*.hdf5" --source_replay_num_data "$nper" --seq_replay_max_transitions "$cap" \
      --sim2real_cell "$CELL" --use_wb 0
  stamp "train_simdist rc=$?"
fi
[ -s "$S1/model0.pt" ] || { stamp "FATAL: no $S1/model0.pt"; exit 1; }

# ---- 5. stage-4 gate: MPPI ------------------------------------------------------------------------------------------
$PY evaluate/eval.py --weight "$S1/model0.pt" --mode simdist --use_planner 1 --num_game 300 --mp 10 \
    --strict_success 1 --hand_init_perturb_joint_sigma 0.1 > "$S1/eval300_mppi.log" 2>&1
$PY evaluate/eval.py --weight "$S1/model0.pt" --mode simdist --use_planner 0 --num_game 300 --mp 10 \
    --strict_success 1 --hand_init_perturb_joint_sigma 0.1 > "$S1/eval300_policy.log" 2>&1
for f in mppi policy; do stamp "stage-1 $f: $(score_of "$S1/eval300_$f.log") | $(grep -m1 'EVAL ENV' "$S1/eval300_$f.log")"; done
stamp "done"
