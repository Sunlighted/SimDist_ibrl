#!/usr/bin/env bash
# Evaluate a trained RL / SGFT / IBRL checkpoint on Threading_D0, on the TARGET
# and/or SOURCE domain, with optional strict-success gating (anti reward-hack;
# see CLAUDE.md §5c). Works for any `--mode rl` ckpt.
#
# For act_method=ibrl ckpts (SGFT / IBRL), the BC prior is auto re-loaded from
# the run's cfg.yaml (`sgft_source_agent` or `bc_policy`) — no extra flags.
#
# Usage:
#   bash scripts/eval_rl_threading.sh <ckpt.pt>
#   GPU=1 NUM_GAME=200 bash scripts/eval_rl_threading.sh exps/mimicgen/threading_sgft_seed1/latest.pt
#   DOMAIN=source bash scripts/eval_rl_threading.sh <ckpt.pt>          # eval on source (no shift)
#   DOMAIN=both   bash scripts/eval_rl_threading.sh <ckpt.pt>          # target then source
#   STRICT=0      bash scripts/eval_rl_threading.sh <ckpt.pt>          # raw (un-gated) success
#   bash scripts/eval_rl_threading.sh <ckpt.pt> --shift_ring_size_scale 0.85   # extra shift overrides
#
# Comparing the raw vs strict success rate (STRICT=0 vs STRICT=1) on the SAME
# ckpt tells you how much of the success was reward-hacked (clip-through /
# shove-the-ring).
#
# Env vars:
#   GPU       CUDA device id                    (default 0)
#   NUM_GAME  episodes per eval                 (default 100)
#   DOMAIN    target | source | both            (default target)
#   STRICT    1=strict success gating, 0=raw    (default 1)
#   MP        multi-proc eval workers           (default 10)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

WEIGHT="${1:?usage: bash scripts/eval_rl_threading.sh <ckpt.pt> [extra evaluate/eval.py args]}"
shift || true

NUM_GAME="${NUM_GAME:-100}"
DOMAIN="${DOMAIN:-target}"
STRICT="${STRICT:-1}"
MP="${MP:-10}"

run_eval () {  # $1 = tag, $2.. = shift flags (+ any passthrough args)
  local tag="$1"; shift
  echo "================ eval | ${tag} (strict=${STRICT}, n=${NUM_GAME}) ================"
  python evaluate/eval.py \
      --weight "$WEIGHT" \
      --mode rl \
      --num_game "$NUM_GAME" \
      --mp "$MP" \
      --strict_success "$STRICT" \
      "$@"
}

case "$DOMAIN" in
  target) run_eval "target" --shift_from_cfg 1 "$@" ;;
  source) run_eval "source" --shift_mode none "$@" ;;
  both)
    run_eval "target" --shift_from_cfg 1 "$@"
    run_eval "source" --shift_mode none "$@"
    ;;
  *) echo "DOMAIN must be target|source|both, got '$DOMAIN'" >&2; exit 1 ;;
esac
