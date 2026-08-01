#!/usr/bin/env bash
# SGFT (Simulation-Guided Fine-Tuning, Yin et al. ICLR 2025) on top of IBRL,
# Threading_D0, TARGET domain. A frozen source-domain RL agent is reused as
# BOTH the IBRL BC prior AND the V_src for potential-based reward shaping
#     r' = r + sgft_coef * (gamma^n * V_src(s') - V_src(s)).
# Potential-based => the optimal target policy is unchanged; V_src only guides
# exploration. See CLAUDE.md §10. Only the target-domain fine-tune is run here;
# the source agent must already exist (Phase 0).
#
# ── Phase 0: source agent (no shift) ───────────────────────────────────────
# Train a source-domain RL agent first — its critic gives V_src and its policy
# is the IBRL BC prior. The source-RL script already does exactly this:
#     bash scripts/train_rl_source_threading.sh
#       -> exps/mimicgen/threading_rl_source_seed1/latest.pt
# (Any QAgent ckpt trained on the no-shift env with a comparable q_agent works.
#  If you enable strict_success on the target run, train the source with the
#  same flag so V_src reflects the same objective — optional, shaping stays
#  policy-invariant either way.)
#
# IMPORTANT — observation space must match: the frozen source agent consumes
# THIS run's obs (V_src + BC prior), so use_state / state_stack / obs_stack /
# prop_stack must equal the source's (train_rl asserts this with a clear error).
# The existing source agents (threading_ibrl_source_seed1, ...) used
# state_stack=1, which is what threading_d0_state_sgft.yaml is set to. If your
# source used a different stacking, override e.g. --state_stack N to match.
#
# ── Phase 1: SGFT fine-tune (this script) ──────────────────────────────────
# Usage:
#   bash scripts/train_rl_sgft_threading.sh
#   GPU=1 SEED=2 bash scripts/train_rl_sgft_threading.sh
#   SOURCE_AGENT=exps/.../latest.pt bash scripts/train_rl_sgft_threading.sh
#   bash scripts/train_rl_sgft_threading.sh --q_agent.sgft_coef 0.5 --num_train_step 400000
#
# Ablations (one config, three baselines):
#   # IBRL with the source as BC prior, NO value transfer (isolates shaping):
#   SGFT_SHAPING=0 METHOD=ibrl_srcprior bash scripts/train_rl_sgft_threading.sh
#   # vanilla IBRL (separate BC prior, no source agent) — use the IBRL script:
#   #   bash scripts/train_rl_dr_threading.sh ... --bc_policy <bc.pt>  (see that script)
#
# Env vars:
#   GPU           CUDA device id                    (default 0)
#   TASK          task tag for run name             (default threading)
#   METHOD        method tag for run name           (default sgft)
#   SEED          training seed                     (default 1)
#   SOURCE_AGENT  path to source QAgent ckpt        (default exps/mimicgen/threading_rl_source_seed${SEED}/latest.pt)
#   SGFT_SHAPING  1=value transfer on, 0=IBRL-only  (default 1)
#
# ── Eval the trained checkpoint (also: scripts/eval_rl_threading.sh) ───────
# Target domain (reproduce the YAML's eval shift) + strict-success gating:
#   python evaluate/eval.py --weight exps/mimicgen/threading_sgft_seed1/latest.pt \
#       --mode rl --num_game 100 --shift_from_cfg 1 --strict_success 1
# Source domain (no shift):
#   python evaluate/eval.py --weight exps/mimicgen/threading_sgft_seed1/latest.pt \
#       --mode rl --num_game 100 --shift_mode none --strict_success 1
# (eval auto re-loads the source agent as the BC prior from the run's cfg.yaml
#  `sgft_source_agent`, which act_method=ibrl needs.) Or simply:
#   bash scripts/eval_rl_threading.sh exps/mimicgen/threading_sgft_seed1/latest.pt
#   DOMAIN=both bash scripts/eval_rl_threading.sh exps/mimicgen/threading_sgft_seed1/latest.pt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh
source set_env.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

TASK="${TASK:-threading}"
METHOD="${METHOD:-sgft}"
SEED="${SEED:-1}"
SGFT_SHAPING="${SGFT_SHAPING:-1}"
SOURCE_AGENT="${SOURCE_AGENT:-exps/mimicgen/threading_rl_source_seed${SEED}/latest.pt}"

if [[ ! -f "$SOURCE_AGENT" ]]; then
  echo "ERROR: SGFT source agent checkpoint not found:" >&2
  echo "    $SOURCE_AGENT" >&2
  echo "Train one first (Phase 0):  bash scripts/train_rl_source_threading.sh" >&2
  echo "or pass SOURCE_AGENT=/path/to/source/latest.pt" >&2
  exit 1
fi

CONFIG="release/cfgs/mimicgen/threading_d0_state_sgft.yaml"
SAVE_DIR="exps/mimicgen/${TASK}_${METHOD}_seed${SEED}"

echo "SGFT source agent : $SOURCE_AGENT"
echo "sgft_shaping       : $SGFT_SHAPING  (1=value transfer on, 0=IBRL-only)"
echo "save_dir           : $SAVE_DIR"

python train_rl.py \
    --config_path "$CONFIG" \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --sgft_source_agent "$SOURCE_AGENT" \
    --q_agent.sgft_shaping "$SGFT_SHAPING" \
    "$@"
