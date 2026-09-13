#!/bin/bash
# Full recovery chain, unattended:
#   1. wait for the threading s1 probe collection to finish
#   2. VALIDATE it: rerun the dynens_h5_slow_anchor recipe on the fresh demos and
#      compare against the surviving dynens_h5_slow_anchor_seed1 score
#   3. only if it validates, rebuild every remaining demo + the two derived sets
#
# Why the gate: re-collected demos are NOT the original ones (env rollouts are
# stochastic), and docs/demo-collection.md 38 measured +0.139 from a demo-standard
# change alone. If the new demos land somewhere else, every existing 10-seed
# baseline is void and the whole table has to be re-run -- so find out with ONE
# 2.5h adapt instead of after 100+ jobs.
#
# Gate thresholds, from the measured noise floors (CLAUDE.md rule 1):
#   |delta| < 0.06   same-config replicate sd is 0.027 -> accept
#   |delta| > 0.10   recipe is wrong (mode / snapshot) -> stop, do not rebuild
#   between          ambiguous -> stop and report, a human decides
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-}
LOG() { echo "[chain $(date -u +%H:%M:%S)] $*"; }

S1=release/data/mimicgen/threading_d0_sat_160000_s1
LOG "step 1: waiting for $S1"
for i in $(seq 1 120); do
  [ -f "$S1/env_cfg.json" ] && grep -q "snapshot_160000" "$S1/env_cfg.json" 2>/dev/null && \
    $PY -c "
import h5py,sys
try: sys.exit(0 if len(h5py.File('$S1/dataset.hdf5','r')['data'])>=25 else 1)
except Exception: sys.exit(1)" && break
  sleep 30
done
$PY -c "
import h5py,numpy as np
g=h5py.File('$S1/dataset.hdf5','r')['data']
L=[g[k]['actions'].shape[0] for k in g]
print(f'[chain] collected {len(L)} demos, len {min(L)}/{int(np.median(L))}/{max(L)}')
assert len(L)>=25" || { LOG "collection incomplete -- stop"; exit 1; }

# ---------------------------------------------------------------- step 2
VAL=exps/validate_newdemo/dynens_anchor_s1
LOG "step 2: validation adapt -> $VAL (~2.5h)"
if [ ! -f "$VAL/train.log" ]; then
  CUDA_VISIBLE_DEVICES=${VAL_GPU:-0} $PY -u train_simdist_adapt.py \
    --stage1_ckpt exps/mimicgen/threading_simdist_dynens_diversev2_h5_seed1/latest.pt \
    --save_dir "$VAL" --seed 1 \
    --target_demo_path $S1/dataset.hdf5 --target_num_data 20 \
    --bc_anchor_coef 1.0 \
    --bc_anchor_path release/data/mimicgen/threading_d0_source_diverse_v2/dataset.hdf5 \
    --max_transitions 100000 --adapt_interval 1000 --eval_every_rounds 5 \
    --num_eval_episode 100 --num_eval_proc 10 \
    --eval_disagree_coefs "0,0.01,0.05,0.1" \
    --eval_hand_init_perturb_joint_sigma 0.1 --eval_policy_only 1 \
    --collect_disagree_coef 0.5 --original_mppi 1 \
    --use_wb 0 --state_only_no_render 1 > "${VAL##*/}.log" 2>&1
fi

GATE=$($PY - <<'PY'
import re, numpy as np
def tail3(p):
    t=open(p,errors="replace").read()
    v=[float(m.group(1)) for m in re.finditer(r"^\d+: score/score\s+: ([0-9.]+)", t, re.M)]
    return np.mean(v[-3:]) if len(v)>=3 else None
new=tail3("exps/validate_newdemo/dynens_anchor_s1/train.log")
old=tail3("exps/simdist_threading_3arm/dynens_h5_slow_anchor_seed1/train.log")
if new is None or old is None:
    print("INCOMPLETE 0 0"); raise SystemExit
d=new-old
print(f"{'PASS' if abs(d)<0.06 else ('FAIL' if abs(d)>0.10 else 'AMBIGUOUS')} {new:.4f} {old:.4f} {d:+.4f}")
PY
)
LOG "step 2 result: $GATE"
set -- $GATE
[ "$1" = PASS ] || { LOG "gate=$1 -> NOT rebuilding. new=$2 old=$3 delta=$4"; exit 2; }

# ---------------------------------------------------------------- step 3
LOG "step 3: rebuilding all demos"
for s in 1 101 201 301 401 501 601 701 801 901; do
  for t in threading160 threading400 square600 square400 coffee; do
    bash scripts/rebuild_demos.sh $t $s ${COLLECT_GPU:-3} >> rebuild_demos_all.log 2>&1 \
      && LOG "  ok $t s$s" || LOG "  FAIL $t s$s"
  done
done
LOG "step 3b: derived sets"
SEEDS="1 101 201 301 401 501 601 701 801 901" bash scripts/prep_coffee_baselines.sh rlpd \
  >> rebuild_demos_all.log 2>&1 && LOG "  ok coffee_rlpd_prior"
$PY scripts/extract_anchor_pool.py \
  release/data/mimicgen/square_d0_source_rotw1_diverse/dataset.hdf5 \
  release/data/mimicgen/square_anchor_pool_4k/dataset.hdf5 --episodes 4000 \
  >> rebuild_demos_all.log 2>&1 && LOG "  ok square_anchor_pool_4k"
LOG "CHAIN DONE"
