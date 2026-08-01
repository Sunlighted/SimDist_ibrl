#!/bin/bash
# Wait for the 1000-demo collection to finish, verify the dataset, then re-run
# the policy-finetune + 4-way eval experiment with N_DEMO=1000.
set -e
cd /home/Chenyang/ibrl
DS=release/data/mimicgen/threading_d0_target_expert_n1000/dataset.hdf5

echo "[chain] waiting for collect_target_demos.py to finish..."
while pgrep -f "collect_target_demos.py" >/dev/null; do sleep 20; done
echo "[chain] collection process exited."

source /home/Chenyang/anaconda3/etc/profile.d/conda.sh && conda activate ibrl
source set_env.sh >/dev/null 2>&1

# verify episode count
NEP=$(python - <<EOF
import h5py
try:
    f=h5py.File("$DS","r"); print(len(f["data"].keys()))
except Exception as e:
    print("0")
EOF
)
echo "[chain] dataset episodes = $NEP"
if [ "$NEP" -lt 900 ]; then
    echo "[chain] ABORT: fewer than 900 demos collected ($NEP); not running eval."
    exit 1
fi

echo "[chain] launching 1000-demo experiment on GPU2..."
CUDA_VISIBLE_DEVICES=2 N_DEMO=1000 LABEL="1000-demo" \
  DEMOS="$DS" \
  SAVE_DIR=exps/mimicgen/threading_simdist_adapt_ibrlmppi_400k_bc1000_seed1 \
  python scripts/finetune_policy500_eval.py
