#!/bin/bash
# Stage-1 distillation, paired A/B: base (random shift only) vs full augmentation.
# The two configs differ in the augmentation block and NOTHING else -- same data,
# same seed, same caps -- so the delta isolates augmentation.
#
#   setsid bash scripts/launch_stage1_v2.sh > logs/stage1_v2.log 2>&1 </dev/null & disown
set -uo pipefail
cd /home/Chenyang/ibrl
source set_env.sh >/dev/null 2>&1
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_MODE=offline
EXPERT=exps/mimicgen/threading_realcell_ibrl_seed1/model0.pt
say() { echo "[stage1 $(date '+%F %T')] $*"; }

# The two configs must differ ONLY in the augmentation knobs; anything else is a
# confound that would be attributed to augmentation.
$PY - <<'PY' || { say "ABORT: the arms differ outside the augmentation block"; exit 1; }
import yaml, sys
a = yaml.safe_load(open("release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_v2.yaml"))
b = yaml.safe_load(open("release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_aug_v2.yaml"))
def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else out.setdefault(f"{p}{k}", v)
    return out
fa, fb = flat(a), flat(b)
diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
allowed = {"wm.aug_prop_noise", "wm.aug_color_jitter", "wm.aug_blur_sigma_max", "wm.aug_shift_pad"}
extra = [k for k in diff if not any(k.endswith(a2.split(".")[-1]) for a2 in allowed)]
print(f"[stage1] arms differ in: {diff}")
sys.exit(1 if extra else 0)
PY

say "GPU0 base -> exps/mimicgen/threading_realcell_pixel_stage1_v2_seed1"
CUDA_VISIBLE_DEVICES=0 setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_v2.yaml \
  --source_agent "$EXPERT" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_seed1 \
  > logs/stage1_v2_base.log 2>&1 </dev/null &
A=$!
sleep 60      # stagger: both arms mmap ~390 GB of shards at startup
say "GPU1 aug  -> exps/mimicgen/threading_realcell_pixel_stage1_v2_aug_seed1"
CUDA_VISIBLE_DEVICES=1 setsid $PY -u train_simdist.py \
  --config_path release/cfgs/sim2real/threading_realcell_pixel_simdist_r18_224_aug_v2.yaml \
  --source_agent "$EXPERT" \
  --save_dir exps/mimicgen/threading_realcell_pixel_stage1_v2_aug_seed1 \
  > logs/stage1_v2_aug.log 2>&1 </dev/null &
B=$!
wait $A; rcA=$?
wait $B; rcB=$?
say "base rc=$rcA  aug rc=$rcB"
