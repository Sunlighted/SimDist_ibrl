#!/bin/bash
# Re-collect the target demos that commit da6f993 destroyed, for ONE task/seed.
#
# Every ckpt and shift value below is READ BACK from what survived, not guessed:
#   - the surviving demo dirs carry env_cfg.json with the exact `source_weight`
#     (e.g. square_sat_400000_s501 -> square_ibrl_target_B_rotw1_sig01_seed1/
#     snapshot_400000.pt), so `sat_<N>` is literally `snapshot_<N>.pt`
#   - the shift block in that same env_cfg.json is the collection-time domain
#     shift, and matches docs/infra.md 13
#
# ⚠️ collect_target_demos.py's --hole_offset (0.01) and --solref_scale (0.8)
# OVERRIDE the checkpoint unconditionally, so every knob is passed explicitly --
# omitting them silently collects against the OLD target (CLAUDE.md rule 13).
# ⚠️ MUJOCO_GL=osmesa: egl dies at context creation on this box even with
# --drop_images 1.
#
# ⚠️ COLLECTION sigma is PER TASK -- Threading 0.02, Square/Coffee 0.1 (read
# back from each surviving env_cfg.json). Verified by diffing a re-collected
# threading_d0_sat_160000_s1 against the original recovered from the cluster:
# everything matched (same snapshot, same shift, same 25 demos, |a| 0.3425 vs
# 0.3476) EXCEPT hand_init_perturb_joint_sigma, 0.1 vs the original 0.02 -- and
# the consequence was visible (length median 95 vs 88, sd 17.2 vs 11.7, +7%
# transitions). The `sig01` in the checkpoint name refers to the TRAINING/EVAL
# sigma; demos are collected at robosuite's 0.02 default, which is also the
# MimicGen demo distribution. Override with COLLECT_SIGMA if a task differs.
#
#   bash scripts/rebuild_demos.sh <task> <seed> [gpu]
#     task = threading160 | threading400 | square600 | square400 | coffee
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/Chenyang/.conda/envs/ibrl/bin/python
export PYTHONPATH=$PWD:${PYTHONPATH:-} MUJOCO_GL=osmesa
TASK=$1; SEED=$2; GPU=${3:-3}

case $TASK in
  threading160) CK=exps/mimicgen/threading_ibrl_newtgt_sig01_snap_seed1/snapshot_160000.pt
                OUT=release/data/mimicgen/threading_d0_sat_160000_s$SEED
                SHIFT="--hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0
                       --friction_scale 0.5 --gripper_kp_scale 0.12 --mass_scale 3.0
                       --ring_friction_scale 2.0 --ring_size_scale 0.85
                       --needle_size_scale 1.0" 
                SIGMA=0.02 ;;
  threading400) CK=exps/mimicgen/threading_ibrl_newtgt_sig01_snap_seed1/snapshot_400000.pt
                OUT=release/data/mimicgen/threading_d0_sat_400000_s$SEED
                SHIFT="--hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0
                       --friction_scale 0.5 --gripper_kp_scale 0.12 --mass_scale 3.0
                       --ring_friction_scale 2.0 --ring_size_scale 0.85
                       --needle_size_scale 1.0" 
                SIGMA=0.02 ;;
  square600)    CK=exps/mimicgen/square_ibrl_target_B_rotw1_sig01_seed1/snapshot_600000.pt
                OUT=release/data/mimicgen/square_sat_600000_s$SEED
                SHIFT="--hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0
                       --gripper_kp_scale 0.3 --peg_size_scale 1.2
                       --peg_friction_scale 2.0" 
                SIGMA=0.1 ;;
  square400)    CK=exps/mimicgen/square_ibrl_target_B_rotw1_sig01_seed1/snapshot_400000.pt
                OUT=release/data/mimicgen/square_sat_400000_s$SEED
                SHIFT="--hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0
                       --gripper_kp_scale 0.3 --peg_size_scale 1.2
                       --peg_friction_scale 2.0" 
                SIGMA=0.1 ;;
  coffee)       CK=exps/mimicgen/coffee_ibrl_target_G5x_seed1/snapshot_400000.pt
                OUT=release/data/mimicgen/coffee_g5x_sat400k_s$SEED
                SHIFT="--hole_offset 0.0 --solref_scale 1.0 --solimp_scale 1.0
                       --friction_scale 0.2 --gripper_kp_scale 0.3 --mass_scale 8.0
                       --lid_damping_scale 5.0 --pod_holder_friction_scale 5.0" 
                SIGMA=0.1 ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac
[ -f "$CK" ] || { echo "MISSING ckpt $CK"; exit 1; }
[ -f "$OUT/dataset.hdf5" ] && { echo "[skip] $OUT exists"; exit 0; }

CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/collect_target_demos.py \
  --weight "$CK" --mode rl --shift_mode fixed $SHIFT \
  --hand_init_perturb_joint_sigma "${COLLECT_SIGMA:-$SIGMA}" \
  --num_demo 25 --max_demo_len 400 \
  --interactive 0 --record_video 0 --drop_images 1 \
  --seed "$SEED" --out_dir "$OUT"
