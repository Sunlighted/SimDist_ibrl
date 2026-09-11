#!/bin/bash
# Follow-on for the r18@224 plain_h3 stage-1 chain (rung A of the pixel ladder).
#
# WHY THIS EXISTS: stage-1 self-chunks (pixel_stage1.slrm re-submits itself with
# --dependency=afterany until global_step reaches STEPS), so there is no single
# job id to hang the stage-2 dependency on at submit time -- chunk 1 is not the
# last chunk, and the last chunk's id does not exist until chunk N-1 runs. So the
# chain calls THIS script from its completion hook (THEN_SBATCH), and this script
# submits stage-2/3 and then chains the headline eval behind it.
#
# It is invoked ON A COMPUTE NODE by the final stage-1 chunk. Keep it to sbatch
# calls only -- no GPU work, no waiting.
#
# The three rungs must be adapted at an IDENTICAL budget or they are not
# comparable (cmp_adapt_runs.py refuses to compare across budgets). Rung C
# (already done, 0.766) ran MAX_TRANS=100000 / INTERVAL=2000 / EVAL_N=100 and
# reached the full 100k in 51 eval points; these values reproduce that.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/project/6101830/yifanr/SimDist_ibrl}"

STAGE1_DIR=exps/mimicgen/threading_pix224_plainh3_stage1_seed1
ADAPT_DIR=exps/mimicgen/pix224_plainh3_adapt_seed1

# PROPAGATE THE ACCOUNT. adapt.slrm and eval_simdist.slrm both hardcode
# `#SBATCH --account=aip-gigor`, so without an explicit flag this follow-on would
# silently land back on gigor even though the stage-1 chain that called it was
# running on aip-rhinehar (fairshare 0.657 vs 0.106 => ~6x the priority factor).
# Same class of bug as the --mem fallback that OOM'd job 5212432: --export
# carries env vars, never sbatch resource flags.
ACCT="${SLURM_JOB_ACCOUNT:-aip-gigor}"
echo "submitting follow-on under account: $ACCT"

[ -s "$STAGE1_DIR/latest.pt" ] || { echo "FATAL: no $STAGE1_DIR/latest.pt"; exit 1; }

ADAPT_ID=$(ARM=pix224 \
  STAGE1="$STAGE1_DIR/latest.pt" \
  DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5 \
  ANCHOR=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5 \
  SAVE="$ADAPT_DIR" \
  NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 ANCHOR_COEF=0 \
  sbatch --parsable --job-name=A_plainh3_ad --time=23:59:00 --mem=96G \
         --account="$ACCT" scripts/slurm/adapt.slrm)
rc=$?
if [ $rc -ne 0 ] || [ -z "$ADAPT_ID" ]; then
    echo "FATAL: adapt submit failed (rc=$rc)"; exit 1
fi
echo "submitted rung-A adapt: $ADAPT_ID"

EVAL_ID=$(ADAPT="$ADAPT_DIR" N=500 MP=10 TAG=pix224_plainh3_final \
  sbatch --parsable --job-name=A_plainh3_fin --time=2:59:00 --account="$ACCT" \
         --dependency=afterany:"$ADAPT_ID" scripts/slurm/eval_simdist.slrm)
if [ -n "$EVAL_ID" ]; then
    echo "submitted rung-A headline eval: $EVAL_ID (afterany:$ADAPT_ID)"
else
    echo "!! eval submit failed -- run eval_simdist.slrm on $ADAPT_DIR by hand"
fi
echo "rung-A follow-on chain established"
