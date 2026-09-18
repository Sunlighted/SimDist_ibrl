#!/bin/bash
# §48 follow-on: fan a completed stage-1 out into its 5 ADAPT seeds (and, for the
# dynens stage-1, BOTH the no-anchor and +anchor rungs), then chain one headline
# n=500 eval behind each adapt.
#
# WHY THIS EXISTS (infra §7z ①): stage-1 self-chunks, so there is no single job
# id to hang --dependency=afterany on at submit time. pixel_stage1.slrm therefore
# calls this from its THEN_SBATCH hook, exactly once, guarded by
# <save>/.then_submitted. Invoked ON A COMPUTE NODE -- sbatch calls only, no GPU
# work, no waiting.
#
# Usage (from THEN_SBATCH, or by hand if the hook ever misses):
#   ARCH=plain_h3  bash scripts/slurm/launch_aug_ladder_followon.sh   # rung A
#   ARCH=dynens    bash scripts/slurm/launch_aug_ladder_followon.sh   # rungs B + C
#
# ★ THE DESIGN (see docs/prompts/aug-5seed-ladder-prompt.md):
#   ONE stage-1 per architecture, FIVE adapt seeds -- which is what the published
#   state ladder did (it lists "one stage-1 world model" in its own Caveats, and
#   its 10 seeds are ADAPT seeds). Rungs B and C SHARE the dynens stage-1: the
#   anchor is an adapt-time knob, so retraining for C would only add a variance
#   component the published error bars do not have.
# ★ DEMO_OFFSET=0/20/40/60/80 gives each seed a DISJOINT 20-demo draw out of the
#   200-episode pool. Without it every seed trains on the identical first 20 and
#   the error bars silently exclude §38's single-draw sigma = 0.092. Within a
#   seed all rungs share the draw => paired deltas (~0.027); across seeds the
#   draws differ => the error bar on the mean delta carries 0.092 honestly.
# ★ Budget is IDENTICAL on every arm or cmp_adapt_runs.py refuses to compare:
#   NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 DYN_ANCHOR_COEF=0.1.
#   INTERVAL=2000 is the MINIMUM that reaches 100k inside 23:59 -- raise INTERVAL
#   if you ever must economise, NEVER EVAL_N (it drives TopkSaver selection, so
#   changing it is a protocol change; iron law #15 / §46i).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/yifanr/simdist/repo}"

ARCH="${ARCH:?set ARCH=plain_h3|dynens}"
# DRYRUN=1 prints the exact sbatch lines instead of submitting, so the hook path
# can be exercised end-to-end (including the stage-1 ckpt existence checks)
# without putting 15 x 24 h jobs in the queue.
DRYRUN="${DRYRUN:-0}"
sb () { if [ "$DRYRUN" = 1 ]; then echo "DRYRUN sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch "$@"; fi; }
OFFSETS="${OFFSETS:-0 20 40 60 80}"
DEMOS=release/data/mimicgen/threading_d0_target_dynamics_pix224_s400k_n200/dataset.hdf5
ANCHOR_POOL=/scratch/yifanr/simdist/data/thr_d0_src_pix224/shard_1/dataset.hdf5

case "$ARCH" in
  plain_h3)
    STAGE1_DIR=exps/mimicgen/thr_pix224_aug_plainh3_stage1
    RUNGS="A:0"          # rung A only -- one head, H=3
    ;;
  dynens)
    STAGE1_DIR=exps/mimicgen/thr_pix224_aug_dynens_stage1
    RUNGS="B:0 C:1.0"    # B and C share this stage-1; they differ ONLY in the anchor
    ;;
  *) echo "FATAL: ARCH must be plain_h3|dynens (got '$ARCH')"; exit 1 ;;
esac

# PROPAGATE THE ACCOUNT: adapt.slrm and eval_simdist.slrm both hardcode
# `#SBATCH --account=aip-gigor`, so without the explicit flag this follow-on
# silently drops back to gigor (fairshare 0.106) even though the stage-1 chain
# that called it ran on aip-rhinehar (0.657, ~6x the priority factor). Same class
# of bug as the --mem fallback that OOM'd job 5212432: --export carries env vars,
# never sbatch resource flags.
# ★ MEASURED 2026-09-15: gigor beat rhinehar by ~13 h on an identical 200G
# stage-1 (19:46 vs 09:07 next day), despite CLAUDE.md recording rhinehar's
# fairshare as ~6x better. Fairshare is only one term in the priority, so do NOT
# assume -- alternate across both accounts and let whichever queue is moving
# take the job. Results are account-independent, so this costs nothing.
ACCTS=(aip-rhinehar aip-gigor)
ACCT="${SLURM_JOB_ACCOUNT:-aip-rhinehar}"
# ★ ALTERNATE BY **SEED**, NOT BY SUBMISSION INDEX. Alternating per job put
# B{2,4}+C{1,3,5} on one account and B{1,3,5}+C{2,4} on the other -- and since a
# ladder delta must be PAIRED WITHIN SEED, the fast account delivered five
# finished runs that shared NO seed with each other, i.e. ZERO measurable
# anchor deltas. Keying on the seed puts both rungs of a seed on the same queue,
# so whatever finishes first yields a usable pair.
echo "=== §48 follow-on | ARCH=$ARCH | account=$ACCT ==="
echo "stage1 = $STAGE1_DIR"
[ -s "$STAGE1_DIR/latest.pt" ] || { echo "FATAL: no $STAGE1_DIR/latest.pt"; exit 1; }
[ -s "$DEMOS" ] || { echo "FATAL: no $DEMOS"; exit 1; }

n_ok=0; n_bad=0
for rung_spec in $RUNGS; do
    RUNG="${rung_spec%%:*}"
    COEF="${rung_spec##*:}"
    for OFF in $OFFSETS; do
        SEED=$(( OFF / 20 + 1 ))
        ADAPT_DIR="exps/mimicgen/aug48_${RUNG}_s${SEED}"
        ADAPT_ID=$(ARM=pix224 \
          STAGE1="$STAGE1_DIR/latest.pt" \
          DEMOS="$DEMOS" ANCHOR="$ANCHOR_POOL" \
          SAVE="$ADAPT_DIR" \
          NUM_DEMO=20 MAX_TRANS=100000 INTERVAL=2000 EVAL_N=100 \
          DYN_ANCHOR_COEF=0.1 ANCHOR_COEF="$COEF" DEMO_OFFSET="$OFF" \
          sb --parsable --job-name="a48${RUNG}s${SEED}" \
                 --time=23:59:00 --mem=96G --cpus-per-task=12 \
                 --account="${ACCTS[$((SEED % 2))]}" scripts/slurm/adapt.slrm)
        if [ -z "$ADAPT_ID" ]; then
            echo "!! FAILED to submit rung $RUNG seed $SEED (offset $OFF)"
            n_bad=$((n_bad + 1)); continue
        fi
        A_EVAL="${ACCTS[$((SEED % 2))]}"
        EVAL_ID=$(ADAPT="$ADAPT_DIR" N=500 MP=10 TAG="aug48_${RUNG}_s${SEED}" \
          sb --parsable --job-name="e48${RUNG}s${SEED}" \
                 --time=2:59:00 --account="$A_EVAL" \
                 --dependency=afterany:"$ADAPT_ID" scripts/slurm/eval_simdist.slrm)
        echo "rung $RUNG seed $SEED  offset=$OFF anchor=$COEF  adapt=$ADAPT_ID eval=${EVAL_ID:-FAILED}"
        [ -z "$EVAL_ID" ] && echo "   !! eval submit failed -- run eval_simdist.slrm on $ADAPT_DIR by hand"
        n_ok=$((n_ok + 1))
    done
done
echo "=== submitted $n_ok adapt+eval pairs ($n_bad failures) for ARCH=$ARCH ==="
