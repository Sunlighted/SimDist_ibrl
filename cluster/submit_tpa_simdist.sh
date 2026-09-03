#!/bin/bash
# Submit TPA_D0 SimDist STAGE-1 distillation on a Compute Canada cluster.
# Run this ON THE CLUSTER, from the repo root, after cluster/sync_tpa_simdist.sh.
#
#   ACCOUNT=def-<pi> bash cluster/submit_tpa_simdist.sh
#   ACCOUNT=def-<pi> SIF=cluster/ibrl.sif bash cluster/submit_tpa_simdist.sh
#   ACCOUNT=def-<pi> DRY=1 bash cluster/submit_tpa_simdist.sh    # check only
#
# It preflights every asset first. That matters here because the failure mode
# without it is expensive: the job loads a 10.2 GB replay for several minutes
# and only THEN dies on a missing 8 MB prior checkpoint.
#
# Env:
#   ACCOUNT    (required) Slurm account, e.g. def-<pi>
#   SIF        apptainer image; omit to use the venv route
#   CHAIN_MAX  max self-resubmissions (default 5); stage-1 needs ~3 x 12 h
#   TIME       override the 12:00:00 walltime
#   DRY        1 = preflight only, do not submit
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=release/cfgs/mimicgen/tpa_d0_state_simdist.yaml
SAVE=exps/mimicgen/tpa_simdist_diversev1_h5_seed1
DATA=release/data/mimicgen/tpa_d0_source_diverse_v1/dataset.hdf5
EXPERT=exps/mimicgen/tpa_source_expert_frozen
PRIOR=exps/mimicgen/tpa_source_rnn_bc_tp1_n1000_seed1
JOB=cluster/job_tpa_simdist.sh

fail=0
chk() {  # chk <path> <what> [min_bytes]
  if [ ! -e "$1" ]; then
    printf '  MISSING  %-64s %s\n' "$1" "$2"; fail=1; return
  fi
  sz=$(stat -Lc%s "$1" 2>/dev/null || echo 0)
  if [ -n "${3:-}" ] && [ "$sz" -lt "$3" ]; then
    printf '  TOO SMALL %-63s %s (%s B < %s)\n' "$1" "$2" "$sz" "$3"; fail=1; return
  fi
  printf '  ok       %-64s %s\n' "$1" "$(numfmt --to=iec "$sz" 2>/dev/null || echo "$sz")"
}

echo "== preflight =="
chk "$CFG"                  "stage-1 config"
chk "$JOB"                  "sbatch script"
chk "$DATA"                 "source diverse replay"        10000000000
chk "$EXPERT/model0.pt"     "frozen source expert (V_src)"   100000000
chk "$EXPERT/cfg.yaml"      "expert cfg (load_frozen_agent reads it)"
# The one that is easy to forget: the expert's cfg names this BC prior and
# load_frozen_agent loads it while rebuilding the expert.
chk "$PRIOR/model0.pt"      "expert's BC prior (Tp=1 RNN-GMM)"
chk "$PRIOR/cfg.yaml"       "prior cfg"

# The replay symlink must have been followed by rsync -L, not shipped as a link.
if [ -L "release/data/mimicgen/tpa_d0_source_diverse_v1" ] && \
   [ ! -e "release/data/mimicgen/tpa_d0_source_diverse_v1/" ]; then
  echo "  DANGLING SYMLINK: re-sync with rsync -L"; fail=1
fi

# Resuming? Say so explicitly -- a stale resume.pt silently skips steps.
if [ -f "$SAVE/resume.pt" ]; then
  echo "  note     $SAVE/resume.pt present -> the job RESUMES from it."
  echo "           delete it to start stage-1 from scratch."
fi

: "${ACCOUNT:?set ACCOUNT=def-<pi>}"
[ "$fail" = 0 ] || { echo "preflight FAILED -- not submitting."; exit 1; }
echo "preflight OK"

if [ "${DRY:-0}" = "1" ]; then echo "DRY=1, stopping here."; exit 0; fi

mkdir -p cluster/logs
set -x
sbatch --account="$ACCOUNT" ${TIME:+--time="$TIME"} \
       --export=ALL,CHAIN=1,CHAIN_MAX="${CHAIN_MAX:-5}"${SIF:+,SIF="$SIF"} \
       "$JOB"
