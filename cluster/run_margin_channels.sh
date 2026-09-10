#!/bin/bash
# Launch the panel-(b) seed replications from
# cluster/jobs_threading_margin_channels.txt, N at a time, round-robin over the
# GPUs given in GPUS.  Skips any run whose directory already exists, so it is
# safe to re-run after a partial batch.
#
#   GPUS="0 1" MAX=4 ./cluster/run_margin_channels.sh          # launch
#   ./cluster/run_margin_channels.sh --status                  # progress table
#
# Each run is ~1h45m at 100k transitions and peaks around 2 GB.
set -uo pipefail
cd /home/Chenyang/ibrl
JOBS=cluster/jobs_threading_margin_channels.txt
EXP=exps/simdist_threading_3arm

if [ "${1:-}" = "--status" ]; then
  printf "%-40s %8s %8s\n" run transitions score
  while read -r arm seed; do
    case $arm in both) n=dynaibrl_m001;; dataonly) n=dynaibrl_m001dataonly;;
                 bootonly) n=dynaibrl_m001bootonly;; neither) n=dynaibrl_m0demo;; esac
    d=$EXP/${n}_seed${seed}; [ -d "$d" ] || continue
    last=$(grep -oE "\[adapt @ [0-9]+ transitions\] target_score=[0-9.]+" $d/train.log 2>/dev/null | tail -1)
    printf "%-40s %8s %8s\n" "${n}_seed${seed}" \
      "$(echo "$last" | grep -oE '@ [0-9]+' | tr -d '@ ')" \
      "$(echo "$last" | grep -oE '=[0-9.]+$' | tr -d '=')"
  done < $JOBS
  exit 0
fi

GPUS=${GPUS:-"0 1"}; MAX=${MAX:-4}
gpus=($GPUS); i=0; launched=0
while read -r arm seed; do
  case $arm in both) n=dynaibrl_m001;; dataonly) n=dynaibrl_m001dataonly;;
               bootonly) n=dynaibrl_m001bootonly;; neither) n=dynaibrl_m0demo;; esac
  [ -d "$EXP/${n}_seed${seed}" ] && { echo "skip ${n}_seed${seed} (exists)"; continue; }
  [ "$launched" -ge "$MAX" ] && { echo "reached MAX=$MAX; re-run to launch more"; break; }
  g=${gpus[$((i % ${#gpus[@]}))]}
  echo "launch $arm seed$seed -> GPU $g"
  setsid ./cluster/job_threading_margin_channels.sh "$arm" "$seed" "$g" </dev/null >/dev/null 2>&1 &
  disown; i=$((i+1)); launched=$((launched+1)); sleep 20
done < $JOBS
echo "launched $launched run(s)"
