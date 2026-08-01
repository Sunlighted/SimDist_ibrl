#!/usr/bin/env bash
# GPU memory watchdog.
#
# Runs a command in its OWN process group and polls the target GPU. If the
# used-memory fraction exceeds THRESH for STREAK consecutive polls — OR if
# `nvidia-smi` itself starts timing out (a driver-wedging symptom) for STREAK
# consecutive polls — it kills the whole process group BEFORE the driver dies,
# so you get a clean Python OOM/abort instead of an unrecoverable Xid + reboot.
#
# Usage:
#   GPU=1 THRESH=0.92 bash scripts/gpu_guard.sh <command...>
#   GPU=1 bash scripts/gpu_guard.sh python scripts/generate_simdist_data.py ...
#
# Env:
#   GPU      physical GPU index to watch (matches CUDA_VISIBLE_DEVICES)  (default 0)
#   THRESH   used/total memory fraction danger line                     (default 0.92)
#   STREAK   consecutive breaches (or nvidia-smi failures) before kill  (default 3)
#   POLL     seconds between polls                                      (default 2)
#
# Exit codes: passthrough of the wrapped command, or 42 if the guard killed it.

set -uo pipefail
GPU="${GPU:-0}"
THRESH="${THRESH:-0.92}"
STREAK="${STREAK:-3}"
POLL="${POLL:-2}"

if [ "$#" -eq 0 ]; then
  echo "[gpu_guard] no command given" >&2
  exit 2
fi

# Launch the command as a new session leader so its PID == its process-group id;
# every child (incl. multiprocessing workers) inherits the group, so one
# `kill -- -PGID` reaches all of them.
setsid "$@" &
PGID=$!

cleanup() { kill -TERM -- -"$PGID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[gpu_guard] watching physical GPU $GPU  (THRESH=$THRESH, STREAK=$STREAK, POLL=${POLL}s, pgid=$PGID)"

breaches=0
nvfail=0
while kill -0 "$PGID" 2>/dev/null; do
  nvout="$(timeout 5 nvidia-smi -i "$GPU" \
            --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)"
  rc=$?
  if [ $rc -ne 0 ] || [ -z "$nvout" ]; then
    nvfail=$((nvfail + 1))
    echo "[gpu_guard] nvidia-smi query failed/timeout on GPU $GPU  [$nvfail/$STREAK] — driver may be wedging"
    if [ "$nvfail" -ge "$STREAK" ]; then
      echo "[gpu_guard] DANGER (driver unresponsive) — killing process group $PGID."
      kill -TERM -- -"$PGID" 2>/dev/null || true; sleep 3; kill -KILL -- -"$PGID" 2>/dev/null || true
      exit 42
    fi
    sleep "$POLL"; continue
  fi
  nvfail=0

  used="$(echo "$nvout" | head -1 | awk -F',' '{gsub(/ /,"",$1); print $1}')"
  total="$(echo "$nvout" | head -1 | awk -F',' '{gsub(/ /,"",$2); print $2}')"
  over="$(awk -v u="$used" -v t="$total" -v th="$THRESH" \
          'BEGIN{ if(t>0 && u/t>th) print 1; else print 0 }')"
  if [ "$over" = "1" ]; then
    breaches=$((breaches + 1))
    frac="$(awk -v u="$used" -v t="$total" 'BEGIN{ if(t>0) printf "%.2f", u/t; else print 0 }')"
    echo "[gpu_guard] GPU $GPU mem ${used}/${total} MiB (frac=${frac} > ${THRESH})  [$breaches/$STREAK]"
    if [ "$breaches" -ge "$STREAK" ]; then
      echo "[gpu_guard] DANGER (memory) — killing process group $PGID before the driver dies."
      kill -TERM -- -"$PGID" 2>/dev/null || true; sleep 3; kill -KILL -- -"$PGID" 2>/dev/null || true
      exit 42
    fi
  else
    breaches=0
  fi
  sleep "$POLL"
done

# command finished on its own — reap and propagate its exit code
trap - EXIT INT TERM
wait "$PGID"
exit $?
