#!/usr/bin/env bash
# Serialize round-7 configs onto one GPU: wait for that GPU to go idle, then run
# each label in turn. Lets waves be queued without babysitting, and without
# touching GPUs 0-4 (another user's training).
#
#   nohup bash scripts/round7_queue.sh 5 <pid-to-wait-for> drdpo_strat drdpo_accum \
#        > logs/round7/queue5.log 2>&1 &
#
# WAIT_PID is the pid of the job currently holding that GPU (0 = start immediately).
# Waiting on an explicit pid rather than polling nvidia-smi: `--query-compute-apps`
# returns nothing visible in this environment, so a liveness poll there reports every
# GPU as free and stacks jobs onto a busy device.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1; shift
WAIT_PID=$1; shift

if [ "$WAIT_PID" != "0" ]; then
  echo "[queue$GPU] waiting for pid $WAIT_PID to exit ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[queue$GPU] pid $WAIT_PID exited at $(date -Is)"
fi

for LABEL in "$@"; do
  echo "[queue$GPU] === starting $LABEL at $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=$GPU bash scripts/run_round7.sh "$LABEL" > "logs/round7/$LABEL.log" 2>&1
  echo "[queue$GPU] === finished $LABEL at $(date -Is) EXIT=$? ==="
  sleep 30   # let the GPU drain before the free_gpu poll for the next label
done
echo "[queue$GPU] all done"
