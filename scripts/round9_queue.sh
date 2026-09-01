#!/usr/bin/env bash
# Queue the round-9 arms that could not start, launching each as a GPU frees up.
#
# Round 9 has 12 runs (4 arms x 3 seeds) but only 8 GPUs, and the counterfactual arms
# put 4 response blocks per example in the batch (~4x the activation memory of a plain
# pair), so two of them on one GPU OOMs. This waits for a genuinely idle GPU instead.
#
#   bash scripts/round9_queue.sh "cf_dpo 3" "flat_dpo 3"
set -uo pipefail
cd "$(dirname "$0")/.."
FREE_MIB=${FREE_MIB:-150000}   # a GPU is "free" when this much memory is available

free_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -v need="$FREE_MIB" -F', *' '$2 > need { print $1; exit }'
}

for JOB in "$@"; do
  set -- $JOB
  ARM=$1; SEED=$2
  while :; do
    G=$(free_gpu)
    if [ -n "$G" ]; then
      echo "[queue] ${ARM}_s${SEED} -> GPU $G"
      CUDA_VISIBLE_DEVICES=$G nohup bash scripts/run_round9.sh "$ARM" "$SEED" \
        > "logs/round9/${ARM}_s${SEED}.log" 2>&1 &
      sleep 120   # let it allocate before considering the next job
      break
    fi
    sleep 60
  done
done
echo "[queue] all queued jobs launched"
