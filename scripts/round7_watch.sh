#!/usr/bin/env bash
# Wait for every round-7 training job (and its queue) to exit, then evaluate all
# checkpoints on the full 2000 test set and summarize. Fires unattended.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[watch] start $(date -Is)"
while pgrep -f "round7_queue.sh|run_round7.sh" > /dev/null; do sleep 120; done
echo "[watch] all training finished $(date -Is)"
bash scripts/eval_round7.sh
.venv/bin/python scripts/subset_gaps.py --out metrics/subset_gaps.csv
.venv/bin/python scripts/summarize_round6.py --glob "metrics/round7_*_fulltest.csv" \
  --out metrics/round7_summary.csv
touch logs/round7/WATCH_DONE
echo "[watch] done $(date -Is)"
