#!/usr/bin/env bash
# Wait for all 8 Round-6 training runs to finish (each run_round6.sh prints
# "[round6] <label> EXIT="), then run the full-test eval + summary. Meant to be
# launched in the background so it self-drives to the final table.
set -uo pipefail
cd "$(dirname "$0")/.."
LABELS=(orpo lddpo rdpo_robust rdpo drdpo sampo lmpo tie)

echo "[watch] waiting for 8 training runs to finish..."
while true; do
  done=0
  for l in "${LABELS[@]}"; do
    grep -q "\[round6\] $l EXIT=" "logs/round6/$l.log" 2>/dev/null && done=$((done+1))
  done
  echo "[watch] $(date +%H:%M:%S) finished=$done/8"
  [ "$done" -eq 8 ] && break
  sleep 60
done

echo "[watch] all training done — exit codes:"
for l in "${LABELS[@]}"; do grep "\[round6\] $l EXIT=" "logs/round6/$l.log" | tail -1; done

echo "[watch] running full-test eval..."
bash scripts/eval_round6.sh > logs/round6/eval_all.log 2>&1
echo "[watch] running summary..."
.venv/bin/python scripts/summarize_round6.py > logs/round6/summary.txt 2>&1
echo "[watch] DONE. Summary:"
cat logs/round6/summary.txt
touch logs/round6/WATCH_DONE
