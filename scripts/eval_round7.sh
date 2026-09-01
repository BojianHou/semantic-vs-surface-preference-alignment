#!/usr/bin/env bash
# Round-7: evaluate each saved Dr.DPO-tuning checkpoint on the FULL decomposed test
# set (2000), split by response-length direction. Writes metrics/round7_<label>_fulltest.csv.
# Only uses GPUs 5-7 (0-4 belong to another user's training).
# Run from project root:  bash scripts/eval_round7.sh [label ...]   (default: all present)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
LABELS=("$@")
if [ ${#LABELS[@]} -eq 0 ]; then
  LABELS=(drdpo_bp025 drdpo_bp05 drdpo_bp2 drdpo_strat drdpo_rdpo drdpo_i4 drdpo_i4b drdpo_accum)
fi
GPUS=(5 6 7)
i=0
for label in "${LABELS[@]}"; do
  ckpt="checkpoints/round7/$label/final"
  if [ ! -d "$ckpt" ]; then echo "SKIP $label (no $ckpt)"; continue; fi
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  echo "=== eval $label on gpu $gpu ==="
  CUDA_VISIBLE_DEVICES=$gpu $V scripts/eval_length_split_checkpoint.py \
    --checkpoint "$ckpt" --label "round7_$label" \
    --out_csv "metrics/round7_${label}_fulltest.csv"
  i=$(( i + 1 ))
done
echo "[round7] eval complete"
