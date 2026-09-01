#!/usr/bin/env bash
# Round-6: evaluate each saved checkpoint on the FULL decomposed test set (2000),
# split by response-length direction. Writes metrics/round6_<label>_fulltest.csv.
# Run from project root:  bash scripts/eval_round6.sh [label ...]   (default: all)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
LABELS=("$@"); [ ${#LABELS[@]} -eq 0 ] && LABELS=(orpo lddpo rdpo_robust rdpo drdpo sampo lmpo tie)
gpu=0
for label in "${LABELS[@]}"; do
  ckpt="checkpoints/round6/$label/final"
  if [ ! -d "$ckpt" ]; then echo "SKIP $label (no $ckpt)"; continue; fi
  echo "=== eval $label on gpu $gpu ==="
  CUDA_VISIBLE_DEVICES=$gpu $V scripts/eval_length_split_checkpoint.py \
    --checkpoint "$ckpt" --label "round6_$label" \
    --out_csv "metrics/round6_${label}_fulltest.csv"
  gpu=$(( (gpu+1) % 8 ))
done
