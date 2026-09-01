#!/usr/bin/env bash
# Re-evaluate every surviving checkpoint on the REBUILT v2 decomposed eval set.
#
# The v2 set (data/decomposed_v2/decomposed_v2_validation.parquet) fixes the three
# defects found in reports/AUDIT_2026-08-29.md: 2000 UNIQUE prompts (legacy: 195),
# counterfactuals token-length matched to `chosen` (so g2 carries no length term), and
# rejected_prime/rejected_double_prime that are structural rearrangements of each other
# rather than unrelated paraphrases. It also carries the *_v2 syntax variants, which let
# us measure the per-example syntax effect twice and separate signal from noise.
#
# Run from the project root:   bash scripts/eval_all_on_v2.sh
# Output: metrics/<legacy-name>_v2.csv, consumed by
#         .venv/bin/python scripts/audit_all_methods.py --eval v2
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
PARQUET=data/decomposed_v2/decomposed_v2_validation.parquet
KEYS="chosen,rejected_prime,rejected_double_prime,rejected,rejected_prime_v2,rejected_double_prime_v2"
NGPU=${NGPU:-8}

[ -f "$PARQUET" ] || { echo "missing $PARQUET — run scripts/assemble_decomposed_v2.py first"; exit 1; }
mkdir -p logs/v2eval

# label|checkpoint|output-csv-stem  (stem matches scripts/audit_all_methods.py SOURCES + _v2)
MODELS=(
  "BASE_untrained|Qwen/Qwen2.5-1.5B-Instruct|length_split_fulltest_BASE_untrained_v2"
  "DPO_v3|checkpoints/dpo-v3/final|length_split_fulltest_DPO_v3_v2"
  "SimPO_v2|checkpoints/simpo-strong/final|length_split_fulltest_SimPO_v2_v2"
  "drdpo|checkpoints/round6/drdpo/final|round6_drdpo_fulltest_v2"
  "rdpo|checkpoints/round6/rdpo/final|round6_rdpo_fulltest_v2"
  "lddpo|checkpoints/round6/lddpo/final|round6_lddpo_fulltest_v2"
  "rdpo_robust|checkpoints/round6/rdpo_robust/final|round6_rdpo_robust_fulltest_v2"
  "sampo|checkpoints/round6/sampo/final|round6_sampo_fulltest_v2"
  "tie|checkpoints/round6/tie/final|round6_tie_fulltest_v2"
  "orpo|checkpoints/round6/orpo/final|round6_orpo_fulltest_v2"
  "lmpo|checkpoints/round6/lmpo/final|round6_lmpo_fulltest_v2"
  "drdpo_i4b|checkpoints/round7/drdpo_i4b/final|round7_drdpo_i4b_fulltest_v2"
  "drdpo_i4|checkpoints/round7/drdpo_i4/final|round7_drdpo_i4_fulltest_v2"
  "drdpo_rdpo|checkpoints/round7/drdpo_rdpo/final|round7_drdpo_rdpo_fulltest_v2"
  "drdpo_strat|checkpoints/round7/drdpo_strat/final|round7_drdpo_strat_fulltest_v2"
  "drdpo_accum|checkpoints/round7/drdpo_accum/final|round7_drdpo_accum_fulltest_v2"
  "drdpo_bp05|checkpoints/round7/drdpo_bp05/final|round7_drdpo_bp05_fulltest_v2"
  "drdpo_bp2|checkpoints/round7/drdpo_bp2/final|round7_drdpo_bp2_fulltest_v2"
  "drdpo_bp025|checkpoints/round7/drdpo_bp025/final|round7_drdpo_bp025_fulltest_v2"
)

i=0
for M in "${MODELS[@]}"; do
  IFS='|' read -r LABEL CKPT STEM <<< "$M"
  if [ "$CKPT" != "Qwen/Qwen2.5-1.5B-Instruct" ] && [ ! -d "$CKPT" ]; then
    echo "[skip] $LABEL — no checkpoint at $CKPT"; continue
  fi
  GPU=$((i % NGPU))
  CUDA_VISIBLE_DEVICES=$GPU $V scripts/eval_length_split_checkpoint.py \
      --checkpoint "$CKPT" --label "$LABEL" \
      --eval_dataset parquet --data_files "$PARQUET" --split train \
      --response_keys "$KEYS" --eval_batch_size 16 \
      --out_csv "metrics/${STEM}.csv" > "logs/v2eval/${LABEL}.log" 2>&1 &
  echo "[launch] $LABEL on GPU $GPU"
  i=$((i + 1))
  if [ $((i % NGPU)) -eq 0 ]; then wait; fi   # one wave per GPU set
done
wait
echo "done: $(ls metrics/*_v2.csv 2>/dev/null | wc -l) v2 evaluations"
