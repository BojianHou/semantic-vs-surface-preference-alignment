#!/usr/bin/env bash
# Round-9 — the audit's Step 3 + Step 4.
#
# Step 3: match the OBJECTIVE to the research question. "Judge by meaning, not surface form"
#         is a robustness goal, so the loss should penalize the WORST surface variant of each
#         example rather than the average -- that is the round-8 Support-Augmented KL-Dr.DPO
#         (--use_counterfactual_kl_dro). Round 8 could not test it properly because the only
#         counterfactual training data was 3000 rows over 524 unique prompts, all of it
#         built with the defective recipe. This round trains it on the rebuilt v2 set
#         (1999 unique prompts, token-length-matched, structurally-controlled variants).
#
# Step 4: 3 seeds per arm. Everything in rounds 1-8 was single-seed.
#
# Arms (each x3 seeds):
#   base_drdpo   Dr.DPO, cf OFF          -- the audit's best method on the STATED goal
#   cf_drdpo     Dr.DPO + inner KL-DRO   -- the robustness objective
#   cf_dpo       plain DPO + inner KL-DRO -- isolates the inner term from the Dr.DPO dual
#   flat_dpo     DPO with the counterfactuals as INDEPENDENT rows -- the information-matched
#                control round 8 was missing (same extra data, no worst-case aggregation)
#
# Run from the project root:  CUDA_VISIBLE_DEVICES=0 bash scripts/run_round9.sh cf_drdpo 1
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
ARM=$1
SEED=${2:-1}
LABEL="${ARM}_s${SEED}"
TRAIN=data/decomposed_v2_train/decomposed_v2_train.parquet
FLAT=data/decomposed_v2_train/decomposed_v2_train_flat.parquet

ACCUM=2
[ "$ARM" = "flat_sm" ] && ACCUM=6      # step-match the flat control to the cf arms

COMMON=(--train_dataset parquet --train_split train
        --learning_rate 5e-6 --batch_size 8 --grad_accum "$ACCUM" --num_train_epochs 2
        --eval_every 50 --max_eval_samples 256 --eval_at_start --report_to none
        --seed "$SEED" --result_dir "results/round9/$LABEL"
        --output_dir "checkpoints/round9/$LABEL")

CF=(--use_counterfactual_kl_dro --cf_epsilon_length 0.10 --cf_epsilon_length_syntax 0.05
    --cf_kl_temperature 1.0)

case "$ARM" in
  base_drdpo) ARGS=(--method drdpo --drdpo_beta_prime 1.0 --train_data_files "$TRAIN") ;;
  cf_drdpo)   ARGS=(--method drdpo --drdpo_beta_prime 1.0 "${CF[@]}" --train_data_files "$TRAIN") ;;
  cf_dpo)     ARGS=(--method dpo "${CF[@]}" --train_data_files "$TRAIN") ;;
  # information-matched control: the 3 rejected variants as separate training rows, so the
  # arm sees exactly the same text as cf_* but aggregates it with a plain mean.
  flat_dpo)   ARGS=(--method drdpo --drdpo_beta_prime 1.0 --train_data_files "$FLAT") ;;
  # STEP-matched flat control. flat_dpo is information-matched (same pairs) but takes 3x the
  # optimizer steps, because 3 separate rows per example at the same batch size = 3x the
  # steps per epoch. grad_accum 6 puts 48 pairs in an effective batch, matching what the cf
  # arms consume per step (16 examples x 3 counterfactuals), so both arms run 250 steps.
  flat_sm)    ARGS=(--method drdpo --drdpo_beta_prime 1.0 --train_data_files "$FLAT") ;;
  # Why cf may be losing to flat: with the spec's nominal p=(0.85,0.10,0.05) and tau_v=1 the
  # adversary barely moves off nominal (measured counterfactual mass ~0.14 in round 8), so the
  # two counterfactuals receive only ~15% of the gradient -- while the flat control gives them
  # 67%. cf_uniform sets p=(1/3,1/3,1/3) so the orbit is weighted like the flat control, which
  # separates "worst-case aggregation is the wrong idea" from "the default epsilons starved it".
  cf_uniform) ARGS=(--method drdpo --drdpo_beta_prime 1.0 --use_counterfactual_kl_dro
                    --cf_epsilon_length 0.3333 --cf_epsilon_length_syntax 0.3333
                    --cf_kl_temperature 1.0 --train_data_files "$TRAIN") ;;
  *) echo "unknown arm $ARM (base_drdpo|cf_drdpo|cf_dpo|flat_dpo|flat_sm|cf_uniform)"; exit 1 ;;
esac

[ -f "${ARGS[-1]}" ] || { echo "missing training parquet ${ARGS[-1]}"; exit 1; }

echo "[round9] $LABEL : ${ARGS[*]}"
$V train.py "${ARGS[@]}" "${COMMON[@]}"
STATUS=$?
echo "[round9] $LABEL EXIT=$STATUS"
if [ $STATUS -eq 0 ]; then rm -rf "checkpoints/round9/$LABEL"/checkpoint-*; fi
