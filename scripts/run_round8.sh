#!/usr/bin/env bash
# Round-8 Support-Augmented KL-Dr.DPO — inner KL-regularized DRO over the rejected
# counterfactual orbit (rejected / rejected_double_prime / rejected_prime), outer
# aggregation unchanged. One config per GPU. Run from the project root:
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_round8.sh cf_drdpo_tau1
#
# All arms train on Bojian92/tldr_preference_decomposed[train] (n=3000) — the only
# set carrying the counterfactual columns — so the cf-off control sees exactly the
# same prompts and the same (chosen, rejected) pairs as the cf arms.
#
# Labels
#   base_drdpo        control: Dr.DPO, cf OFF (same data, same steps)
#   base_dpo          control: DPO,    cf OFF
#   cf_drdpo_tau1     headline: inner KL-DRO (tau_v=1.0) + outer Dr.DPO dual
#   cf_drdpo_tau025   sharper inner adversary (tau_v=0.25 -> closer to max_k)
#   cf_drdpo_tau4     softer inner adversary  (tau_v=4.0  -> closer to nominal mean)
#   cf_dpo_tau1       inner KL-DRO with the plain batch mean outside (isolates the
#                     inner term from the Dr.DPO dual)
#
# Trajectories -> results/round8_cf_kl_dro/<label>/<method>/
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
LABEL=$1

COMMON=(--train_dataset Bojian92/tldr_preference_decomposed --train_split train
        --learning_rate 5e-6 --batch_size 8 --grad_accum 2 --num_train_epochs 2
        --eval_every 50 --max_eval_samples 256 --eval_at_start --report_to none
        --no_save --result_dir "results/round8_cf_kl_dro/$LABEL")

CF=(--use_counterfactual_kl_dro --cf_epsilon_length 0.10 --cf_epsilon_length_syntax 0.05)

case "$LABEL" in
  base_drdpo)      ARGS=(--method drdpo --drdpo_beta_prime 1.0) ;;
  base_dpo)        ARGS=(--method dpo) ;;
  cf_drdpo_tau1)   ARGS=(--method drdpo --drdpo_beta_prime 1.0 "${CF[@]}" --cf_kl_temperature 1.0) ;;
  cf_drdpo_tau025) ARGS=(--method drdpo --drdpo_beta_prime 1.0 "${CF[@]}" --cf_kl_temperature 0.25) ;;
  cf_drdpo_tau4)   ARGS=(--method drdpo --drdpo_beta_prime 1.0 "${CF[@]}" --cf_kl_temperature 4.0) ;;
  cf_dpo_tau1)     ARGS=(--method dpo "${CF[@]}" --cf_kl_temperature 1.0) ;;
  *) echo "unknown label $LABEL"; exit 1 ;;
esac

echo "[round8] $LABEL : ${ARGS[*]}"
$V train.py "${ARGS[@]}" "${COMMON[@]}"
echo "[round8] $LABEL EXIT=$?"
