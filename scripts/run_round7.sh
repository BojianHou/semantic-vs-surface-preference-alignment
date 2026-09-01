#!/usr/bin/env bash
# Round-7 Dr.DPO tuning — same headline recipe as round 6 (full trl-lib/tldr-preference,
# LR 5e-6, 2 epochs, batch 8 x grad_accum 2) so every number is comparable to
# metrics/round6_summary.csv. One config per GPU. Run from project root:
#
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_round7.sh drdpo_bp05
#
# Labels
#   beta' sweep (round 6 only tried beta'=1.0):
#     drdpo_bp025 drdpo_bp05 drdpo_bp2 drdpo_bp4
#   mechanisms:
#     drdpo_strat  (B) KL-DRO applied WITHIN each length environment, groups equally weighted
#     drdpo_rdpo   (C) Dr.DPO + R-DPO explicit length penalty
#     drdpo_i4     (D) Dr.DPO + I4 length-invariance penalty (round-5 winner term)
#     drdpo_accum  (E) DRO dual over the full accumulated batch, not the micro-batch of 8
#
# Final checkpoints -> checkpoints/round7/<label>/final
# Trajectories      -> results/round7_drdpo_tuning/<label>/<method>/
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
LABEL=$1

COMMON=(--learning_rate 5e-6 --batch_size 8 --grad_accum 2 --num_train_epochs 2
        --eval_every 100 --max_eval_samples 256 --eval_at_start --report_to none
        --save_steps 100000 --result_dir "results/round7_drdpo_tuning/$LABEL"
        --output_dir "checkpoints/round7/$LABEL")

case "$LABEL" in
  # --- beta' sweep: how hard to down-weight high-loss pairs (round 6 = 1.0) ---
  drdpo_bp025) ARGS=(--method drdpo --drdpo_beta_prime 0.25) ;;
  drdpo_bp05)  ARGS=(--method drdpo --drdpo_beta_prime 0.5) ;;
  drdpo_bp2)   ARGS=(--method drdpo --drdpo_beta_prime 2.0) ;;
  drdpo_bp4)   ARGS=(--method drdpo --drdpo_beta_prime 4.0) ;;
  # --- mechanisms ---
  drdpo_strat) ARGS=(--method drdpo --drdpo_beta_prime 1.0 --drdpo_stratify_length) ;;
  drdpo_rdpo)  ARGS=(--method drdpo --drdpo_beta_prime 1.0 --rdpo_alpha 0.05 --drdpo_length_penalty) ;;
  drdpo_i4)    ARGS=(--method drdpo --drdpo_beta_prime 1.0 --lambda_len_inv 5e-4) ;;
  # lambda=5e-4 OVERSHOOTS on top of Dr.DPO (acc_cl 0.691 > acc_rl 0.622 — the bias
  # flips), while lambda=0 undershoots (0.614/0.687). The crossing is near 2.5e-4.
  drdpo_i4b)   ARGS=(--method drdpo --drdpo_beta_prime 1.0 --lambda_len_inv 2.5e-4) ;;
  drdpo_accum) ARGS=(--method drdpo --drdpo_beta_prime 1.0 --drdpo_buffer 128) ;;
  *) echo "unknown label $LABEL"; exit 1 ;;
esac

echo "[round7] $LABEL : ${ARGS[*]}"
$V train.py "${ARGS[@]}" "${COMMON[@]}"
STATUS=$?
echo "[round7] $LABEL EXIT=$STATUS"

# Keep only final/ — the mid-run checkpoint carries optimizer state (~8.7G/run)
# and nothing downstream reads it. Disk is at 90%.
if [ $STATUS -eq 0 ]; then
  rm -rf "checkpoints/round7/$LABEL"/checkpoint-*
  echo "[round7] $LABEL pruned intermediate checkpoints"
fi
