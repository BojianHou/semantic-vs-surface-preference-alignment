#!/usr/bin/env bash
# Round-6 length-robust preference methods — headline recipe (full trl-lib/tldr-preference,
# LR 5e-6, 2 epochs, batch 8 x grad_accum 2), one method per GPU. Run from project root:
#   bash scripts/run_round6.sh <label>        # launch one method (foreground)
# Labels: orpo lddpo rdpo_robust rdpo drdpo sampo lmpo tie
# Final checkpoints -> checkpoints/round6/<label>/final ; trajectories -> results/round6_length_robust/<label>/<method>/
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
V=.venv/bin/python
LABEL=$1
# EPOCHS: strict runs do 2 epochs of 92,858 (=185,716 examples seen). The tie mix
# is already 2x size (strict+ties), so 1 epoch over it matches examples-seen.
EPOCHS=2
COMMON=(--learning_rate 5e-6 --batch_size 8 --grad_accum 2
        --eval_every 100 --max_eval_samples 256 --eval_at_start --report_to none
        --save_steps 100000 --result_dir "results/round6_length_robust/$LABEL"
        --output_dir "checkpoints/round6/$LABEL")

case "$LABEL" in
  orpo)        ARGS=(--method orpo) ;;
  lddpo)       ARGS=(--method dpo --ld_alpha 0.5) ;;
  rdpo_robust) ARGS=(--method dpo --dpo_loss_type robust --label_smoothing 0.1) ;;
  rdpo)        ARGS=(--method rdpo --rdpo_alpha 0.05) ;;
  drdpo)       ARGS=(--method drdpo --drdpo_beta_prime 1.0) ;;
  sampo)       ARGS=(--method sampo) ;;
  lmpo)        ARGS=(--method lmpo) ;;
  tie)         ARGS=(--method tie --tie_weight 1.0 --tie_dataset data/tie_mix_a0.5.parquet); EPOCHS=1 ;;
  *) echo "unknown label $LABEL"; exit 1 ;;
esac
COMMON+=(--num_train_epochs $EPOCHS)

echo "[round6] $LABEL : ${ARGS[*]}"
$V train.py "${ARGS[@]}" "${COMMON[@]}"
echo "[round6] $LABEL EXIT=$?"
