# Follow-up Tasks — Round 3 — DPO/SimPO Decomposition

*Date: 2026-06-30. Retrain-on-length-split experiment. Train: FULL `trl-lib/tldr-preference` (92,858 pairs) split by response-length direction. Eval: `Bojian92/tldr_preference_decomposed` validation, first 256, every 10 steps. Full detail in `REPORT.md` §9.6; this file is the frozen standalone summary of round 3.*

> **Convention.** Each batch of follow-up tasks gets a frozen `FOLLOWUP_TASKS_ROUND<N>.md` digest; `REPORT.md` is the single up-to-date superset. Round 1 → `FOLLOWUP_TASKS_ROUND1.md` (§5 + §9.1–§9.3); round 2 → `FOLLOWUP_TASKS_ROUND2.md` (§9.4–§9.5).

## Motivation

Round-2 Task 2 (§9.4b) was a **post-hoc eval-set slice**: it showed the mixed SimPO model ranks by brevity (acc ~0.21 when chosen is longer vs ~0.90 when chosen is shorter) but could not say *why*. Round 3 **retrains** on length-controlled data to disambiguate:

| Hypothesis | Prediction |
|---|---|
| **Data artifact** — the ~56% chosen-longer skew taught it | Train only on chosen-longer pairs → model prefers *longer* → acc[chosen longer] rises above 0.5 (heuristic flips) |
| **Objective bias** — length-normalized SimPO has an intrinsic brevity prior | Brevity ranking persists regardless of training-data length direction |

## Setup

Three conditions, all len-norm SimPO, identical hyperparameters (lr 5e-6, 2 epochs, β=2.0, γ=0.5, eval n=256 every 10):

| Condition | Training data | N |
|---|---|---|
| chosen_longer arm | trl-lib pairs where chosen is the longer response | 37,835 |
| rejected_longer arm | trl-lib pairs where rejected is the longer response | 37,835 |
| mixed baseline (control) | full trl-lib (existing `results/core_v2_aggressive/simpo`) | 92,858 |

The two arms are **equal-N subsampled** (seed 42) so the only difference is length direction, not data volume. Length-direction split of the full train set (Qwen2.5 tokenizer, chat-template response tokens):

| Direction | Count | Share |
|---|---|---|
| chosen_longer | 51,750 | 55.7% |
| rejected_longer | 37,835 | 40.7% |
| ties (dropped) | 3,273 | 3.5% |

**Environment note.** The original 3.12 + torch env was gone (ephemeral host); it was reconstructed into `.venv` from the wandb-frozen `requirements.txt`. `trl==0.29.1` / `transformers==4.57.3` / `datasets==4.0.0` match the original exactly; torch resolved to 2.12.1+cu130 (vs original 2.7.1+cu128) — runs on 8×B200, does not affect the SimPO math.

## Result

End-of-training metrics (avg over last 20 evals), eval set sliced by length direction:

| Trained on | acc[chosen longer] | acc[rejected longer] | acc[all] | sum C−R [chosen longer] | mean C−R [chosen longer] (per-token) |
|---|---|---|---|---|---|
| mixed baseline | 0.208 | 0.897 | 0.606 | −18.5 | −0.068 |
| **chosen_longer arm** | **0.292** | 0.856 | 0.610 | −9.1 | **+0.863** |
| **rejected_longer arm** | **0.000** | **1.000** | 0.586 | −481.7 | −7.796 |

Convergence sane for both arms (final loss ~0.05, grad-norm ~20–28, 2 full epochs / ~4,730 steps, 474 evals). The rejected_longer arm's end overall sum-margin blew up to +101.8 (vs baseline +7.1) — the aggressive setup fully absorbing the length signal.

## Findings

1. **The heuristic is primarily an objective/inductive bias, not a data artifact.** Training *only* on chosen-longer pairs did **not** flip it: acc[chosen longer] moved only 0.21 → 0.29 (still far below 0.5). The model **resisted** learning "longer = better."
2. **Sharp asymmetry (the key result).** Training only on rejected-longer pairs (where correct = shorter) produced a **perfect brevity ranker: 0.000 / 1.000**. The objective **eagerly** absorbs "shorter = better" but **fights** "longer = better" — a brevity prior independent of the data direction.
3. **Overall accuracy is immovable.** All three conditions sit at ~0.586–0.613. **Rebalancing training-data length does not fix the ~0.60 ceiling** — sharpening §7 conclusion #3 and round-2 §9.4b.
4. **Objective-level nuance.** Per-token preference (`mean C−R` = SimPO's actual signal) *did* respond to the data: the chosen_longer arm's chosen-longer-subset per-token gap went −0.068 → **+0.863** (it learned to prefer the longer-correct content per token). But length normalization means this per-token gain never overcomes the length term in the summed ranking — so sum-based accuracy stays brevity-dominated. Data controls the per-token direction; the objective controls the ranking.

## Cross-task takeaways

- **You cannot debias the ~0.60 length-ranking by fixing the data** — the length-normalized SimPO objective re-imposes brevity regardless.
- **The brevity prior is asymmetric**: trivially learned when data agrees (→ 0.00/1.00), strongly resisted when data disagrees (→ stays ~baseline).
- Consistent with round-2 §9.5: brevity is a per-token-efficiency effect of normalization; here it manifests as an inductive bias that data cannot overturn at the ranking level.

### Artifacts
- Split: `results/length_split/split_indices.json` (`build_length_split.py`)
- Runs: `results/length_arm_chosen_longer/simpo/`, `results/length_arm_rejected_longer/simpo/` (474 per-step CSVs each + `decomposed_summary.csv`); control `results/core_v2_aggressive/simpo/`
- Analysis: `analyze_arms_length_split.py` → `arms_length_split_metrics.csv`, `results/length_split/arms_accuracy_by_length.png`
- Code: `train.py` `--train_length_subset {chosen_longer,rejected_longer}` (+ `--length_split_file`)
- Env: `.venv` (reconstructed from wandb `requirements.txt`)
