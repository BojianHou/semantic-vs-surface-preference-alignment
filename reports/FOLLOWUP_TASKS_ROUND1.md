# Follow-up Tasks — Round 1 — DPO/SimPO Decomposition

*Date: 2026-06-23. Dataset: `Bojian92/tldr_preference_decomposed` (TL;DR), Qwen2.5 tokenizer. Eval: first 256 of validation, eval every 10 steps, 1162 evals → step 11608. Full detail lives in `REPORT.md` §9 + §5; this file is the frozen standalone summary of round 1 (four follow-up tasks).*

> **Convention.** Each batch of follow-up tasks gets a frozen `FOLLOWUP_TASKS_ROUND<N>.md` digest; `REPORT.md` is the single up-to-date superset. This is round 1 (Tasks 1–4). Round 2 (normalization redistribution, length-direction split, `rejected″` drop mechanism) lives in `FOLLOWUP_TASKS_ROUND2.md` → `REPORT.md` §9.4–§9.5.*

## Task status

| # | Task | Status | Primary artifact |
|---|---|---|---|
| 1 | Plot chosen & rejected (4 objects) separately to test entanglement | ✅ done | `results/core_dpo_v3/task1_absolute_logp_4objects.png` |
| 2 | Set SimPO `average_log_prob=False` (no length norm) to test length bias | ✅ done | `results/ablation_simpo_sumlogp/`, `length_bias_summary.csv` |
| 3 | Audit how often chosen is longer than rejected in the dataset | ✅ done | §9.2 (this report, Task 3) |
| 4 | Make DPO hyperparameters aggressive so DPO actually moves | ✅ done | `results/core_dpo_v3/`, `checkpoints/dpo-v3/final` |

---

## Task 1 — Entanglement of the four objects

Plotting the **absolute** log-prob of each object (`chosen`, `rejected′`, `rejected″`, `rejected`) separately — rather than their gaps — exposes how coupled they move during training.

| Run | chosen Δ | rejected Δ | per-step corr(Δchosen, Δrejected) | reading |
|---|---|---|---|---|
| DPO v2 | +0.9 | +0.9 | +0.68 | flat, mild coupling |
| **DPO v3** (aggressive) | **−19.8** | **−22.5** | **+0.98** | **near-perfectly entangled — all 4 fall in lockstep** |
| SimPO v2 | +2.4 | +1.8 | +0.93 | curves visibly diverge (chosen ↑, rejected″ ↓ −3.3) |

**Finding.** DPO v3's "margin gain" is mostly a **rigid downward shift of the whole distribution** (corr 0.98): the four objects fall together and the margin widens only because `rejected` falls slightly faster. SimPO v2 instead **reshapes the chain** — the objects separate (chosen rises while the rejected tail is crushed) rather than translating as a block. This is the visual confirmation of the likelihood-displacement story in Task 4.

---

## Task 2 — SimPO length-normalization ablation (`average_log_prob=False`)

**Setup.** TRL hardcodes SimPO's length normalization at `cpo_trainer.py:804`. `train.py` now exposes `--no-simpo_average_log_prob`, which overrides `get_batch_logps` in a `CPOTrainer` subclass to use **summed** (not length-averaged) log-probs. Run matches SimPO v2 (lr 5e-6, 2 epochs) → `results/ablation_simpo_sumlogp/`.

> **Caveat — not a single-variable ablation.** Summed log-probs are ~−90 nats, so β=2.0 saturates `logsigmoid` and the model won't train. β had to be lowered to **0.1**, so this run varies *both* length-normalization (off) **and** β (2.0→0.1). Read it as "sum-logp SimPO at a trainable β," not a pure flip of one flag.

**Metric.** Per eval example, compare the model's **per-token** preference (`chosen_mean_logp − rejected_mean_logp`) against which response is longer. "% favoring longer" = share of pairs where the per-token-preferred response is the longer one; 50% = length-neutral. Per-token (not summed) strips the mechanical "more tokens → more negative sum" effect. Plot: `results/ablation_simpo_sumlogp/length_bias_comparison.png`.

| Run (end of training) | sum margin | % favoring longer | corr(per-token margin, len-diff) | acc gap (chosen-longer − chosen-shorter) |
|---|---|---|---|---|
| init (step 0) | 6.44 | 46.8% | −0.083 | −0.45 |
| **SimPO v2 (len-norm, β=2.0)** | 7.05 | **42.8%** (↓ prefers shorter) | −0.256 | −0.69 |
| **SimPO sum-logp (no norm, β=0.1)** | 6.68 | **46.8%** (≈ neutral, holds) | −0.157 | −0.71 |
| DPO v3 (sum, β=0.1) | 9.07 | 49.6% (↑ slightly longer) | −0.007 | −0.38 |

**Findings.**
1. **Length normalization actively pushes SimPO toward shorter responses.** Normalized v2 drops "% favoring longer" 46.8 → 42.8 and drives `corr(per-token margin, length)` more negative (−0.26) — it learns to prefer the more concise response. This is the length-debiasing SimPO is designed for.
2. **Turning normalization off removes that shortening pressure.** Sum-logp SimPO holds at ~46.8% (≈ untrained, near-neutral) with a weaker correlation (−0.16) — more length-tolerant than the normalized version, the predicted effect.
3. **Cost: weaker preference learning.** Sum-logp margin grew only +0.24 nats (6.44 → 6.68) vs v2's +0.60 — at β=0.1 it separates chosen/rejected less (confounded with β, per caveat).
4. "Summed log-prob" alone does **not** imply length bias: DPO v3 (also summed) is the most length-neutral per token (corr −0.007). Data skew (Task 3) and β/normalization interact.

**Reproducibility.** This run was re-executed end-to-end with the eval protocol matched to v2/v3 (`--max_eval_samples 256`). Every metric above reproduced exactly (corr −0.157, % favoring longer 46.8%, sum margin 6.68) — the conclusions are not an artifact of eval-set size. Script: `analyze_length_bias.py` → `length_bias_summary.csv`.

---

## Task 3 — Length audit of `Bojian92/tldr_preference_decomposed`

Splits: **train = 3000, validation = 2000** (training eval'd only the first 256 of validation). Length measured with the Qwen2.5 tokenizer.

| Split | `len(chosen) > len(rejected)` (tokens) | ties | mean(chosen − rejected) |
|---|---|---|---|
| train | 1672 / 3000 = **55.7%** | 62 | +3.5 tok (median +3) |
| validation | 1031 / 2000 = **51.5%** | 109 | +0.7 tok (median +1) |

**Findings.**
- A **mild skew toward chosen being longer**, stronger in train (55.7%) than validation (51.5%).
- The skew is small enough that the data is not strongly length-confounded — consistent with Task 2's near-neutral baseline (init at 46.8% favoring longer).
- Incidental: by **word** count, chosen is *never* longer than `rejected′`/`rejected″` (0%), though it is in ~34–40% of cases by tokens (with many ties). The decomposition stages appear constructed to preserve/extend word length relative to chosen.

---

## Task 4 — Aggressive hyperparameters to make DPO move

DPO was flat at the original settings. "Aggressive" = **higher peak LR + a second epoch** (β/γ unchanged).

| | DPO v1 | DPO v2 | **DPO v3** | SimPO v2 |
|---|---|---|---|---|
| peak LR | 5e-7 | 1e-6 (2×) | **5e-6 (5×)** | 5e-6 (5×) |
| epochs | 1 | 2 | **2** | 2 |
| beta | 0.1 | 0.1 | 0.1 | 2.0 |

**Did it move?** Yes — DPO v3 is the only DPO run that clearly moves.

| Run | margin start → end | Δ margin | acc end | verdict |
|---|---|---|---|---|
| DPO v1 | 6.44 → 6.38 | −0.06 | 0.605 | flat |
| DPO v2 | 6.44 → 6.51 | +0.07 | 0.602 | ~flat (2× LR) |
| **DPO v3** | 6.44 → **9.08** | **+2.63** | 0.602 | **moves most (5× LR)** |
| SimPO v2 | 6.44 → 7.05 | +0.60 | 0.605 | clearly moving |

**Findings.**
1. **The 5× learning rate, not the second epoch, unlocks DPO.** DPO v2 (2× LR, 2 epochs) stays flat; DPO v3 (5× LR) lifts off in epoch 1 and plateaus by mid-run. SimPO needs the second epoch because its length-normalized β=2.0/γ=0.5 loss builds margin more slowly.
2. **But the margin gain is likelihood displacement, not better ranking.** Absolute log-probs both crash: chosen −90.8 → −110.6 (**−19.8**), rejected −97.2 → −119.7 (**−22.5**). The model becomes ~20 nats *less* likely to produce even the chosen summary; the margin widens only because rejected falls faster. Ties directly to Task 1's corr-0.98 entanglement.
3. **Accuracy does not improve in any run** (all ~0.60–0.66, DPO v3 included). **Margin growth ≠ better ordering** — the headline caution of this whole study.

---

## Cross-task takeaways

- **DPO (aggressive) vs SimPO move the margin by different mechanisms.** DPO v3 translates the whole log-prob distribution downward (likelihood displacement, entanglement 0.98); SimPO v2 reshapes the chain and keeps absolute log-probs roughly stable.
- **Length normalization is doing real work in SimPO**: it actively biases toward shorter responses; removing it returns the model to length-neutral but learns the preference less strongly (confounded with β).
- **The data is only mildly length-skewed** (~52–56% chosen longer), so observed length effects are driven by the objective, not the dataset.
- **No method improves pairwise accuracy** — every run sits at ~0.60–0.66. Treat margin/reward gains with suspicion absent an accuracy or downstream-quality signal.

### Artifacts
- Plots: `results/core_dpo_v3/task1_absolute_logp_4objects.png`, `results/ablation_simpo_sumlogp/length_bias_comparison.png`, `results/core_dpo_v3/decomposed_v3_comparison.png`
- Data: `results/core_dpo_v3/dpo/`, `results/core_v2_aggressive/simpo/`, `results/ablation_simpo_sumlogp/simpo/` (each `decomposed_summary.csv` + 1162 per-step CSVs)
- Scripts: `analyze_length_bias.py` → `length_bias_summary.csv`
- Code: `train.py` (`--no-simpo_average_log_prob`), `CPOTrainer` subclass overriding `get_batch_logps`
