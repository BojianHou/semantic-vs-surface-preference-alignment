# Follow-up Tasks — Round 2 — DPO/SimPO Decomposition

*Date: 2026-06-29. Dataset: `Bojian92/tldr_preference_decomposed` (TL;DR), Qwen2.5 tokenizer. Eval: first 256 of validation, eval every 10 steps (matched to round 1). Full detail lives in `REPORT.md` §9.4–§9.5; this file is the frozen standalone summary of round 2.*

> **Convention.** Each batch of follow-up tasks gets a frozen `FOLLOWUP_TASKS_ROUND<N>.md` digest; `REPORT.md` is the single up-to-date superset. Round 1 (Tasks 1–4: entanglement, length audit, length-norm ablation, aggressive DPO) lives in `FOLLOWUP_TASKS_ROUND1.md` → `REPORT.md` §5 + §9.1–§9.3.*

## Task status

| # | Task | Status | Primary artifact | REPORT.md |
|---|---|---|---|---|
| 1 | Normalization-effect plot: how len-norm vs sum-logp redistributes the decomposition chain | ✅ done | `results/ablation_simpo_sumlogp/normalization_effect.png` | §9.4a |
| 2 | Length-direction split: recompute every metric split by chosen-longer vs rejected-longer | ✅ done | `results/ablation_simpo_sumlogp/length_split_{lennorm_v2,sumlogp}.png`, `length_split_metrics.csv` | §9.4b |
| 3 | Mechanism of the `rejected″ − rejected` ~5-nat drop under len-norm SimPO | ✅ done | `results/ablation_simpo_sumlogp/rpp_drop_mechanism.png` | §9.5 |

---

## Task 1 — Normalization redistributes the margin without changing the net outcome

Two SimPO variants at matched eval protocol (n=256, every 10 steps): **len-norm v2** (`average_log_prob=True`, β=2.0) vs **sum-logp** (`average_log_prob=False`, β=0.1). End-of-training component values (avg over last 20 evals). The identity `C−R = (C−R′) + (R′−R″) + (R″−R)` holds for both. Plot: `normalization_effect.png` (6-panel overlay). Script: `plot_normalization_effect.py`.

| gap (sum logp) | len-norm v2 | sum-logp | Δ (sum − norm) |
|---|---|---|---|
| chosen − rejected (net) | 7.065 | 6.685 | −0.38 |
| chosen − rejected (mean logp, **objective**) | 0.108 | 0.104 | −0.004 |
| chosen − rejected′ | 19.042 | 12.761 | **−6.28** |
| rejected′ − rejected″ | −6.835 | −7.274 | −0.44 |
| rejected″ − rejected | −5.141 | **+1.198** | **+6.34** |
| accuracy P(chosen>rejected) | 0.606 | 0.602 | −0.004 |

**Finding.** Net margin (~7 vs ~6.7) and accuracy (~0.60) are nearly identical, but normalization **redistributes where the margin lives**: two large, nearly offsetting shifts — `chosen−rejected′` collapses (19.0→12.8) while `rejected″−rejected` flips sign (−5.14→+1.20). The observable outcome is unchanged; the internal chain is reshaped. The sign flip is explained mechanistically in Task 3.

---

## Task 2 — Accuracy is almost entirely a length artifact

Split the n=256 eval set by length direction (`chosen_tokens >` vs `< rejected_tokens`, ties dropped) and recompute every metric. Plots: `length_split_{lennorm_v2,sumlogp}.png`. Script: `analyze_length_split.py` → `length_split_metrics.csv`.

| run | subset | n | acc | sum_C−R | mean_C−R | C−R′ | R′−R″ | R″−R |
|---|---|---|---|---|---|---|---|---|
| len-norm v2 | chosen longer | 106 | **0.208** | −18.5 | −0.068 | 13.4 | −12.8 | −19.1 |
| len-norm v2 | rejected longer | 144 | **0.897** | +25.9 | +0.235 | 23.5 | −2.5 | +4.9 |
| sum-logp | chosen longer | 106 | **0.198** | −16.8 | −0.032 | 7.4 | −15.1 | −9.2 |
| sum-logp | rejected longer | 144 | **0.903** | +24.0 | +0.201 | 17.0 | −1.5 | +8.6 |

**Findings.**
1. The headline ~0.60 accuracy decomposes into **~21% when the correct (chosen) answer is longer vs ~90% when it is shorter** — both variants essentially **rank by brevity**.
2. The flip persists **per-token** (`mean_C−R` sign-flips with length direction), so it is not just the mechanical "more tokens → more negative sum."
3. **Length normalization barely dents this**: the accuracy split is near-identical (0.21 / 0.90) for both variants.
4. This sharpens REPORT.md conclusion #3 (§7): *no method improves ranking* — what the ~0.60 hides is a near-deterministic length preference.

---

## Task 3 — Why `rejected″ − rejected` drops ~5 nats under len-norm SimPO

The `rejected″ − rejected` (sum-logp) curve falls −0.05 → −5.14 over len-norm SimPO training. Hypothesis tested: rejected″ is longer than rejected and SimPO prefers shorter. Scripts: `audit_lengths.py`, `investigate_rpp_drop.py` → `rpp_drop_mechanism.png`.

**Length facts (exact eval tokenizer).** rejected″ is constructed to match **chosen's** length (corr 0.96 train / 0.91 val), so on train it is longer than rejected 55.6% of the time (+2.34 tok). But on the full val it is ~neutral (49.6%, +0.33 tok) and on the n=256 eval slice it is incidentally ~1.8 tok **shorter** (38.3% longer).

| split | mean tok rej″ / rej | % rej″ > rej | mean(rej″−rej) | corr(rej″, chosen) |
|---|---|---|---|---|
| train (3000) | 29.8 / 27.4 | 55.6% | +2.34 | 0.96 |
| validation (2000) | 34.1 / 33.8 | 49.6% | +0.33 | 0.91 |
| eval slice (256) | 36.1 / 37.9 | 38.3% | −1.76 | — |

**The drop is per-token, not length.** Decomposing Δ(sum gap) into a per-token term `(Δgap_mean)·len` and a length term:

| run | gap_sum (init→end) | from **per-token** | from **length** | abs per-tok logp: rej″ | abs per-tok logp: rej |
|---|---|---|---|---|---|
| **len-norm v2** | −0.05 → **−5.14** | **−5.44** | +0.35 | falls −0.089 | **rises** +0.058 |
| sum-logp (control) | −0.05 → **+1.20** | +1.28 | −0.02 | rises +0.165 | rises +0.131 |

**Findings.**
1. Length contributes ~nothing (+0.35). The drop is driven entirely by len-norm SimPO pushing rejected″'s **per-token** log-prob down (−0.089/tok) while the true rejected's rises (+0.058); over ~37 tokens that becomes the ~5-nat sum-logp fall.
2. **Decisive control:** the sum-logp run sees identical responses/lengths but does the **opposite** (gap rises to +1.20). Same data, opposite sign ⇒ length normalization itself produces the drop.
3. Mechanism: normalization makes the loss reward per-token efficiency, so the more-verbose rejected″ (a chosen-length paraphrase of rejected content) gets a lower per-token probability than the terse rejected; **length is only the ~37× amplifier** that turns the small per-token gap into a large sum-logp drop. The original length hypothesis is right in spirit (a normalization/conciseness effect) but the drop is not produced by rejected″ being longer on the eval set.

---

## Cross-task takeaways

- **Normalization reshapes the decomposition but not the outcome** — matched net margin and accuracy, two large offsetting internal shifts (`C−R′` ↓, `R″−R` sign flip).
- **The ~0.60 accuracy is a brevity heuristic**, not preference learning: ~0.21 when chosen is longer, ~0.90 when shorter — and normalization does not fix it.
- **The `R″−R` sign flip is a normalization/per-token-efficiency effect**, not a raw-length effect — proven by the same-data opposite-sign control.

### Artifacts
- Plots: `results/ablation_simpo_sumlogp/normalization_effect.png`, `results/ablation_simpo_sumlogp/length_split_{lennorm_v2,sumlogp}.png`, `results/ablation_simpo_sumlogp/rpp_drop_mechanism.png`
- Data: `results/ablation_simpo_sumlogp/length_split_metrics.csv`
- Scripts: `plot_normalization_effect.py`, `analyze_length_split.py`, `audit_lengths.py`, `investigate_rpp_drop.py`
