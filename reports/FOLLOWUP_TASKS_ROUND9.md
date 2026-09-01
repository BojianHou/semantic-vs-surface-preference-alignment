# Follow-up Tasks — Round 9 — Rebuilt eval set, corrected re-ranking, and the death of Support-Augmented KL-Dr.DPO

*Date: 2026-08-30. Executes the four steps of `AUDIT_2026-08-29.md`: fix the metric, rebuild the
data, match the objective to the question, and replicate over seeds. Full detail will land in
`REPORT.md` §9.12; this file is the frozen round-9 digest.*

> **Convention.** Frozen per-round digest; `REPORT.md` is the superset. Rounds 1–8:
> `FOLLOWUP_TASKS_ROUND{1..8}.md`.
>
> **Status: complete.** New eval + train sets built by 420 LLM writer/auditor agents (0 errors);
> 19 legacy checkpoints re-scored; 18 new training runs (6 arms × 3 seeds).

---

## 1. The rebuilt datasets

Written by agents, not by rule-based code, then QC'd mechanically
(`scripts/assemble_decomposed_v2.py`).

| | legacy | v2 |
|---|---|---|
| eval examples | 2000 rows / **195 unique prompts** | **1979 rows / 1979 unique prompts** |
| variants token-matched to `chosen` | 30% | **100%, exactly** |
| `rejected_prime` vs `rejected_double_prime` similarity | 0.545 (two free paraphrases) | **0.709** (same words, restructured) |
| fully degenerate rows | 6.2% | 0 |
| twin variants for a noise floor | none | `*_v2` for both counterfactuals |
| training set | 3000 rows / 524 prompts | **1998 rows / 1998 prompts**, disjoint from eval |

The twins are the important addition: two variants written to the *same* spec differ only by
wording, so their score gap is the irreducible noise floor. Every other gap is now reported in
units of it, which is what makes `|g2| = 21 nats` interpretable.

## 2. Re-ranking the 19 existing checkpoints — the round-7 leaderboard does not survive

Semantics ÷ syntax, both measured against the noise floor (higher = more meaning-driven):

| method | ratio | vs untrained | 95% CI | significant |
|---|---|---|---|---|
| `drdpo_strat` | 2.371 | +0.242 | [+0.16, +0.32] | **yes** |
| DPO v3 | 2.345 | +0.216 | [+0.15, +0.28] | **yes** |
| Dr.DPO | 2.308 | +0.179 | [+0.09, +0.27] | **yes** |
| **`drdpo_i4b`** (round-7 champion) | 2.153 | +0.024 | [−0.05, +0.10] | **no** |
| SimPO v2 | 2.153 | +0.024 | — | no |
| BASE (untrained) | 2.129 | — | — | — |

1. **The round-7 champion is statistically indistinguishable from the untrained model** on the
   question the project is actually asking.
2. **Its mechanism is now visible.** `drdpo_i4b` has the *lowest* systematic length bias (−3.33)
   but the *second-highest* per-example length sensitivity (2.33× noise vs 1.49× for untrained).
   The I4 penalty equalizes the mean margin between length groups — exactly the statistic that
   cancels. It optimized the artifact, which is why it looked like a record.
3. **Training does improve the ratio, but modestly.** 2.13 → 2.37 at best (+11%), while the raw
   semantic gap nearly doubles (20.5 → 39.7). Most of preference optimization is amplification;
   a small, real part is better discrimination. Three very different objectives (DPO v3, Dr.DPO,
   `drdpo_strat`) land within 0.06 of each other — a ceiling, not a method difference.

## 3. Round 9 — the Support-Augmented KL-Dr.DPO objective fails its own control

Every arm sees the **same counterfactual text**; only the aggregation differs. 3 seeds each.

| arm | steps | accuracy | semantics/syntax | what it is |
|---|---|---|---|---|
| base_drdpo | 250 | 0.5191 ±0.0008 | 2.187 ±0.008 | no counterfactuals at all |
| cf_drdpo | 250 | 0.5205 ±0.0013 | 2.216 ±0.005 | worst-case, p=(.85,.10,.05) — **the round-8 method** |
| cf_dpo | 250 | 0.5196 ±0.0003 | 2.205 ±0.003 | same, plain mean outside |
| cf_uniform | 250 | 0.5309 ±0.0012 | 2.300 ±0.002 | worst-case, p=(⅓,⅓,⅓) |
| **flat_sm** | 250 | **0.5452 ±0.0005** | **2.353 ±0.013** | counterfactuals as plain pairs, step-matched |
| flat_dpo | 750 | 0.5449 ±0.0015 | 2.361 ±0.022 | same, 3× steps |
| UNTRAINED | 0 | 0.5179 | 2.129 | |

**The method does not work, and round 8's apparent gain was the data.** Simply adding the
counterfactuals as ordinary preference pairs beats the worst-case aggregation on both metrics
(+0.137 semantics/syntax, +0.025 accuracy), at matched optimizer steps and matched text. Seed
spread is 0.005–0.022, so the gap is 6–25 sd.

**The mechanism, and why the spec's defaults guaranteed it.** With p=(0.85, 0.10, 0.05) and
τ_v = 1, the inner adversary barely moves off nominal (measured counterfactual mass ≈ 0.14 in
round 8), so the two counterfactuals receive **~15% of the gradient** — while the flat control
gives them 67%. The benefit tracks how much weight the counterfactuals get, monotonically:

    counterfactual weight:   0%      ~15%     ~50%      67%
    semantics/syntax:       2.187 -> 2.216 -> 2.300 -> 2.353
                          (base)  (cf_drdpo)(cf_uniform)(flat)

Raising ε to uniform recovers most of the gap (2.216 → 2.300), confirming the starvation
diagnosis — but flat still wins at every matched weight. **The worst-case machinery contributes
nothing; the extra data does all the work.** Recommendation: drop the inner DRO and train on
counterfactuals as ordinary pairs.

## 4. An unexpected positive: counterfactual data is ~46× more efficient

`flat_sm` reaches semantics/syntax **2.353 from 1998 training examples**. The best model trained
on the full 92,858-example set (`drdpo_strat`) reaches **2.371**. Essentially the same ratio from
**46× less data** — the counterfactual pairs are dramatically more informative *for this metric*.

The catch: accuracy does not come along (0.545 vs 0.589). Counterfactual training buys the
meaning-vs-surface ratio, not preference accuracy. Those appear to be separate axes.

## Caveats

- Round-9 arms train on 1998 examples and are therefore undertrained in absolute terms; their
  accuracies (0.52–0.55) are not comparable to the 92k-trained checkpoints (0.58–0.60). The
  **within-round** comparison is matched and is what the conclusions rest on.
- The re-ranking covers 19 checkpoints; the round-5 invariance winners no longer exist on disk.
- One eval example and one training example were dropped for token-count misses (~0.05%).

## Follow-ups

1. Scale the flat counterfactual training set (the 46× efficiency result says this is the
   highest-value direction) and check whether the ratio keeps climbing past 2.37 or plateaus.
2. The ~2.37 ceiling is hit by three unrelated objectives — test whether it is a property of the
   1.5B base model by repeating on a larger one.
3. Accuracy and the semantics/syntax ratio move independently. Worth one experiment to establish
   whether they are genuinely orthogonal.
