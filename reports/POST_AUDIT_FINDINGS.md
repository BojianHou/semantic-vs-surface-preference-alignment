# Post-Audit Findings — what the corrected measurement says

*Date: 2026-08-31. This is the consolidated report on everything done after
`AUDIT_2026-08-29.md`: the rebuilt datasets, the re-measurement of all 19 existing
checkpoints, and 18 new seeded training runs. It supersedes the pre-audit numbers in
`REPORT.md` §§9.9–9.11, which were computed with a measurement procedure now known to be
confounded. Round-level detail: `FOLLOWUP_TASKS_ROUND9.md`. Superset: `REPORT.md` §9.12.*

---

## 0. Executive summary

1. **The instrument was broken, and fixing it changes the answer.** Averaged gaps cancel;
   summed log-probabilities carry a mechanical length term; and without a noise floor no gap
   is interpretable. All three are now fixed.
2. **The project's crowned best method does not survive re-measurement.** `drdpo_i4b`
   (round-7 record) is statistically indistinguishable from the **untrained** model on the
   research question.
3. **Preference optimization does make models more meaning-driven, but only by ~11%** — and
   it achieves this entirely by raising meaning-sensitivity, **not at all** by lowering
   surface-sensitivity.
4. **The stated goal (g2 → 0, g3 → 0) is not met by any of the 19 methods.** Syntax
   sensitivity sits at ~1.5× the noise floor for every model, including the untrained one.
5. **The method added in round 8 (Support-Augmented KL-Dr.DPO) fails its own control.**
   Using the same counterfactuals as ordinary preference pairs beats it.
6. **One promising positive:** counterfactual training data appears ~46× more sample-efficient
   for this metric than ordinary preference data.

---

## 1. What was wrong with the old measurement

Three independent defects, each affecting every round from 1 to 8.

**(a) Averaging cancels the effect.** All rounds reported `mean(g2)`, which answers "is there a
*systematic* syntax preference?" It does not answer "does the model respond to syntax?" Large
per-example effects of opposite sign average to ≈0. On the legacy set Dr.DPO's mean g2 was −0.86
(looks like invariance) while its per-example |g2| was **24.9 nats** — larger than its entire
semantic gap.

**(b) A mechanical length floor.** A summed log-probability falls ~2.5 nats per extra token
regardless of quality, so g2/g3 can never reach 0 while the paired responses differ in length.
DPO v3's celebrated `g3 = −1.73` decomposes into **−0.19 content and −1.54 pure arithmetic**.
The target was unreachable by construction.

**(c) No reference scale.** Any two different sentences score differently. Without knowing how
big that baseline wobble is, "|g2| = 21 nats" is uninterpretable.

Two further data problems: the legacy eval set had **195 unique prompts** across 2000 rows
(so error bars were ~2× the naive ones), and its two "syntax variants" differed by **46% of
their characters** — they were unrelated paraphrases, so g2 measured vocabulary, not syntax.

---

## 2. What was rebuilt

### 2.1 The metric (`scripts/decomposition_metrics.py`)

Every gap is now split exactly into a **content** term and a **length** term, and three
distinct statistics are reported instead of one:

| statistic | question it answers | target |
|---|---|---|
| `bias` = mean(g) | is there a systematic preference? | 0 for g2/g3 |
| `sensitivity` = mean(\|g\|) | does the model respond at all? | 0 for g2/g3 |
| `consistency` = mean/sd | is the effect directional? | large for g1, ~0 for g2/g3 |

Confidence intervals are bootstrapped over **prompts**, not rows.

### 2.2 The noise floor — the key addition

The rebuilt data contains, for each example, a **second** version of each counterfactual written
to identical instructions (same meaning, same length, same syntactic target). Two such twins are
interchangeable, so their score gap is pure wording noise. Every other gap is then expressed in
units of it:

```
meaning score = |g1| / noise      syntax score = |g2| / noise      length score = |g3| / noise
ratio = meaning score / syntax score      <- the single number answering the research question
```

A ratio of 2.3 means: *the model responds to a change in meaning about 2.3× as strongly as to a
change in wording.*

### 2.3 The datasets (written by 420 LLM writer/auditor agents, 0 errors)

| | legacy | v2 |
|---|---|---|
| eval | 2000 rows / **195 unique prompts** | **1979 rows / 1979 unique prompts** |
| variants token-matched to `chosen` | 30.3% | **100%, exactly** |
| syntax-pair character similarity | 0.545 | **0.709** |
| fully degenerate rows | 6.2% | 0 |
| twins for a noise floor | none | yes |
| training set | 3000 rows / 524 prompts | **1998 rows / 1998 prompts**, disjoint from eval |

Schema (6 responses per example):
`prompt, chosen, rejected, rejected_double_prime, rejected_prime, rejected_double_prime_v2,
rejected_prime_v2`. There is deliberately no twin for `rejected` — it is authentic human text
from `trl-lib/tldr-preference`, so no second equally-valid version exists.

Files: `data/decomposed_v2/decomposed_v2_validation.parquet`,
`data/decomposed_v2_train/decomposed_v2_train{,_flat}.parquet`.

---

## 3. Finding 1 — the re-ranking overturns the leaderboard

All 19 existing checkpoints re-scored on the rebuilt eval set. **These models were not
retrained**; only the measurement changed. Data: `metrics/audit_all_methods_v2.csv`.

| method | accuracy | meaning | syntax | length | **ratio** | vs untrained |
|---|---|---|---|---|---|---|
| `drdpo_strat` | 0.589 | 3.506 | 1.479 | 1.554 | **2.371** | +0.242 |
| `drdpo_bp2` | 0.579 | 3.552 | 1.510 | 1.548 | 2.352 | +0.223 |
| DPO v3 | 0.580 | 3.535 | 1.508 | 1.551 | 2.345 | +0.216 |
| rDPO | 0.568 | 3.531 | 1.520 | 1.536 | 2.323 | +0.194 |
| LD-DPO | 0.548 | 3.495 | 1.506 | 1.492 | 2.320 | +0.191 |
| Dr.DPO | 0.592 | 3.457 | 1.498 | 1.538 | 2.308 | +0.179 |
| LMPO | 0.531 | 3.638 | 1.577 | 1.208 | 2.307 | +0.178 |
| SamPO | 0.547 | 3.456 | 1.501 | 1.469 | 2.302 | +0.173 |
| Tie | 0.543 | 3.426 | 1.491 | 1.486 | 2.298 | +0.169 |
| R-DPO | 0.565 | 3.532 | 1.539 | 1.661 | 2.295 | +0.166 |
| `drdpo_bp025` | 0.587 | 3.237 | 1.413 | 1.213 | 2.292 | +0.162 |
| `drdpo_accum` | 0.596 | 3.418 | 1.493 | 1.544 | 2.290 | +0.160 |
| `drdpo_rdpo` | 0.596 | 3.472 | 1.520 | 1.654 | 2.285 | +0.156 |
| `drdpo_bp05` | 0.604 | 3.231 | 1.441 | 1.290 | 2.242 | +0.112 |
| ORPO | 0.514 | 3.331 | 1.495 | 1.486 | 2.228 | +0.098 |
| SimPO v2 | 0.511 | 3.329 | 1.546 | 1.362 | 2.153 | +0.024 |
| **`drdpo_i4b`** (r7 "best") | 0.589 | 3.484 | **1.618** | **2.330** | **2.153** | **+0.024** |
| **BASE (untrained)** | 0.518 | 3.238 | 1.521 | 1.490 | **2.129** | — |
| `drdpo_i4` | 0.581 | 3.434 | 1.630 | 2.618 | **2.106** | **−0.023** |

Significance (bootstrap over prompts, 95%):

| method | difference from untrained | CI | significant |
|---|---|---|---|
| `drdpo_strat` | +0.242 | [+0.162, +0.323] | yes |
| DPO v3 | +0.216 | [+0.147, +0.284] | yes |
| Dr.DPO | +0.179 | [+0.094, +0.268] | yes |
| `drdpo_i4b` | +0.024 | [−0.047, +0.096] | **no** |
| SimPO v2 | +0.024 | [−0.032, +0.081] | **no** |

**1. The round-7 champion is indistinguishable from doing nothing.** Its 95% interval straddles
zero.

**2. And the mechanism is diagnosable.** `drdpo_i4b` has the *lowest* systematic length bias of
any model but the *second-highest* per-example length sensitivity (2.330 vs 1.490 for untrained).
Those are consistent: it favours long on some examples and short on others, cancelling to ≈0 on
average. The I4 penalty it was trained with optimizes precisely that average. **It optimized the
statistic that cancels**, which is why it topped a leaderboard built on that statistic.
`drdpo_i4`, the same method at a stronger λ, lands *below* the untrained model.

**3. One method is actively harmful** on this axis (`drdpo_i4`, −0.023).

**4. The surviving winners cluster in a narrow band** (2.29–2.37) across very different
objectives — consistent with a ceiling of the base model rather than a method difference.

---

## 4. Finding 2 — training raises meaning-sensitivity but does not lower surface-sensitivity

This is the sharpest result in the report. Compare the columns:

| | untrained | best trained | change |
|---|---|---|---|
| meaning score | 3.238 | 3.506 | **+8%** |
| syntax score | 1.521 | 1.479 | **−3%** (noise) |
| length score | 1.490 | 1.554 | +4% |
| ratio | 2.129 | 2.371 | +11% |

**Not one of the 19 methods meaningfully reduces syntax sensitivity.** It sits at ~1.5× the noise
floor for everything, untrained included. The entire ratio improvement comes from the model
noticing meaning *more*, never from it ignoring surface form.

The raw numbers hide this: the un-normalized semantic gap roughly doubles (20.5 → 39.7 nats),
which is what the old tables reported as a large win. But the noise floor grows almost as much
(8.69 → 14.71). **Most of what preference optimization does is amplify all score differences;
a small, real part is improved discrimination.**

Against the stated goal — g1 large, g2 ≈ 0, g3 ≈ 0 — the verdict is: **g1 yes, g2 no, g3 no.**

---

## 5. Finding 3 — Support-Augmented KL-Dr.DPO loses to its own control

18 runs, 6 arms × 3 seeds, trained on the new counterfactual training set. **Every arm sees
identical text**; only the aggregation differs. Evaluated on the rebuilt eval set.

| arm | steps | accuracy | ratio | what it is |
|---|---|---|---|---|
| base_drdpo | 250 | 0.5191 ±0.0008 | 2.187 ±0.008 | no counterfactuals at all |
| **cf_drdpo** | 250 | 0.5205 ±0.0013 | 2.216 ±0.005 | **the round-8 method**, p=(.85,.10,.05) |
| cf_dpo | 250 | 0.5196 ±0.0003 | 2.205 ±0.003 | same, plain mean outside |
| cf_uniform | 250 | 0.5309 ±0.0012 | 2.300 ±0.002 | same, p=(⅓,⅓,⅓) |
| **flat_sm** | 250 | **0.5452 ±0.0005** | **2.353 ±0.013** | counterfactuals as plain pairs, step-matched |
| flat_dpo | 750 | 0.5449 ±0.0015 | 2.361 ±0.022 | same, 3× steps |
| UNTRAINED | 0 | 0.5179 | 2.129 | |

**The worst-case aggregation adds nothing.** Simply appending the counterfactuals as ordinary
preference pairs wins by +0.137 ratio and +0.025 accuracy at matched text *and* matched optimizer
steps — 6–25 seed-standard-deviations. Round 8's apparent success was the extra data.

**Mechanism, and why the specified defaults guaranteed it.** With p = (0.85, 0.10, 0.05) and
τ_v = 1 the inner adversary barely leaves nominal (measured counterfactual mass ≈ 0.14), so the
counterfactuals receive **~15% of the gradient**; the flat control gives them 67%. Benefit tracks
that weight monotonically:

```
counterfactual weight:   0%       ~15%      ~50%      67%
ratio:                  2.187 -> 2.216 -> 2.300 -> 2.353
                       (base)  (cf_drdpo)(cf_uniform)(flat)
```

Raising ε to uniform recovers most of the gap, confirming the starvation diagnosis — but flat
still wins at matched weight. **Recommendation: drop the inner DRO; train on counterfactuals as
ordinary pairs.**

---

## 6. Finding 4 — counterfactual data is ~46× more sample-efficient

`flat_sm` reaches ratio **2.353 from 1998 training examples**. The best model trained on the full
**92,858**-example set (`drdpo_strat`) reaches **2.371**. Nearly the same ratio from 46× less data.

The catch: accuracy does not follow (0.545 vs 0.589). Counterfactual training buys the
meaning-vs-surface ratio, not preference accuracy — they behave as separate axes.

**This claim is the least secure in the report**: the two runs differ in data source, training
length and dataset size simultaneously. It needs a proper data-scaling curve before being relied on.

---

## 7. What is now answered, and what is not

| question | answer | confidence |
|---|---|---|
| Does alignment give the model a *systematic* length/syntax bias? | **No** — and the untrained model's length bias is partly removed by training | high |
| Does alignment make the model *judge by meaning rather than surface form*? | **Only slightly.** Ratio 2.13 → 2.37 (+11%) | high |
| Does any method drive g2 → 0? | **No.** Syntax sensitivity is ~1.5× noise for all 19 methods and the untrained model | high |
| Is the round-7 record real? | **No** — indistinguishable from untrained | high |
| Does the round-8 method work? | **No** — loses to its own control | high |
| Is counterfactual data more efficient? | Probably, ~46× | **low** — confounded |
| Is the ~2.37 ratio a ceiling of the 1.5B model? | Unknown — only one base model tested | untested |

---

## 8. Publication readiness

There is a paper here, but it is a **benchmark-and-negative-result** paper, not a
new-method paper.

**Thesis.** *Standard evaluation of semantic vs. surface preference learning is confounded;
under controlled measurement, no existing preference-optimization method reduces a model's
sensitivity to meaning-preserving surface changes.*

**Contributions in hand:** the methodological critique (with a concrete case of a "winning"
method that is indistinguishable from untrained); a reusable controlled benchmark with a noise
floor; a 19-method seeded negative result; and a mechanism-identified negative result for a new
method.

**Gaps that must close first:**

| gap | why it matters | effort |
|---|---|---|
| Only one base model (Qwen2.5-1.5B) | The central claim may be a property of this one small model. Needs 2–3 models across sizes. | ~1 day GPU |
| Counterfactual meaning never validated | Only *mechanical* checks (token length, similarity) were run. If the agents drifted the meaning, g1 is contaminated. Needs an LLM-judge or human check on ~200 samples. | ~half day |
| The 46× claim is confounded | Needs a real data-scaling curve. | ~1 day |
| No positive control | A reviewer can argue the metric simply cannot move. Train something *designed* to reduce g2 and show the metric detects it. | ~1 day |

The second item is the one that could **invalidate** results rather than merely strengthen them,
and should be done first.

---

## 9. Artifacts

| what | where |
|---|---|
| Re-ranking table | `metrics/audit_all_methods_v2.csv` |
| Per-model raw scores | `metrics/*_v2.csv` (39 files) |
| Eval set | `data/decomposed_v2/decomposed_v2_validation.parquet` |
| Training sets | `data/decomposed_v2_train/decomposed_v2_train{,_flat}.parquet` |
| Corrected metric | `scripts/decomposition_metrics.py` |
| Re-ranking driver | `scripts/audit_all_methods.py --eval v2` |
| Eval sweep | `scripts/eval_all_on_v2.sh` |
| Round-9 training | `scripts/run_round9.sh <arm> <seed>` |
| Dataset generation | `scripts/wf_decomposed_v2.js` (workflow), `scripts/v2_shard_status.py` (resume), `scripts/assemble_decomposed_v2.py` (QC) |
| Audit diagnosis | `reports/AUDIT_2026-08-29.md` |
| Round-9 digest | `reports/FOLLOWUP_TASKS_ROUND9.md` |

Reproduce the headline table:

```bash
.venv/bin/python scripts/audit_all_methods.py --eval v2
```
