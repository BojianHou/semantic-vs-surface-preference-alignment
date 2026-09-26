# Does preference alignment learn *meaning*, or *surface form*?

A controlled study of whether preference-optimized language models prefer a summary because of
**what it means** or because of **how it is worded**.

Nineteen preference-optimization methods, one untrained baseline, a purpose-built counterfactual
benchmark, and a measurement procedure that — unlike the obvious one — can actually tell the two
apart.

**Headline result: preference optimization makes models notice meaning more, but does not make
them any less sensitive to meaning-preserving changes in wording or length.**

---

## The question, and why it is hard to measure

Given a Reddit post, a model scores candidate summaries. We want to know whether its preference
tracks meaning or surface form. The design isolates three effects by comparing four responses:

| response | description |
|---|---|
| `chosen` | the preferred summary |
| `rejected` | the dispreferred summary |
| `rejected″` | rejected's **meaning**, rewritten to `chosen`'s **length** |
| `rejected′` | rejected's **meaning**, at `chosen`'s **length and syntax** |

```
g1 = score(chosen)     − score(rejected′)    → differs only in MEANING   (want large)
g2 = score(rejected′)  − score(rejected″)    → differs only in SYNTAX    (want ~0)
g3 = score(rejected″)  − score(rejected)     → differs only in LENGTH    (want ~0)
```

Three things make the naive version of this measurement wrong, and all three are corrected here:

1. **Averaging cancels.** `mean(g2)` answers "is there a *systematic* syntax preference?", not
   "does the model respond to syntax?" Effects of opposite sign average to ≈0. One model in this
   study has `mean(g2) = −0.86` while its per-example `|g2|` is 25 nats — larger than its entire
   semantic gap.
2. **Summed log-probabilities carry a length term.** A summed log-prob drops ~2.5 nats per extra
   token regardless of quality, so `g2`/`g3` cannot reach 0 unless the responses are token-length
   identical. Each gap is therefore split exactly into a *content* and a *length* component.
3. **There is no natural scale.** Any two different sentences score differently. Is `|g2| = 21`
   large? Unanswerable without a reference — see the noise floor below.

### The noise floor

Every example carries a **second** version of each counterfactual, written to identical
instructions (same meaning, same length, same syntactic target). Two such twins are
interchangeable, so their score gap is pure wording noise. Every other gap is reported in units
of it:

```
meaning score = |g1| / noise      syntax score = |g2| / noise
ratio         = meaning score / syntax score
```

A ratio of 2.3 means the model responds to a change in meaning ~2.3× as strongly as to a change
in wording. **This ratio is the headline metric.**

---

## Results

### Preference optimization barely shifts the meaning-vs-surface balance

19 checkpoints re-scored on the benchmark (`metrics/audit_all_methods_v2.csv`). CIs bootstrapped
over prompts.

| method | accuracy | meaning | syntax | **ratio** | vs untrained |
|---|---|---|---|---|---|
| `drdpo_strat` | 0.589 | 3.506 | 1.479 | **2.371** | +0.242 [+0.16, +0.32] |
| DPO v3 | 0.580 | 3.535 | 1.508 | 2.345 | +0.216 [+0.15, +0.28] |
| Dr.DPO | 0.592 | 3.457 | 1.498 | 2.308 | +0.179 [+0.09, +0.27] |
| … 13 methods in 2.23–2.32 … | | | | | |
| `drdpo_i4b` | 0.589 | 3.484 | 1.618 | 2.153 | +0.024 [−0.05, +0.10] **n.s.** |
| **untrained baseline** | 0.518 | 3.238 | 1.521 | **2.129** | — |
| `drdpo_i4` | 0.581 | 3.434 | 1.630 | 2.106 | −0.023 |

- Training improves the ratio from **2.13 → 2.37** — real, but ~11%.
- It does so **entirely by raising meaning-sensitivity** (3.24 → 3.51). **Syntax sensitivity does
  not move** (1.52 → 1.48): every method, and the untrained model, sits at ~1.5× the noise floor.
- The un-normalized semantic gap roughly doubles (20.5 → 39.7 nats) — but the noise floor grows
  nearly as much (8.7 → 14.7). **Most of preference optimization is amplification, not
  discrimination.**
- Methods cluster in a narrow band despite very different objectives, suggesting a base-model
  ceiling.

### Averaged metrics can crown a method that does nothing

`drdpo_i4b` held this project's internal record under the averaged metric. Under corrected
measurement it is **indistinguishable from the untrained model**. It has the *lowest* systematic
length bias but the *second-highest* per-example length sensitivity — it favours long on some
examples and short on others, cancelling to ≈0. Its training penalty optimizes exactly that
average. It optimized the statistic that cancels.

### A worst-case-over-counterfactuals objective loses to its own control

We implemented **Support-Augmented KL-Dr.DPO**: a KL-regularized DRO over each example's
counterfactual orbit, nested inside Dr.DPO's batch dual (`length_pref.py`). 6 arms × 3 seeds, all
seeing identical text:

| arm | accuracy | ratio |
|---|---|---|
| no counterfactuals | 0.5191 | 2.187 |
| worst-case DRO, p=(.85,.10,.05) | 0.5205 | 2.216 |
| worst-case DRO, p=(⅓,⅓,⅓) | 0.5309 | 2.300 |
| **counterfactuals as ordinary pairs** | **0.5452** | **2.353** |

Simply adding the counterfactuals as ordinary preference pairs wins, at matched text and matched
optimizer steps. The nominal weights give the counterfactuals only ~15% of the gradient versus
67% for the flat control, and benefit tracks that weight monotonically. **The worst-case
machinery contributes nothing.**

### Counterfactual data looks far more sample-efficient (provisional)

Ratio **2.353 from 1998** counterfactual examples vs **2.371 from 92,858** ordinary ones — but
accuracy does not follow (0.545 vs 0.589), and the comparison varies data source, size and
training length simultaneously. Treat as a lead, not a result.

---

## The benchmark

`data/decomposed_v2/decomposed_v2_validation.parquet` — 1979 examples, **1979 unique prompts**,
derived from the validation split of [`trl-lib/tldr-preference`](https://huggingface.co/datasets/trl-lib/tldr-preference).

```
prompt, chosen, rejected,
rejected_double_prime, rejected_prime,            # the two counterfactuals
rejected_double_prime_v2, rejected_prime_v2       # their twins → the noise floor
```

Counterfactuals were written by LLM agents (a writer plus an independent adversarial auditor per
batch, 420 agents total) and then mechanically QC'd. Guarantees:

- every counterfactual is **exactly** `chosen`'s token length under the Qwen2.5 tokenizer
- `rejected′` and `rejected″` are structural rearrangements sharing vocabulary (0.709 character
  similarity), not unrelated paraphrases
- one row per unique prompt — no clustering
- no degenerate rows

A disjoint 1998-example **training** set built the same way is in `data/decomposed_v2_train/`,
along with a `_flat` variant that expands each counterfactual into its own preference row.

> **Known limitation.** Meaning preservation was verified *mechanically* (token length, string
> similarity), never by a human or LLM judge. If the writing agents drifted the semantics, `g1`
> is contaminated. This validation is the top open item.

---

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# the headline table
.venv/bin/python scripts/audit_all_methods.py --eval v2

# score a checkpoint on the benchmark
.venv/bin/python scripts/eval_length_split_checkpoint.py \
  --checkpoint <path-or-hf-id> --label mymodel \
  --eval_dataset parquet --data_files data/decomposed_v2/decomposed_v2_validation.parquet \
  --split train \
  --response_keys chosen,rejected_prime,rejected_double_prime,rejected,rejected_prime_v2,rejected_double_prime_v2 \
  --out_csv metrics/mymodel_v2.csv

# train a counterfactual arm (see the script for all 6)
bash scripts/run_round9.sh flat_sm 1

# unit tests for the Support-Augmented KL-Dr.DPO implementation (34 cases)
.venv/bin/python -m unittest test_counterfactual_kl_dro
```

Model checkpoints are not included (~2.6 TB). All experiments use
`Qwen/Qwen2.5-1.5B-Instruct` as the base model.

---

## Layout

```
train.py                     entry point: --method {dpo,simpo,sipo,orpo,rdpo,drdpo,sampo,lmpo,tie}
length_pref.py               R-DPO / Dr.DPO / SamPO / LMPO / Tie + Support-Augmented KL-Dr.DPO
sipo.py, invariance_train.py earlier bespoke methods (SIPO; the I2–I5 invariance family)
test_counterfactual_kl_dro.py

scripts/
  decomposition_metrics.py   the corrected metric: content/length split, noise floor, clustered CIs
  audit_all_methods.py       re-ranking driver → metrics/audit_all_methods_v2.csv
  eval_length_split_checkpoint.py
  assemble_decomposed_v2.py  QC + assembly of agent output into the benchmark parquet
  wf_decomposed_v2.js        the agent workflow that writes the counterfactuals
  v2_shard_status.py         resumable generation state
  run_round9.sh, eval_all_on_v2.sh, ...

data/, metrics/              the benchmark and every CSV behind the tables above
```

---

