# SimPO vs DPO on TL;DR — Decomposed Log-Probability Report

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`
**Train data:** `trl-lib/tldr-preference`
**Eval data:** `Bojian92/tldr_preference_decomposed` (validation, n=256), scored every 10 training steps
**Stack:** `trl==0.29.1`, `transformers==4.57.3`, `torch==2.7.1+cu128`, `peft==0.18.0`
**Env:** `/data/users/bojianhou/projects/envs/pref_align_env`
**Date:** 2026-06-17 (updated 2026-06-21 with DPO v3; 2026-06-22 with §9 follow-ups; 2026-08-19 with §9.10 / Round 7)

---

## 1. Summary (TL;DR)

- **DPO v3 (5× LR = 5e-6, 2 epochs) finally moves — and the most of any run: margin +2.63 nats (6.44 → 9.08, peak 10.55).** The earlier "DPO is conservative" conclusion was *also* an LR-budget artifact: DPO v2 was only given a 2× LR bump while SimPO v2 got 5×. Matched at 5× LR, DPO's margin moves ~4× more than SimPO v2's.
- **But DPO v3's gain is driven by likelihood displacement, not by making the chosen response more likely.** Absolute `chosen_logp` *drops* ~20 nats over the run; rejected just drops ~22, so the *gap* widens. SimPO v2, by contrast, *raises* chosen (+2.4) while suppressing rejected — a "healthier" shape. See §5.4.
- **Accuracy `P(chosen>rejected)` does not improve for any run** (~0.60–0.66 throughout, including DPO v3). The bigger margins are about magnitude/separation, not ordering correctness.
- **SimPO v2 also trains:** margin +0.60 nats (6.44 → 7.05), concentrated in epoch 2; it most strongly suppresses the fully-rejected response.
- **DPO v2 was essentially flat** (+0.07 nats) — but that reflects its smaller 2× LR, not a property of DPO (see DPO v3).
- Methods are **authentic** TRL implementations (real `DPOTrainer` / experimental `CPOTrainer` with `loss_type="simpo"`), not reimplementations.
- **Standing best on the length-robustness metric (as of Round 7, §9.10):** `worst_subset` = min(acc_chosen_longer, acc_rejected_longer) = **0.672** with a length gap of **0.001**, from Dr.DPO + the I4 raw-margin invariance penalty at λ=2.5e-4. Progression: DPO v3 0.607 → round-5 I4 0.638 → round-7 0.672. Overall accuracy still does not move (~0.66) — the gains are in *balance* across length subsets, not in ranking correctness.

---

## 2. Method authenticity

The training code (`train.py`) does **not** copy code from the reference URLs — it imports the genuine TRL classes, so authenticity rides on the installed `trl==0.29.1`, not on web access.

| Method | Implementation | Config |
|---|---|---|
| DPO | `trl.DPOTrainer` | `loss_type="sigmoid"`, `beta=0.1`, `ref_model=None` (policy cloned as frozen reference — standard DPO) |
| SimPO | `trl.experimental.cpo.CPOTrainer` | `loss_type="simpo"`, `simpo_gamma=0.5`, `cpo_alpha=0.0` (pure SimPO), `beta=2.0` — the officially documented way to run SimPO |

**Evidence from training logs** (the numbers are diagnostic):

| Signal | DPO | SimPO | Interpretation |
|---|---|---|---|
| metric keys | `rewards/*`, `logps/*`, `logits/*` | same **+ `nll_loss`** | exact `DPOTrainer` vs `CPOTrainer` signatures |
| `logps/chosen` | ≈ −76 (summed) | ≈ −2.97 (length-normalized) | SimPO length normalization is active — its defining feature |
| `rewards/chosen` | small | ≈ −5.94 = `beta × mean_logp` (2.0 × −2.97) | confirms β=2.0 + normalized implicit reward |
| `nll_loss` | absent | 0.0 | matches `cpo_alpha=0.0` (pure SimPO) |
| initial `loss` | ≈ 0.69 = −ln(0.5) | — | correct sigmoid-DPO init when policy == ref |

---

## 3. Experiment configurations

| | DPO v1 | DPO v2 | DPO v3 | SimPO v1 | SimPO v2 |
|---|---|---|---|---|---|
| peak LR | 5e-7 | **1e-6** (2×) | **5e-6** (5×) | 1e-6 | **5e-6** (5×) |
| epochs | 1 | **2** | **2** | 1 | **2** |
| total steps | 5804 | 11608 | 11608 | 5804 | 11608 |
| beta | 0.1 | 0.1 | 0.1 | 2.0 | 2.0 |
| simpo_gamma | — | — | — | 0.5 | 0.5 |
| LR schedule | cosine, warmup 0.1 | cosine, warmup 0.1 | cosine, warmup 0.1 | cosine, warmup 0.1 | cosine, warmup 0.1 |
| results dir | `results/core_v1_conservative/dpo` | `results/core_v2_aggressive/dpo` | `results/core_dpo_v3/dpo` | `results/core_v1_conservative/simpo` | `results/core_v2_aggressive/simpo` |

"Aggressive" = higher peak LR + a second epoch. β / γ / cpo_alpha unchanged. **DPO v3 raises peak LR to 5e-6 to match SimPO v2's 5× multiplier** (β kept at 0.1 — the run's logs/wandb do not record a β override; peak LR 5e-6 confirmed from `logs/dpo_v3.log`).

---

## 4. Completion status (v2 / v3 runs)

| Check | DPO v2 | DPO v3 | SimPO v2 |
|---|---|---|---|
| `Done.` marker in log | ✅ | ✅ | ✅ |
| `final/` checkpoint | ✅ `checkpoints/dpo-strong/final` | ✅ `checkpoints/dpo-v3/final` | ✅ `checkpoints/simpo-strong/final` |
| Eval summary (1162 evals → step 11608) | ✅ | ✅ | ✅ |
| Live training process | none | none | none |

---

## 5. Results

### 5.1 Overall preference margin & accuracy

Margin = mean over 256 val examples of the sum-logp gap `chosen − rejected`.

| Run | steps | margin start → end | Δ margin | margin max | acc end | acc max | verdict |
|---|---|---|---|---|---|---|---|
| DPO v1 | 5804 | 6.44 → 6.38 | **−0.06** | 6.58 | 0.605 | 0.609 | flat |
| DPO v2 | 11608 | 6.44 → 6.51 | **+0.07** | 6.66 | 0.602 | 0.660 | ~flat (2× LR) |
| DPO v3 | 11608 | 6.44 → **9.08** | **+2.63** | **10.55** | 0.602 | 0.664 | **moves most (5× LR)** |
| SimPO v1 | 5804 | 6.44 → 6.40 | **−0.04** | 6.63 | 0.559 | 0.664 | flat |
| SimPO v2 | 11608 | 6.44 → 7.05 | **+0.60** | 7.30 | 0.605 | 0.660 | clearly moving |

Note: **accuracy does not improve in any run** (all ~0.60–0.66, DPO v3 included). Margin growth ≠ better ordering.

### 5.2 Decomposition component gaps (start → end)

| gap (sum logp) | DPO v2 | DPO v3 | SimPO v2 |
|---|---|---|---|
| chosen − rejected′ | +15.9 → +16.0 | +15.9 → **+17.4** | +15.9 → **+19.1** |
| rejected′ − rejected″ | −9.45 → −9.27 | −9.45 → **−7.14** | −9.45 → **−6.82** |
| rejected″ − rejected | −0.05 → −0.26 | −0.05 → **−1.13** | −0.05 → **−5.20** |

DPO v3 and SimPO v2 move the same gaps in the same directions, but with different emphasis: **DPO v3 spreads its effect across the chain**, while **SimPO v2 concentrates on crushing the fully-rejected tail** (`rejected″ − rejected`: −5.20 vs DPO v3's −1.13).

### 5.3 When does SimPO v2 move?

Margin sampled across the run (0 / 25 / 50 / 75 / 100% of steps):

- **DPO v2:** +6.44 / +6.45 / +6.47 / +6.47 / +6.51 — flat throughout.
- **DPO v3:** +6.44 / +8.63 / +9.04 / +9.11 / +9.08 — **lifts off in epoch 1, plateaus by mid-run.** Most of the gain is banked before the halfway point.
- **SimPO v2:** +6.44 / +6.42 / +6.94 / +6.98 / +7.05 — **flat through epoch 1, lifts off in epoch 2.**

DPO v3 (5× LR) moves earlier *and* further than SimPO v2 (5× LR) — a 5× learning rate, not a second epoch, is what unlocks DPO here. SimPO needs the second epoch because its length-normalized loss with β=2.0 / γ=0.5 takes longer to build the margin.

### 5.4 Where does DPO v3's margin come from? (likelihood displacement)

Absolute mean response log-prob over the val set, start → end:

| run | chosen_logp | rejected_logp | reading |
|---|---|---|---|
| DPO v2 | −90.8 → −89.8 (+0.9) | −97.2 → −96.4 (+0.9) | both barely move |
| **DPO v3** | **−90.8 → −110.6 (−19.8)** | **−97.2 → −119.7 (−22.5)** | **both pushed down ~20 nats; rejected pushed down more → gap widens** |
| SimPO v2 | −90.8 → −88.4 (+2.4) | −97.2 → −95.4 (+1.8) | both rise slightly; chosen rises most |

DPO v3's large margin is **likelihood displacement**: the model becomes ~20 nats *less* likely to produce even the chosen summary; the margin grows only because rejected falls faster. This is a known DPO failure mode (the loss only constrains the *difference* of log-ratios, not their level). SimPO's length-normalized objective keeps absolute log-probs roughly stable. The training log corroborates it: DPO v3 `logps/chosen` falls −76 → −95 while `rewards/margins` climbs 0 → 0.51.

---

## 6. Plots (`results/core_v2_aggressive/`)

| File | Comparison | What it shows |
|---|---|---|
| `decomposed_cross_method.png` | **between methods** | SimPO vs DPO, one panel per metric (v2) |
| `decomposed_within_method.png` | **between metrics** | decomposition gaps overlaid within each method (v2) |
| `decomposed_v1_vs_v2.png` | **aggressive vs conservative** | solid = v2, dashed = v1, per method — SimPO red-solid lifts off; all others hug baseline |
| `results/core_dpo_v3/decomposed_v3_comparison.png` | **DPO v1/v2/v3 + SimPO v2** | margin (DPO v3 dark-blue rockets to ~9–10), accuracy (all flat ~0.6), and absolute chosen/rejected logp showing DPO v3's displacement (both fall) vs SimPO v2 (both stable) |
| `results/core_dpo_v3/task1_absolute_logp_4objects.png` | **entanglement** | absolute logp of all 4 objects per run; DPO v3's four curves plunge ~20 nats in lockstep, SimPO v2's fan apart (§9.1) |
| `results/ablation_simpo_sumlogp/length_bias_comparison.png` | **length-bias ablation** | SimPO with vs without length normalization — margin, %-favoring-longer, and corr(per-token margin, length) (§9.3) |
| `results/ablation_simpo_sumlogp/normalization_effect.png` | **normalization effect** | len-norm vs sum-logp SimPO, 6-panel per metric — isolates how normalization redistributes the decomposition (§9.4a) |
| `results/ablation_simpo_sumlogp/length_split_{lennorm_v2,sumlogp}.png` | **length-direction split** | all metrics split by chosen-longer vs rejected-longer; accuracy ~0.21 vs ~0.90 (§9.4b) |
| `results/ablation_simpo_sumlogp/rpp_drop_mechanism.png` | **rejected″ drop mechanism** | sum vs per-token gap + absolute per-token logp; drop is per-token, normalization-caused (§9.5) |
| `results/length_split/arms_accuracy_by_length.png` | **retrain-on-length-split** | eval accuracy by length direction for chosen_longer/rejected_longer/mixed training arms; brevity heuristic is an objective bias, not a data artifact (§9.6) |
| `results/length_split/sipo_g1_g3_frontier.png` | **g1⊥g3 Pareto frontier** | every SIPO config + SimPO/DPO/base on the semantics-vs-length plane; target corner (high g1 & g3≈0) unreached — SimPO-green + DPO-purple can't coexist (§9.7) |

(Conservative v1 equivalents live in `results/`.)

---

## 7. Conclusions

1. **DPO is *not* inherently conservative on this setup — it was under-LR'd.** The v2 "DPO barely moves" result was an artifact of giving DPO only a 2× LR bump while SimPO got 5×. Matched at 5× LR (v3), DPO produces the **largest** margin shift of any run (+2.63 nats, peak +10.55), and it does so early (within epoch 1). This *supersedes* the prior report's conclusion #3.
2. **But DPO v3's margin is largely likelihood displacement, not preference learning.** Absolute `chosen_logp` drops ~20 nats; the margin grows only because rejected drops faster (§5.4). SimPO v2 instead keeps absolute log-probs stable / slightly rising — a qualitatively healthier optimization, even though its margin gain is smaller.
3. **No method improves ranking accuracy.** `P(chosen>rejected)` stays ~0.60–0.66 everywhere. On this 1.5B model + TL;DR setup, both objectives reshape log-prob *magnitudes* without flipping many pairwise decisions.
4. **SimPO works once given enough LR / steps** and most strongly suppresses the fully-rejected tail (rejected″−rejected: −0.05 → −5.20), concentrating its effect at the bottom of the decomposition chain.
5. **The earlier "quick convergence" was a training-budget artifact** (too-low LR, single epoch) for *both* methods — not evidence either fails.

## 8. Recommended next steps

- **Quantify DPO v3's displacement vs. usefulness:** generate samples from `checkpoints/dpo-v3/final` and judge summary quality (length, coherence) — a 20-nat drop in chosen likelihood may signal degradation despite the wider margin.
- **DPO with displacement controls:** re-run 5× LR with **lower β** (0.05) and/or an SFT/NLL anchor term (or `loss_type` variants like `ipo`) to see whether the margin can be earned *without* tanking absolute chosen likelihood.
- **Add a KL-to-reference / absolute-logp panel to the in-training eval** so displacement is visible live, not just post-hoc.
- **SimPO:** sweep `simpo_gamma`; confirm the trend holds beyond 2 epochs; check for over-optimization.
- **Eval consistency:** the in-training callback (`train.py`) and standalone `evaluate_decomposed_tldr_logps.py` differ in `max_len` (1024 vs 2048), `prompt_truncation_side` (left vs right), and EOS handling — align them before comparing absolute numbers across the two eval paths.
- **Record the launch config** (β, LR) into each run's dir — v3's β had to be inferred because wandb offline configs weren't persisted.

---

## 9. Follow-up analyses (added 2026-06-22)

Follow-up tasks are run in batches ("rounds"). This report is the single up-to-date superset; each round also has a frozen standalone digest. **Round 1** (§9.1–§9.3 + §5): object-level entanglement (§9.1), a dataset length audit (§9.2), and a SimPO length-normalization ablation (§9.3) → `FOLLOWUP_TASKS_ROUND1.md`. **Round 2** (§9.4–§9.5, added 2026-06-29): normalization redistribution (§9.4a), length-direction accuracy split (§9.4b), and the `rejected″` drop mechanism (§9.5) → `FOLLOWUP_TASKS_ROUND2.md`. **Round 3** (§9.6, added 2026-06-30): retrain-on-length-split — is the brevity heuristic a data artifact or an objective bias? → `FOLLOWUP_TASKS_ROUND3.md`. **Round 4** (§9.7, added 2026-07-05): SIPO + auto-research loop — can a new method combine SimPO's g1 with DPO's g3? (honest negative result) → `FOLLOWUP_TASKS_ROUND4.md`. **Round 5** (§9.8, added 2026-07-12): DPO v3 on the FULL test set with length split — accuracy + g1/g2/g3; DPO v3 is ~5× more length-neutral than SimPO v2; plus two follow-ups — group-DRO (no effect) and a 36-config invariance auto-research sweep (I2–I5; winner DPO v3 + I4, worst 0.638) → `FOLLOWUP_TASKS_ROUND5.md`. **Round 6** (§9.9, added 2026-07-27): eight published/native *off-the-shelf* length-robust methods (ORPO, LD-DPO, rDPO, R-DPO, Dr.DPO, SamPO, LMPO, Tie Training) trained + full-test length-split eval — only Dr.DPO/R-DPO marginally beat DPO v3; the accuracy ceiling and worst⊥g3 wall survive all of them → `FOLLOWUP_TASKS_ROUND6.md`. **Round 7** (§9.10, added 2026-08-19): per-subset g1/g2/g3 for all 19 checkpoints, a mechanistic account of Dr.DPO (length-blind outlier suppression — §9.9 finding 1 corrected), and eight Dr.DPO-tuning configs — **new record `worst_subset` 0.672** (Dr.DPO + I4 λ=2.5e-4, length gap 0.001), which also weakens the §9.8 worst⊥g3 wall → `FOLLOWUP_TASKS_ROUND7.md`. **Round 8** (§9.11, added 2026-08-26): **Support-Augmented KL-Dr.DPO** — a KL-*regularized* (fixed-temperature) DRO over each example's rejected counterfactual orbit (rejected / rejected″ / rejected′) placed INSIDE the unchanged Dr.DPO batch dual; worst-case-over-orbit accuracy +6 pts (0.473→0.543) and P(chosen>rejected″) +9 pts, with plain accuracy and the length gap unmoved → `FOLLOWUP_TASKS_ROUND8.md`. **Round 9** (§9.12, added 2026-08-30): executes the audit — a rebuilt eval set (1979 unique prompts vs 195, token-exact counterfactuals, twin variants giving a **noise floor**), a corrected re-ranking of all 19 checkpoints that finds the round-7 champion `drdpo_i4b` **statistically indistinguishable from the untrained model** (+0.024, CI crosses 0) while `drdpo_strat`/DPO v3/Dr.DPO are genuinely better, and 18 seeded training runs showing **Support-Augmented KL-Dr.DPO loses to its own information- and step-matched control** — the counterfactual data helps, the worst-case aggregation does not → `FOLLOWUP_TASKS_ROUND9.md`.

> **AUDIT (2026-08-29) — read `AUDIT_2026-08-29.md` before trusting any g2/g3 number below.**
> A full review against the original research question found three measurement defects that
> affect every round: (1) summed g2/g3 carry a mechanical length floor of ~−1 to −1.5 nats, so
> "→0" was never reachable; (2) all rounds reported the *mean* gap, which cancels — per example
> |g2| is ~15–26 nats, larger than g1 itself, and training makes it worse; (3) the eval set has
> only **195 unique prompts** across its 2000 rows, so error bars are ~2× the naive ones (the
> round-7 headline survives this; round-6's margins do not). It also adds the
> **untrained-baseline** evaluation that never existed. Net: the "no systematic length/syntax
> bias" claim holds and is real; the stronger "judges by meaning, not surface form" claim is
> false for ~44% of examples, for every method tried.

### 9.1 Entanglement of the four objects

Plotting the **absolute** log-prob of each object (`chosen`, `rejected′`, `rejected″`, `rejected`) separately — rather than the gaps — exposes how coupled they are. Plot: `results/core_dpo_v3/task1_absolute_logp_4objects.png`.

| Run | chosen Δ | rejected Δ | per-step corr(Δchosen, Δrejected) | reading |
|---|---|---|---|---|
| DPO v2 | +0.9 | +0.9 | +0.68 | flat, mild coupling |
| **DPO v3** | **−19.8** | **−22.5** | **+0.98** | **near-perfectly entangled — all 4 fall in lockstep** |
| SimPO v2 | +2.4 | +1.8 | +0.93 | curves visibly diverge (chosen ↑, rejected″ ↓−3.3) |

**DPO v3's "margin gain" is mostly a rigid downward shift of the whole distribution** (correlation 0.98), confirming §5.4's likelihood-displacement reading from a different angle. SimPO v2 reshapes the chain (objects separate) rather than translating it.

### 9.2 Length audit of `Bojian92/tldr_preference_decomposed`

Splits are **train = 3000, validation = 2000** (training eval'd only the first 256 of validation). Length via the Qwen2.5 tokenizer:

| Split | `len(chosen) > len(rejected)` (tokens) | ties | mean(chosen − rejected) |
|---|---|---|---|
| train | 1672 / 3000 = **55.7%** | 62 | +3.5 tok (median +3) |
| validation | 1031 / 2000 = **51.5%** | 109 | +0.7 tok (median +1) |

A mild skew toward chosen being longer (stronger in train). Incidental: by **word** count chosen is *never* longer than `rejected′`/`rejected″` (0%) though it is in ~34–40% by tokens with many ties — the decomposition stages appear constructed to preserve/extend word length relative to chosen.

### 9.3 SimPO length-normalization ablation (`average_log_prob=False`)

**Setup.** TRL hardcodes SimPO's length normalization at `cpo_trainer.py:804`; `train.py` now exposes `--no-simpo_average_log_prob`, which overrides `get_batch_logps` in a `CPOTrainer` subclass to use **summed** (not length-averaged) log-probs. Run: lr 5e-6, 2 epochs (matching SimPO v2), into `results/ablation_simpo_sumlogp/` + `checkpoints/simpo-sumlogp/`.

> **Caveat — not a single-variable ablation.** Summed log-probs are ~−90 nats, so β=2.0 saturates `logsigmoid` and the model won't train. β had to be lowered to **0.1**, so this run varies *both* length-normalization (off) **and** β (2.0→0.1). Read it as "sum-logp SimPO at a trainable β," not a pure flip of one flag.

**Length-bias metric.** Per eval example, compare the model's **per-token** preference (`chosen_mean_logp − rejected_mean_logp`) against which response is longer. "% favoring longer" = share of pairs where the per-token-preferred response is the longer one; 50% = length-neutral. (Per-token, not summed, to strip the mechanical "more tokens → more negative sum" effect.) Plot: `results/ablation_simpo_sumlogp/length_bias_comparison.png`.

| Run (end of training) | sum margin | % favoring longer | corr(per-token margin, len-diff) | acc gap (chosen-longer − chosen-shorter) |
|---|---|---|---|---|
| init (step 0) | 6.44 | 46.8% | −0.083 | −0.45 |
| **SimPO v2 (len-norm, β=2.0)** | **7.05** | **42.8%** (↓ prefers shorter) | −0.256 | −0.69 |
| **SimPO sum-logp (no norm, β=0.1)** | 6.68 | **46.8%** (≈ neutral, holds) | −0.157 | −0.71 |
| DPO v3 (sum, β=0.1) | 9.07 | 49.6% (↑ slightly longer) | −0.007 | −0.38 |

**Findings.**
1. **Length normalization actively pushes SimPO toward shorter responses.** Normalized SimPO v2 drops "% favoring longer" 46.8 → 42.8 and drives `corr(per-token margin, length)` more negative (−0.26) — i.e. it learns to prefer the more concise response. This is the length-debiasing SimPO is designed for.
2. **Turning normalization off removes that shortening pressure.** Sum-logp SimPO holds at ~46.8% (≈ the untrained, near-neutral level) with a weaker length–preference correlation (−0.16) — it is **more length-tolerant / more length-biased toward longer** than the normalized version, exactly the predicted effect.
3. **Cost: weaker preference learning.** Sum-logp SimPO's margin grew only +0.24 nats (6.44 → 6.68) vs v2's +0.60 — at β=0.1 it separates chosen/rejected less. (Confounded with β, per the caveat.)
4. DPO v3 (also summed log-probs) ends up the **most length-neutral per token** (corr −0.007), so "summed log-prob" alone does not imply length bias — the data skew (§9.2) and β/normalization interact.

**Reproducibility note (added 2026-06-23).** The sum-logp run was re-executed end-to-end (2 epochs, full 11608 steps) with the eval protocol matched to v2/v3 (`--max_eval_samples 256`, eval every 10 steps). Every length-bias metric above reproduces exactly: `corr(per-token margin, len-diff)` −0.157, % favoring longer 46.8%, sum margin 6.68 (v2 reference: −0.256 / 42.8%). Confirms the §9.3 conclusions are not an artifact of eval-set size. Analysis script: `analyze_length_bias.py` → `length_bias_summary.csv`.

### 9.4 Normalization effect on the decomposition chain + length-direction split (added 2026-06-29)

Two complementary views of how length normalization reshapes the decomposition, both at matched eval protocol (n=256, every 10 steps). Plots: `results/ablation_simpo_sumlogp/normalization_effect.png` (6-panel, len-norm vs sum-logp overlaid — same layout as `decomposed_cross_method.png` but isolating normalization) and `results/ablation_simpo_sumlogp/length_split_{lennorm_v2,sumlogp}.png`. Scripts: `plot_normalization_effect.py`, `analyze_length_split.py` → `length_split_metrics.csv`.

**(a) Normalization redistributes the margin without changing the net outcome.** End-of-training component values (avg over last 20 evals); identity `C−R = (C−R′)+(R′−R″)+(R″−R)` holds for both:

| gap (sum logp) | len-norm v2 | sum-logp | Δ (sum − norm) |
|---|---|---|---|
| chosen − rejected (net) | 7.065 | 6.685 | −0.38 |
| chosen − rejected (mean logp, **objective**) | 0.108 | 0.104 | −0.004 |
| chosen − rejected′ | 19.042 | 12.761 | **−6.28** |
| rejected′ − rejected″ | −6.835 | −7.274 | −0.44 |
| rejected″ − rejected | −5.141 | **+1.198** | **+6.34** |
| accuracy P(chosen>rejected) | 0.606 | 0.602 | −0.004 |

Net margin (~7 vs ~6.7) and accuracy (~0.60) are nearly identical, but normalization **redistributes where the margin lives**: two large offsetting shifts — `chosen−rejected′` collapses (19.0→12.8) while `rejected″−rejected` flips sign (−5.14→+1.20). See §9.5 for the mechanism behind the sign flip.

**(b) Accuracy is almost entirely a length artifact.** Splitting the eval set by length direction (chosen_tokens > vs < rejected_tokens; ties dropped) and recomputing every metric:

| run | subset | n | acc | sum_C−R | mean_C−R | C−R′ | R′−R″ | R″−R |
|---|---|---|---|---|---|---|---|---|
| len-norm v2 | chosen longer | 106 | **0.208** | −18.5 | −0.068 | 13.4 | −12.8 | −19.1 |
| len-norm v2 | rejected longer | 144 | **0.897** | +25.9 | +0.235 | 23.5 | −2.5 | +4.9 |
| sum-logp | chosen longer | 106 | **0.198** | −16.8 | −0.032 | 7.4 | −15.1 | −9.2 |
| sum-logp | rejected longer | 144 | **0.903** | +24.0 | +0.201 | 17.0 | −1.5 | +8.6 |

The headline ~0.60 accuracy decomposes into **~21% when the correct (chosen) answer is longer vs ~90% when it is shorter** — both variants essentially **rank by brevity**. The flip persists per-token (`mean_C−R` sign-flips with length direction), so it is not purely the mechanical "more tokens → more negative sum." Length normalization barely dents this: the accuracy split is near-identical (0.21/0.90) for both variants. This sharpens conclusion #3 (§7): *no method improves ranking* — what ~0.60 hides is a near-deterministic length preference.

### 9.5 Why `rejected″ − rejected` drops ~5 nats under len-norm SimPO (added 2026-06-29)

The `rejected″ − rejected` (sum-logp) curve falls −0.05 → −5.14 over len-norm SimPO training (the bottom-right panel of `decomposed_cross_method.png`). Hypothesis tested: rejected″ is longer than rejected and SimPO prefers shorter. Scripts: `audit_lengths.py`, `investigate_rpp_drop.py` → `results/ablation_simpo_sumlogp/rpp_drop_mechanism.png`.

**Length facts (exact eval tokenizer).** rejected″ is constructed to match **chosen's** length (corr 0.96 train / 0.91 val), so on train it is longer than rejected 55.6% of the time (+2.34 tok), mirroring chosen>rejected (55.7%). But on the full val it is ~neutral (49.6%, +0.33 tok) and on the n=256 eval slice it is incidentally ~1.8 tok **shorter** (38.3% longer).

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

Length contributes ~nothing (+0.35). The drop is driven entirely by len-norm SimPO pushing rejected″'s **per-token** log-prob *down* (−0.089/tok) while the true rejected's *rises* (+0.058); over ~37 tokens that becomes the ~5-nat sum-logp fall.

**Decisive control: it's caused by normalization, not raw length.** The sum-logp run sees identical responses/lengths but does the **opposite** (gap rises to +1.20). Same data, opposite sign ⇒ length normalization itself produces the drop. Mechanism: normalization makes the loss reward per-token efficiency, so the more-verbose rejected″ (a chosen-length paraphrase of rejected content) gets a lower per-token probability than the terse rejected; **length is only the ~37× amplifier** that turns the small per-token gap into a large sum-logp drop. The original length hypothesis is right in spirit (a normalization/conciseness effect) but the drop is not produced by rejected″ being longer on the eval set.

### 9.6 Retrain on length-split training data: data artifact or objective bias? (added 2026-06-30)

§9.4b was a post-hoc slice of the eval set; it showed the mixed SimPO model ranks by brevity (acc ~0.21 chosen-longer vs ~0.90 chosen-shorter) but not *why*. Round 3 **retrains** to disambiguate: split the FULL trl-lib train (92,858) by response-length direction and train one len-norm SimPO model per direction, equal-N and identical hyperparameters (lr 5e-6, 2 epochs, β=2.0, γ=0.5), against the existing full-mixed run as control. If the heuristic is a *data artifact*, training only on chosen-longer pairs should teach "prefer longer" (acc[chosen longer] > 0.5); if it is an *objective bias*, brevity persists regardless. Scripts: `build_length_split.py` (Qwen2.5 tokenizer split → `results/length_split/split_indices.json`), `train.py --train_length_subset`, `analyze_arms_length_split.py`. Plot: `results/length_split/arms_accuracy_by_length.png`.

**Length-direction split of the full train set:** chosen_longer 51,750 (55.7%), rejected_longer 37,835 (40.7%), ties 3,273 (3.5%) → equal-N **37,835/arm** (seed 42).

End-of-training metrics (avg over last 20 evals), eval sliced by length direction:

| trained on | acc[chosen longer] | acc[rejected longer] | acc[all] | mean C−R[chosen longer] (per-token) |
|---|---|---|---|---|
| mixed baseline (control) | 0.208 | 0.897 | 0.606 | −0.068 |
| **chosen_longer arm** | **0.292** | 0.856 | 0.610 | **+0.863** |
| **rejected_longer arm** | **0.000** | **1.000** | 0.586 | −7.796 |

**Findings.**
1. **Primarily an objective/inductive bias, not a data artifact.** Training *only* on chosen-longer pairs did **not** flip the heuristic — acc[chosen longer] moved just 0.21 → 0.29 (still ≪ 0.5). The model resisted learning "longer = better."
2. **Sharp asymmetry (the key result).** Training only on rejected-longer pairs (correct = shorter) produced a **perfect brevity ranker, 0.000 / 1.000**. The objective eagerly absorbs "shorter = better" but fights "longer = better," independent of data direction.
3. **The ~0.60 ceiling is immovable** — all three conditions sit at 0.586–0.613. Rebalancing training-data length does **not** fix it (sharpens conclusion #3 / §9.4b).
4. **Objective-level nuance.** Per-token preference (`mean C−R`, SimPO's actual signal) *does* respond to the data — the chosen_longer arm's chosen-longer-subset per-token gap rose −0.068 → +0.863 (it learned to prefer the longer-correct content per token) — but length normalization prevents that per-token gain from overturning the summed ranking. Data controls the per-token direction; the objective controls the ranking.

### 9.7 SIPO + auto-research: can a method combine SimPO's g1 with DPO's g3? (added 2026-07-05)

Motivated by panel (d): the goal was a new method with **SimPO's semantic separation (g1~19) AND DPO's length-neutrality (g3~0)** simultaneously — i.e. learn preference by semantics, not length/syntax. Method **SIPO** (Semantics-Isolating Preference Optimization, `sipo.py`, `train.py --method sipo`): reference-free preference on the length+syntax-matched pair (chosen, rejected′) plus explicit invariance penalties on (rejected′,rejected″) and (rejected″,rejected). Searched by an autonomous `running-agent-loops` harness (`.loop/`): 40+ configs, 10 iterations + 1 restart, converged, escalated (`ESCALATION.md`) rather than faking success. Scorer `sipo_eval.py`; figure `results/length_split/sipo_g1_g3_frontier.png`.

**Result — negative (verified independently).** End-of-training, summed-logp scale:

| run | g1 (want ~19) | g3 (want ~0) | g2 | acc | length-acc-gap |
|---|---|---|---|---|---|
| SimPO v2 | **19.0** | −5.1 | −6.8 | .61 | .69 |
| DPO v3 | 17.4 | **−1.1** | −7.1 | .60 | — |
| SIPO iter3m (best of 40) | 16.7 | −1.27 | −9.0 | .60 | .459 |

**Findings.**
1. **The target is provably infeasible in-scope — g1 ⊥ g3 tradeoff.** Pushing semantic separation up drags length-neutrality down (iter16: g1=22.6→g3=−8.1; iter04: g1=20→g3=−5.3). SimPO's green and DPO's purple are opposite ends of one Pareto frontier; the target corner is unreached by 40+ configs. **SIPO iter3m is a slightly-worse DPO v3** (DPO: higher g1 17.4, better |g3| 1.1) — nothing beat DPO, and no method (incl. SimPO/DPO) satisfies g1≥18 ∧ |g2|≤2 ∧ |g3|≤2.
2. **Scale-dependent answer; the summed metric manufactures the length/syntax story.** Per-token (mean logp), *every* method incl. the base model has tiny g2/g3 (~0.1–0.3) with semantics dominant; the ~25× token multiplier inflates summed g2 to −9. iter3m: length-acc-gap 0.459 (sum) vs 0.038 (per-token). Per-token, SIPO gives no advantage — DPO v3 has the best profile (g1/|g2|=3.70), SIPO only 2.38.
3. So "does alignment learn semantics vs length/syntax?" is **scale-dependent**: per-token yes for all methods; on the summed scale length re-enters mechanically. A genuine `score≥0.8` needs a non-summed-confounded syntax metric and honest mean-scale anchors (Option C, not yet run).

### 9.8 DPO v3 on the FULL test set, length split — g1/g2/g3 + accuracy (added 2026-07-12)

Prior length-direction splits (§9.4b) used the n=256 in-training eval slice, flagged in §9.5 as short-biased. Here DPO v3 and SimPO v2 saved checkpoints are re-evaluated on the **full 2000-example** decomposed `validation` (chosen-longer 1031, rejected-longer 860, ties 109), matched encoding (max_len 1024, left trunc). Script: `scripts/eval_length_split_checkpoint.py`; data `metrics/length_split_fulltest_{DPO_v3,SimPO_v2}.csv`.

Sum-logp gaps (g1=semantics, g2=syntax, g3=length):

| method / subset | n | acc | g1 | g2 | g3 |
|---|---|---|---|---|---|
| **DPO v3** — chosen longer | 1031 | 0.607 | +18.96 | −1.10 | −8.61 |
| **DPO v3** — rejected longer | 860 | 0.717 | +14.30 | −2.60 | +5.84 |
| **DPO v3** — all | 2000 | **0.656** | +16.32 | −2.04 | **−1.73** |
| SimPO v2 — chosen longer | 1031 | 0.460 | +18.95 | −1.87 | −20.11 |
| SimPO v2 — rejected longer | 860 | 0.814 | +16.37 | −1.48 | +3.34 |
| SimPO v2 — all | 2000 | 0.621 | +17.39 | −1.84 | **−9.12** |

**Findings.**
1. **DPO v3 is decisively the most length-robust ranker:** overall |g3| = 1.73 vs SimPO v2's 9.12 (~5×), and length-acc-gap 0.110 vs 0.354. When the correct answer is longer, DPO still prefers it 60.7% of the time (>0.5, not brevity-biased); SimPO flips to 0.460.
2. **g1 (semantics) strong for both** (~+14 to +19), length-independent; g2 (syntax) small (−1 to −2.6). The methods differ on length, not semantics.
3. **The 256-slice overstated DPO's length bias** (slice gap 0.377 → full-test 0.110); the slice runs short (§9.5). Per-token, DPO v3 all-set g3 ≈ **−0.001** (vs SimPO −0.230) — near-perfect length invariance.
4. **This sharpens §9.7:** the "big g1 + ~0 g3" target SIPO was built to reach is closely met **by DPO v3 itself**, not by SIPO's reward-shaping — reinforcing the round-4 negative result. The g3 sign flip across subsets is mechanical (`rejected″` tracks chosen length); only |g3| magnitude is meaningful.

**Follow-up — length-group DRO does NOT remove the residual subset bias (added 2026-07-12).** DPO v3 still conditions on length within skewed subsets (acc 0.607 chosen-longer vs 0.717 rejected-longer). Tested: reference-based DPO with **group-DRO** over {chosen-longer, rejected-longer} (optimize the worst group) vs plain DPO, identical 3k setup (`dro_dpo_train.py --group_dro {off,worst}`), evaluated on the full 2000 test set:

| run (identical except objective) | acc[chosen longer] | acc[rejected longer] | length gap | overall acc | g3 (all) |
|---|---|---|---|---|---|
| plain DPO | 0.540 | 0.744 | 0.204 | 0.629 | −4.79 |
| DPO + group-DRO (worst) | 0.524 | 0.720 | 0.196 | 0.610 | −4.86 |

The gap barely moves (0.204→0.196, noise) and overall acc dips; DRO upweights the failing chosen-longer group but `acc[chosen longer]` does not rise (0.540→0.524). **The length bias is objective-level, not data-distributional** — neither exclusive-subset training (§9.6) nor loss reweighting removes it; reweighting only redistributes which subset is sacrificed. Length-invariance instead needs mean-logp ranking (per-token g3≈0 for free) or an explicit invariance term (trades against g1, the §9.7 wall).

**Follow-up 2 — brainstorm + auto-research for length/syntax-invariance (added 2026-07-13).** Goal: a trained method that is semantics-only and length/syntax-invariant *on a skewed test set*. Metric: worst_subset_acc = min(acc_chosen_longer, acc_rejected_longer), full 2000 test. Bars: DPO v3 0.607; a free one-parameter length-debias reaches ~0.64 (no training). Free levers: **I1 mean-logp ranking fails** (0.543, flips the bias); I6 debias ~0.64. Auto-research (`invariance_train.py`/`invariance_eval.py`, running-agent-loops, `results/inv_search/`) swept **36 configs** over I2 (length-matched negative), I3 (counterfactual consistency), I4 (equalize policy margin across length environments +EMA), I5 (IRM/V-REx), DRO.

| | worst-subset | length gap | acc_all | g1 | g3 |
|---|---|---|---|---|---|
| DPO v3 (start) | 0.607 | 0.110 | 0.656 | 16.3 | −1.7 |
| **winner: DPO v3 + I4 len-inv λ=5e-4** | **0.638** | **0.054** | 0.658 | 17.7 | −3.4 |
| clean-g3 config (I4+I3 cons3both) | 0.603 | 0.112 | 0.653 | 16.3 | −1.7 |

**Findings.** (1) Best trained method **halves the length-accuracy gap** (worst 0.607→0.638) but only **≈matches the free debias ceiling** and misses 0.68. (2) **worst-subset ⊥ g3 Pareto wall** — pushing accuracy-invariance up widens the raw-logp length gap; keeping g3 clean drops worst back to 0.603. (3) **acc_all ≈ 0.66 is model/data-bound** (via `worst ≈ acc_all − gap/2`, clearing 0.68 needs acc ~0.70 — unreachable on 1.5B+TL;DR). (4) I2 length-matched negatives blow up the decomposition (g1≈143, g3≈−146); I3 alone hurts. **Answer: partly** — decision-level length-invariance is improvable, but not simultaneously with raw-logp invariance, and the accuracy ceiling is model/data-bound; the best trained method barely beats a one-parameter post-hoc debias.

> **REVISED by Round 7 (§9.10.3).** Finding (2) is too strong — it generalized from a sweep
> that varied the invariance term on top of DPO v3 alone. Stacking the same I4 penalty on
> **Dr.DPO** at λ=2.5e-4 reaches worst **0.672** while holding all-set g3 at **−1.84**
> (DPO v3: −1.73) and *halving* g3 on the hard subset (−4.59 vs −8.61):
>
> | | worst | g3 (all) | g3[cl] |
> |---|---|---|---|
> | DPO v3 | 0.607 | −1.73 | −8.61 |
> | this round's winner (DPO v3 + I4 λ=5e-4) | 0.638 | −3.36 | −9.30 |
> | **Round 7 `drdpo_i4b` (Dr.DPO + I4 λ=2.5e-4)** | **0.672** | **−1.84** | **−4.59** |
>
> So decision-level and raw-logp length-invariance are **not** strictly anti-correlated —
> though not free either (g3[cl] is −4.59, not 0). Findings (1) and (3) stand: the record
> improves to 0.672, but by driving the length gap to ~0 (0.001), not by raising accuracy
> (0.667, n.s. vs DPO v3), exactly as `worst ≈ acc_all − gap/2` predicts. The gap term is now
> spent, so the ~0.66 accuracy ceiling is the sole remaining bound.

### 9.9 Off-the-shelf length-robust methods from the literature (added 2026-07-27)

Round 5 built *bespoke* invariance losses (SIPO §9.7, I2–I5 §9.8). Round 6 asks the complementary question: do **published** preference methods that *claim* length robustness/invariance actually reduce the TL;DR length bias, and can any beat DPO v3 (the §9.8 leader)? Eight **offline** methods were trained on the headline recipe (full `trl-lib/tldr-preference`, LR 5e-6, 2 epochs) and evaluated on the full 2000-example decomposed test, length-split (RL methods Dr.GRPO / GSPO — which need a reward model + rollouts — are deferred to Round 7). Native methods (ORPO `trl.experimental.orpo`; LD-DPO `ld_alpha`; rDPO `loss_type=robust`) use genuine TRL trainers; the custom losses (R-DPO 2403.19159, Dr.DPO 2407.07880, SamPO 2406.10957, LMPO 2502.14643, Tie Training 2605.11134) live in `length_pref.py`, whose plain-`dpo` mode reproduces TRL's DPO init loss (0.693) — formulas taken from each paper's authors' code. Digest: `FOLLOWUP_TASKS_ROUND6.md`; plot `results/round6_length_robust/round6_comparison.png`; data `metrics/round6_summary.csv`.

Full-test gaps (g1=semantics, g2=syntax, g3=length; `worst`=min length-subset acc):

> **Reporting caveat added in Round 7 (§9.10.1).** The `g3` column below is the **all-set**
> value, which is a poor length diagnostic: the two length subsets carry opposite-signed g3
> (mechanically — `rejected″` tracks *chosen's* length, §9.8 finding 4), so averaging them
> cancels most of the signal. Across all 19 evaluated checkpoints, hard-subset accuracy is
> strongly rank-correlated with **g3 measured within that subset** (spearman +0.72,
> permutation p = 0.0006) but essentially uncorrelated with the all-set g3 here (spearman
> +0.10). Per-subset g1/g2/g3 for every method in this table are in
> `metrics/subset_gaps.csv`; prefer them.

| method | acc | g1 | g2 | g3 | g3/tok | acc_cl | acc_rl | len_gap | worst |
|---|---|---|---|---|---|---|---|---|---|
| **Dr.DPO** | 0.648 | 21.95 | −2.19 | −2.73 | −0.018 | 0.614 | 0.687 | **0.073** | **0.614** |
| **R-DPO** | 0.649 | 15.38 | −2.04 | −3.10 | −0.047 | 0.612 | 0.694 | 0.082 | 0.612 |
| LD-DPO | 0.655 | 15.13 | −2.04 | −2.89 | −0.040 | 0.595 | 0.729 | 0.135 | 0.595 |
| rDPO | 0.653 | 15.80 | −2.02 | −2.32 | −0.019 | 0.587 | 0.735 | 0.148 | 0.587 |
| SamPO | 0.659 | 15.20 | −2.26 | −3.01 | −0.045 | 0.586 | 0.749 | 0.163 | 0.586 |
| Tie | 0.628 | 16.25 | −1.67 | −6.08 | −0.135 | 0.570 | 0.707 | 0.137 | 0.570 |
| ORPO | 0.611 | 30.59 | −1.01 | −23.94 | −0.657 | 0.489 | 0.759 | 0.270 | 0.489 |
| LMPO | 0.579 | 8.32 | 1.58 | −0.83 | 0.039 | 0.332 | 0.869 | 0.537 | 0.332 |
| DPO v3 (§9.8) | 0.656 | 16.32 | −2.04 | −1.73 | −0.001 | 0.607 | 0.717 | 0.110 | 0.607 |
| SimPO v2 (§9.8) | 0.621 | 17.39 | −1.84 | −9.12 | −0.230 | 0.460 | 0.814 | 0.354 | 0.460 |

**Findings.**
1. **Only Dr.DPO and R-DPO beat DPO v3 — marginally.** Dr.DPO: worst 0.614 vs 0.607, length_acc_gap 0.073 vs 0.110, acc 0.648, strong g1 21.95; R-DPO: worst 0.612, gap 0.082. Neither reaches the Round-5 *trained* invariance winner (DPO v3 + I4, worst 0.638; §9.8 Follow-up 2). Dr.DPO is *not* a length method — its DRO down-weighting of high-loss pairs *incidentally* trims the worst subset.
   > **CORRECTED by Round 7 (§9.10.2).** Two claims in this finding do not survive testing.
   > (a) **The stated mechanism is wrong.** Reconstructing Dr.DPO's actual KL-DRO weights on
   > real training pairs shows they are **length-blind**: corr(per-pair loss, |y_w|−|y_l|) =
   > −0.04, and the chosen-longer:rejected-longer weight ratio is 1.017 (within 2% of uniform
   > at every β′). It does not preferentially down-weight length-exploiting pairs; it
   > suppresses high-loss outliers *uniformly* (ESS/B = 0.82). (b) **The improvement is not
   > significant.** Dr.DPO's hard-subset gain over DPO v3 is +0.007 (exact McNemar p = 0.54,
   > n.s.); its only significant effect is getting **worse** on the easy subset (−0.030,
   > p = 0.017). 82% of its length-gap reduction comes from degrading the easy subset rather
   > than lifting the hard one, and `worst_subset` rewards that. Read Dr.DPO's round-6 win as
   > largely a metric artifact.
2. **LD-DPO / rDPO / SamPO ≈ DPO v3** (worst 0.586–0.595); SamPO's token down-sampling did not yield decision-level invariance here.
3. **ORPO is strongly brevity-biased** (g3 −23.94; acc_chosen_longer 0.489 < chance) despite the largest g1 (30.59): length-averaged odds-ratio + monolithic SFT ⇒ **length averaging ≠ length invariance**.
4. **LMPO is the sharpest raw-logp ⊥ decision dissociation (sharpens §9.7/§9.8):** smallest raw |g3|=0.83 yet the *worst* decision length gap (0.537, acc_chosen_longer 0.332) and weak g1 (8.32). Length-neutral log-probs ≠ length-invariant decisions.
5. **Tie Training** did not beat DPO v3 (worst 0.570); the small/low-diversity length-tie pool (≈3000 upsampled) is the likely limiter.
6. **The ~0.66 accuracy ceiling and the worst⊥g3 wall survive all eight published methods** — corroborating that the TL;DR length bias is objective-level and the ceiling is model/data-bound on 1.5B+TL;DR (§9.6/§9.8). Caveats: one hyperparameter setting per method (ORPO/LMPO look under-tuned); small tie pool. *(The worst⊥g3 half of this claim is revised by Round 7 — see §9.10.3.)*

### 9.10 Per-subset decomposition, why Dr.DPO wins, and Dr.DPO tuning (added 2026-08-19)

Round 6 crowned Dr.DPO on `worst_subset` without asking *how* it got there. Round 7 asks three
questions: (1) what do the g1/g2/g3 gaps look like **per length subset** rather than pooled;
(2) what is Dr.DPO's actual mechanism; (3) can Dr.DPO be tuned to lift **both** subsets. Eight
configs on the headline recipe (full `trl-lib/tldr-preference`, LR 5e-6, 2 epochs), full
2000-example length-split eval. Digest: `FOLLOWUP_TASKS_ROUND7.md`; data
`metrics/subset_gaps.csv` (per-subset gaps for all 19 evaluated checkpoints),
`metrics/round7_summary.csv`.

**Headline: a new best method.** `drdpo_i4b` = Dr.DPO + the §9.8 I4 raw-margin invariance
penalty at **λ=2.5e-4** reaches `worst_subset` **0.672** — the standing record was 0.638 —
with the two length subsets at **0.673 / 0.672** (length gap **0.001**).

| method | acc | g1 | g3 | acc cl | g3[cl] | acc rl | gap | **worst** |
|---|---|---|---|---|---|---|---|---|
| **`drdpo_i4b`** (Dr.DPO + I4 λ=2.5e-4) | 0.667 | 14.43 | −1.84 | **0.673** | **−4.59** | 0.672 | **0.001** | **0.672** |
| `drdpo_rdpo` (Dr.DPO + R-DPO penalty) | 0.648 | 23.04 | −2.19 | 0.641 | −6.03 | 0.663 | 0.022 | 0.641 |
| I4 winner (§9.8) | 0.658 | 17.72 | −3.36 | 0.638 | −9.30 | 0.692 | 0.054 | 0.638 |
| `drdpo_i4` (λ=5e-4, overshoots) | 0.654 | 13.75 | −3.23 | 0.691 | −5.64 | 0.622 | 0.068 | 0.622 |
| Dr.DPO (§9.9) | 0.648 | 21.95 | −2.73 | 0.614 | −8.43 | 0.687 | 0.073 | 0.614 |
| `drdpo_accum` (E, global normalizer) | 0.649 | 22.70 | −2.17 | 0.610 | −7.64 | 0.697 | 0.086 | 0.610 |
| DPO v3 | 0.656 | 16.32 | −1.73 | 0.607 | −8.61 | 0.717 | 0.110 | 0.607 |
| `drdpo_strat` (B, stratified DRO) | **0.668** | 20.86 | −1.66 | 0.592 | −8.07 | **0.763** | 0.171 | 0.592 |

**Findings.**

1. **Report g3 per subset, not pooled.** Across all 19 evaluated checkpoints, hard-subset
   accuracy is strongly rank-correlated with **g3[cl]** (spearman **+0.720**, permutation
   p = 0.0006) and essentially uncorrelated with the all-set g3 that §9.9 tabulates (spearman
   +0.10), which cancels because the subsets carry opposite-signed g3. The *linear*
   correlation is only +0.31 (n.s.) — two degenerate high-displacement configs act as
   leverage points; excluding them, pearson = +0.82 and spearman = +0.86 (p < 0.0001). g1 does
   **not** predict hard-subset accuracy (spearman +0.02); ORPO is the reductio (largest g1
   anywhere, below-chance accuracy there).
2. **Dr.DPO is an outlier-suppressor, not a length method — §9.9 finding 1 is corrected
   above.** Its KL-DRO weights are length-blind (corr(loss, length-diff) = −0.04; cl:rl weight
   ratio 1.017 at every β′) because the DPO loss is on the **reference-relative** margin while
   the eval ranks the **raw** margin — so *any* loss-keyed reweighting is structurally blind to
   the bias being measured. What it does do is discard ~18% of effective batch size (ESS/B
   0.82), which buys strong early semantic separation (g1 16→20 in the first quarter) while its
   raw length gap grows ~4× worse than DPO v3's. Its hard-subset gain is n.s. (+0.007, p=0.54)
   and 82% of its gap reduction is ceiling-cutting.
3. **λ is sharply single-peaked and I4-on-Dr.DPO is the win — the §9.8 Pareto wall is
   weakened** (see the revision box in §9.8). λ=0 undershoots (0.614/0.687), λ=5e-4 overshoots
   and *flips* the bias (0.691/0.622), λ=2.5e-4 balances (0.673/0.672). `drdpo_i4b` beats the
   round-5 record significantly on the binding subset (+0.035, exact McNemar p = 0.005) with no
   significant loss on the other (−0.020, n.s.), and with *less* likelihood displacement than
   DPO v3 (chosen_logp −94 vs −100) — so it is not the §5.4 failure mode in disguise. It works
   because I4 penalizes the raw policy margin, precisely the quantity §9.10.2 shows the DRO
   reweighting cannot see.
4. **Two informative negatives.** (E) Widening the DRO normalizer from the 8-example
   micro-batch to a 128-loss FIFO is a clean null (worst 0.610 vs 0.614; all p ≥ 0.14) —
   the normalization population was never the limitation. (B) Length-stratified DRO, the
   a-priori favourite, produced the *worst* length gap of the round (0.171): equalizing the two
   environments' DRO does not equalize their accuracy.
5. **First (provisional) crack in the ~0.66 accuracy ceiling.** `drdpo_strat` reaches acc_all
   **0.668**, the highest in the project and the only significant improvement over DPO v3's
   0.656 (p = 0.044) — though it fails badly on the length metric, and the p-value would not
   survive correction for the ~8 configs compared. Needs a seed replication.
6. **The ceiling argument itself survives.** No config's overall accuracy differs
   significantly from DPO v3 except that marginal case. Round 7's gain is in **balance**, not
   accuracy: `worst` 0.672 now sits within 0.005 of acc_all 0.667, i.e. the gap term of
   `worst ≈ acc_all − gap/2` is spent, and further progress requires a better base model.

**Caveats.** Single seed per config; λ=2.5e-4 was *predicted* from the λ=0 / λ=5e-4 endpoints
rather than searched, so its near-perfect balance is partly luck; the I4-winner comparison is
not compute-matched (that run is a continue-train of DPO v3, which makes `drdpo_i4b`'s win
stronger but not a clean head-to-head). Top follow-ups: replicate on 2–3 seeds, finer λ sweep
around 2.5e-4, and ablate I4-at-λ≈2.5e-4 on plain DPO to separate Dr.DPO's contribution.

### 9.11 Support-Augmented KL-Dr.DPO — inner DRO over the counterfactual orbit (added 2026-08-26)

Rounds 6–7 aggregated robustly **across examples** (Dr.DPO's batch dual) but always over a
single (chosen, rejected) pair per example. Round 8 adds a second, *inner* level: each example's
two extra rejected counterfactuals — `rejected_double_prime` (length matched to chosen) and
`rejected_prime` (length **and** syntax matched) — are treated as a nominal support
p = (1−ε_L−ε_LS, ε_L, ε_LS) = (0.85, 0.10, 0.05) around the original pair, and the per-example
loss becomes the KL-regularized worst case over that orbit

    A_i = τ_v · logsumexp_k( log p_k + ℓ_i^k / τ_v ),   ℓ_i^k = −log σ(m_i^k),   τ_v = 1.0

with `A_i` handed to the **unchanged** outer aggregation. Fixed τ_v ⇒ KL-regularized /
KL-penalized DRO, *not* a constrained DRO with an explicit KL radius; no dual variable is
optimized and no KL ball is solved for. Implementation: `length_pref.py`
(`counterfactual_kl_aggregate`, collator `counterfactual=True`), enabled by
`--use_counterfactual_kl_dro`; **off by default and byte-compatible with rounds 1–7** (the
Dr.DPO dual was extracted into `_outer_aggregate` and reused verbatim, and a unit test
recomputes the pre-change dpo/rdpo/drdpo losses independently). Digest:
`FOLLOWUP_TASKS_ROUND8.md`; data `metrics/round8_summary.csv`; runner `scripts/run_round8.sh`;
tests `test_counterfactual_kl_dro.py` (34 cases).

Six configs, `Bojian92/tldr_preference_decomposed[train]` (n=3000), 2 epochs, LR 5e-6, batch 8×2,
256-example decomposed validation:

| label | outer | τ_v | acc | acc_rpp | acc_rp | **worst_orbit** | len_gap | g1 | g2 | g3 |
|---|---|---|---|---|---|---|---|---|---|---|
| base_dpo | mean | — | 0.656 | 0.578 | 0.801 | 0.484 | 0.573 | 15.55 | −9.04 | +0.24 |
| base_drdpo | Dr.DPO | — | 0.648 | 0.551 | 0.777 | 0.473 | 0.575 | 14.70 | −8.48 | +0.26 |
| cf_dpo_tau1 | mean | 1.0 | 0.656 | 0.656 | 0.859 | **0.547** | 0.573 | 19.30 | −8.07 | −4.39 |
| cf_drdpo_tau025 | Dr.DPO | 0.25 | 0.652 | 0.641 | 0.848 | 0.535 | 0.566 | 19.00 | −7.89 | −4.46 |
| cf_drdpo_tau1 | Dr.DPO | 1.0 | 0.652 | 0.648 | 0.855 | 0.543 | 0.566 | 19.05 | −7.51 | −4.98 |
| cf_drdpo_tau4 | Dr.DPO | 4.0 | 0.652 | 0.652 | 0.855 | 0.543 | 0.566 | 19.21 | −7.40 | −5.04 |

1. **Worst-case-over-orbit accuracy — P(chosen beats all three variants) — rises ~6 points**
   (0.473/0.484 → 0.535–0.547), driven by the length-matched pair (+9 pts). That is precisely
   the quantity the inner DRO optimizes, so this is confirmation, not discovery.
2. **The ~0.66 accuracy ceiling and the ~0.57 length-direction gap are untouched.** Every
   counterfactual sits on the *rejected* side, so nothing in this method addresses the
   chosen-vs-rejected length bias — consistent with §9.10's conclusion that the binding
   constraint is the base model, not the aggregation.
3. **τ_v is nearly inert here, for a measurable reason.** End-of-training branch losses span
   only 0.395–0.531, so the inner adversary stays near nominal: counterfactual mass
   q^L+q^LS = 0.133 / 0.139 / 0.146 for τ_v = 0.25 / 1.0 / 4.0 against a nominal 0.150. The
   ordering confirms the sign convention (sharper adversary ⇒ more mass on the hardest branch,
   which by convergence is the *original* pair, ℓ⁰ = 0.53).
4. **Pooled g3 goes strongly negative (−4.4 … −5.0 vs +0.24) — but see the subset table below,
   which is the interpretable version.** Per subset the change is asymmetric, not a uniform
   divergence: rejected_longer moves *toward* 0 and chosen_longer *away* from it. g1 jumps to
   ~19, SimPO-like, without SimPO's length normalization, and improves in both subsets.

### Per-length-subset breakdown (the pooled gaps above are NOT interpretable)

§9.10 finding 3 applies here in full: the two length subsets carry **opposite-signed g3 that
cancels on averaging**, so the controls' pooled g3 of +0.24 — which reads as near-perfect length
invariance — is an artifact. Ties excluded; subsets match `scripts/subset_gaps.py`.
Data: `metrics/round8_summary_by_subset.csv`.

| arm | subset | acc | acc_rpp | **worst_orbit** | g1 | g2 | g3 |
|---|---|---|---|---|---|---|---|
| base_drdpo | chosen_longer (n=106) | 0.321 | 0.434 | 0.302 | 8.86 | −15.09 | **−10.78** |
| base_drdpo | rejected_longer (n=144) | 0.896 | 0.639 | 0.597 | 19.28 | −3.51 | **+8.18** |
| cf_drdpo_tau1 | chosen_longer | 0.330 | 0.557 | 0.330 | 13.49 | −13.50 | **−16.97** |
| cf_drdpo_tau1 | rejected_longer | 0.896 | 0.722 | 0.701 | 23.36 | −2.91 | **+3.55** |

1. **The worst-orbit gain is almost entirely in the easy subset.** rejected_longer 0.597 → 0.701
   (+10 pts); chosen_longer 0.302 → 0.330, i.e. **3 examples out of 106** — statistically
   nothing. This is the same pattern §9.10 diagnosed for Dr.DPO itself: the gain does not land
   on the binding subset. The pooled "+6 pts" headline hides this.
2. **g3 does not simply "diverge to −5"; it becomes ASYMMETRIC.** rejected_longer improves
   toward 0 (+8.18 → +3.55) while chosen_longer worsens (−10.78 → −16.97). Mean |g3| is roughly
   flat (9.5 → 10.3) but the between-subset spread triples (2.6 → 13.4).
3. **g1 and g2 improve in BOTH subsets** (g1 +4.6/+4.1, g2 +1.6/+0.6 toward 0), so the semantic
   and syntax axes are the only places the method helps symmetrically.

**Not comparable to the round-6/7 Dr.DPO numbers.** Those trained on `trl-lib/tldr-preference`
(92,858 rows × 2 epochs); round 8 trains on the decomposed set (**3,000** rows) because that is
the only set carrying the counterfactual columns, and the controls must see identical data. The
round-8 controls are therefore heavily undertrained — `base_drdpo` g2 = −8.5 sits near the init
value −9.45, versus round-6 Dr.DPO's −2.2. **The method is data-capped at n=3000 until
counterfactuals are generated for the full 92k set**; that, not τ_v tuning, is the first-order
limitation of this round.

**Caveats.** Single seed; 256 eval examples (±0.03 at 1σ), so only the control-vs-cf gap is
resolvable — the τ_v ordering and cf_dpo-vs-cf_drdpo are not. The comparison is **step-matched,
not information-matched**: cf arms see 3 rejected responses per prompt, so part of the gain is
"trained on rejected″/rejected′ at all". The information-matched control (counterfactuals as
independent batch rows — the flattening the *method* deliberately avoids, but the right
*baseline*) is the first follow-up.

### 9.12 Post-audit: rebuilt benchmark, corrected re-ranking, and two negative results (added 2026-08-31)

`AUDIT_2026-08-29.md` found that the g1/g2/g3 measurement used by §§9.1–9.11 is confounded three
ways: the reported statistic (`mean(g)`) cancels large per-example effects of opposite sign;
summed log-probs carry a mechanical ~2.5-nat-per-token length term so g2/g3 cannot reach 0; and
with no reference scale a gap of "21 nats" is uninterpretable. This section reports what the
corrected measurement says. **Full detail: `POST_AUDIT_FINDINGS.md`; round digest:
`FOLLOWUP_TASKS_ROUND9.md`; data: `metrics/audit_all_methods_v2.csv`.**

**Rebuilt benchmark.** 420 LLM writer/auditor agents produced a new eval set (**1979 rows, 1979
unique prompts** vs the legacy 2000 rows / **195** prompts), with every counterfactual at exactly
`chosen`'s token length (legacy: 30.3%), syntax-pair similarity 0.709 (legacy 0.545), no
degenerate rows (legacy 6.2%), plus a **second twin of each counterfactual written to identical
instructions**. The twins give a **noise floor** — the score gap between two interchangeable
rewrites — against which every other gap is normalized. A disjoint 1998-example training set was
built the same way. Generation `scripts/wf_decomposed_v2.js`; QC `scripts/assemble_decomposed_v2.py`;
metric `scripts/decomposition_metrics.py`.

**Re-ranking (19 checkpoints re-scored, not retrained).** Ratio = meaning-sensitivity ÷
syntax-sensitivity, both in noise-floor units; CIs bootstrapped over prompts.

| method | accuracy | meaning | syntax | length | ratio | vs untrained (95% CI) |
|---|---|---|---|---|---|---|
| `drdpo_strat` | 0.589 | 3.506 | 1.479 | 1.554 | **2.371** | +0.242 [+0.16, +0.32] |
| DPO v3 | 0.580 | 3.535 | 1.508 | 1.551 | 2.345 | +0.216 [+0.15, +0.28] |
| Dr.DPO | 0.592 | 3.457 | 1.498 | 1.538 | 2.308 | +0.179 [+0.09, +0.27] |
| `drdpo_i4b` (§9.10 record) | 0.589 | 3.484 | 1.618 | 2.330 | 2.153 | +0.024 [−0.05, +0.10] **n.s.** |
| BASE (untrained) | 0.518 | 3.238 | 1.521 | 1.490 | 2.129 | — |
| `drdpo_i4` | 0.581 | 3.434 | 1.630 | 2.618 | 2.106 | −0.023 |

1. **§9.10's record holder does not survive.** `drdpo_i4b` is statistically indistinguishable
   from the untrained model. Its mechanism is now clear: it has the *lowest* systematic length
   bias but the *second-highest* per-example length sensitivity (2.330 vs 1.490 untrained) —
   the I4 penalty optimizes the mean margin across length groups, i.e. exactly the statistic
   that cancels. `drdpo_i4` (stronger λ) lands *below* the untrained model.
2. **Training raises meaning-sensitivity but does not lower surface-sensitivity.** Meaning
   3.238 → 3.506 (+8%); syntax 1.521 → 1.479 (−3%, noise); ratio +11%. **No method drives g2
   toward 0** — it sits at ~1.5× the noise floor for all 19 methods and the untrained model.
   The un-normalized semantic gap doubles (20.5 → 39.7 nats), but the noise floor grows nearly
   as much (8.69 → 14.71): most of preference optimization is amplification, a small real part
   is discrimination. Against the project's target (g1 large, g2≈0, g3≈0): **g1 yes, g2 no, g3 no.**
3. **Survivors cluster in 2.29–2.37** across very different objectives — a base-model ceiling
   rather than a method difference.

**Round 9 — §9.11's method fails its own control.** 6 arms × 3 seeds on the new training set;
every arm sees identical text, only the aggregation differs.

| arm | steps | accuracy | ratio |
|---|---|---|---|
| base_drdpo (no counterfactuals) | 250 | 0.5191 ±0.0008 | 2.187 ±0.008 |
| cf_drdpo (§9.11 method, p=.85/.10/.05) | 250 | 0.5205 ±0.0013 | 2.216 ±0.005 |
| cf_uniform (p=⅓ each) | 250 | 0.5309 ±0.0012 | 2.300 ±0.002 |
| **flat_sm (plain pairs, step-matched)** | 250 | **0.5452 ±0.0005** | **2.353 ±0.013** |
| flat_dpo (plain pairs, 3× steps) | 750 | 0.5449 ±0.0015 | 2.361 ±0.022 |

4. **The worst-case aggregation contributes nothing.** Using the same counterfactuals as ordinary
   preference pairs wins by +0.137 ratio and +0.025 accuracy at matched text and matched steps
   (6–25 seed-sd). §9.11's gain was the data. Mechanism: p=(0.85,0.10,0.05) with τ_v=1 gives the
   counterfactuals only ~15% of the gradient (measured cf mass 0.14) versus 67% for the flat
   control, and benefit tracks that weight monotonically (0% → 2.187, 15% → 2.216, 50% → 2.300,
   67% → 2.353). **Recommendation: drop the inner DRO.**
5. **Counterfactual data looks ~46× more sample-efficient** — ratio 2.353 from 1998 examples vs
   2.371 from 92,858 — but accuracy does not follow (0.545 vs 0.589), and the comparison varies
   data source, size and training length at once. Treat as provisional pending a scaling curve.

**Caveats.** One base model (Qwen2.5-1.5B) throughout; the counterfactuals' *meaning* preservation
has been checked only mechanically (token length, similarity), never by human or LLM judge; the
round-5 invariance checkpoints no longer exist on disk so are absent from the re-ranking;
round-9 arms train on 1998 examples and their absolute accuracies are therefore not comparable
to the 92k-trained checkpoints (the within-round comparison is matched and is what the
conclusions rest on).

