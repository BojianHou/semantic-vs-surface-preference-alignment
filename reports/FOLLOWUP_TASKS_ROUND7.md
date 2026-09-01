# Follow-up Tasks — Round 7 — Per-subset decomposition, why Dr.DPO wins, and Dr.DPO tuning

*Date: 2026-08-19. Three tasks: (1) extend the length-subset breakdown from accuracy to the
full g1/g2/g3 decomposition for every evaluated method; (2) explain mechanistically why
Dr.DPO led the round-6 `worst_subset` metric; (3) tune Dr.DPO to lift BOTH length subsets.
Full detail will land in `REPORT.md` §9.10; this file is the frozen round-7 digest.*

> **Convention.** Frozen per-round digest; `REPORT.md` is the superset. Rounds 1–6:
> `FOLLOWUP_TASKS_ROUND{1..6}.md`.
>
> **Status: complete.** All 8 configs trained (full `trl-lib/tldr-preference`, 2 epochs) and
> evaluated on the full 2000-example decomposed test.

---

## Headline

1. **New best method: `drdpo_i4b` = Dr.DPO + I4 raw-margin invariance at λ=2.5e-4.**
   `worst_subset` **0.672** vs the previous record 0.638 (§9.8), with the two length subsets
   at **0.673 / 0.672** — a length gap of **0.001**. It beats the round-5 record
   significantly on the binding subset (+0.035, p=0.005) with no significant loss on the
   other, and without extra likelihood displacement (chosen_logp −94 vs DPO v3's −100).
2. **Round 6's explanation of Dr.DPO is wrong, and its win was largely a metric artifact.**
   Its DRO weighting is empirically *length-blind*; its hard-subset gain is **not
   significant** (+0.007, p=0.54); 82% of its length-gap reduction comes from *degrading*
   the easy subset. `REPORT.md` §9.9 finding 1 needs correcting.
3. **Report g3 per subset, not on the full set.** Hard-subset accuracy tracks g3 measured
   *within* that subset (spearman +0.72, p=0.0006 over 19 methods) but is unrelated to the
   all-set g3 that §9.9 tabulates (spearman +0.10) — the two subsets carry opposite-signed
   g3 that cancels on averaging.
4. **The ~0.66 accuracy ceiling holds; the *worst-subset* ceiling moved.** Round 5 argued via
   `worst ≈ acc_all − gap/2` that clearing 0.68 needs acc ≈ 0.70, "unreachable on 1.5B +
   TL;DR". That relation survives: `drdpo_i4b` reaches 0.672 not by raising accuracy (0.667,
   n.s. vs DPO v3) but by driving the gap to ~0. The gap term is now spent — remaining
   headroom is bounded by accuracy alone.

---

## 1. Per-length-subset g1/g2/g3 for every method (task 1)

§9.9 reported round-6 gaps on the full test only, with the length split as *accuracy* alone;
§9.8 reported per-subset gaps for DPO v3 / SimPO v2 only. This is a pure recompute from the
saved per-example CSVs — they already carry all four objects' logps and token counts — so no
model forward pass was needed. Script `scripts/subset_gaps.py` → `metrics/subset_gaps.csv`.

Sum-logp scale. cl = chosen longer (n=1031, the hard subset), rl = rejected longer (n=860).
Round-7 configs are marked ‡.

| method | acc | g1 | g2 | g3 | acc cl | g1 cl | g3 cl | acc rl | g1 rl | g3 rl | gap | **worst** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ‡ **`drdpo_i4b`** | 0.667 | 14.43 | −2.67 | −1.84 | **0.673** | 16.02 | −4.59 | 0.672 | 13.34 | +0.89 | **0.001** | **0.672** |
| ‡ `drdpo_rdpo` | 0.648 | 23.04 | −2.53 | −2.19 | 0.641 | 26.97 | −6.03 | 0.663 | 19.53 | +1.78 | 0.022 | 0.641 |
| I4 winner (§9.8) | 0.658 | 17.72 | −2.23 | −3.36 | 0.638 | 20.70 | −9.30 | 0.692 | 15.41 | +3.07 | 0.054 | 0.638 |
| ‡ `drdpo_bp05` | 0.655 | 66.26 | −3.51 | −14.31 | 0.635 | 81.08 | −25.55 | 0.686 | 52.50 | −1.66 | 0.051 | 0.635 |
| ‡ `drdpo_i4` | 0.654 | 13.75 | −2.47 | −3.23 | **0.691** | 15.37 | −5.64 | 0.622 | 12.56 | −1.00 | 0.068 | 0.622 |
| Dr.DPO (§9.9) | 0.648 | 21.95 | −2.19 | −2.73 | 0.614 | 25.38 | −8.43 | 0.687 | 19.06 | +3.36 | 0.073 | 0.614 |
| R-DPO | 0.649 | 15.38 | −2.04 | −3.10 | 0.612 | 17.39 | −8.90 | 0.694 | 14.05 | +3.13 | 0.082 | 0.612 |
| ‡ `drdpo_accum` | 0.649 | 22.70 | −2.36 | −2.17 | 0.610 | 25.97 | −7.64 | 0.697 | 19.90 | +3.70 | 0.086 | 0.610 |
| DPO v3 | 0.656 | 16.32 | −2.04 | −1.73 | 0.607 | 18.95 | −8.61 | 0.717 | 14.30 | +5.84 | 0.110 | 0.607 |
| LD-DPO | 0.655 | 15.13 | −2.04 | −2.89 | 0.595 | 17.03 | −11.59 | 0.729 | 13.91 | +6.82 | 0.135 | 0.595 |
| ‡ `drdpo_bp2` | 0.655 | 16.16 | −2.06 | −1.62 | 0.592 | 18.49 | −8.52 | 0.734 | 14.53 | +5.96 | 0.142 | 0.592 |
| ‡ `drdpo_strat` | **0.668** | 20.86 | −2.38 | −1.66 | 0.592 | 23.75 | −8.07 | **0.763** | 18.44 | +5.49 | 0.171 | 0.592 |
| rDPO | 0.653 | 15.80 | −2.02 | −2.32 | 0.587 | 17.87 | −9.66 | 0.735 | 14.42 | +5.84 | 0.148 | 0.587 |
| SamPO | 0.659 | 15.20 | −2.26 | −3.01 | 0.586 | 17.15 | −11.99 | 0.749 | 13.93 | +7.07 | 0.163 | 0.586 |
| ‡ `drdpo_bp025` | 0.616 | 93.74 | +4.48 | −40.96 | 0.586 | 116.88 | −78.29 | 0.655 | 71.78 | +0.17 | 0.069 | 0.586 |
| Tie | 0.628 | 16.25 | −1.67 | −6.08 | 0.570 | 18.10 | −14.71 | 0.707 | 15.14 | +3.43 | 0.137 | 0.570 |
| ORPO | 0.611 | 30.59 | −1.01 | −23.94 | 0.489 | 32.19 | −33.14 | 0.759 | 29.87 | −14.08 | 0.270 | 0.489 |
| SimPO v2 | 0.621 | 17.39 | −1.84 | −9.12 | 0.460 | 18.95 | −20.10 | 0.814 | 16.37 | +3.34 | 0.354 | 0.460 |
| LMPO | 0.579 | 8.32 | +1.58 | −0.83 | 0.332 | 8.09 | −24.06 | 0.869 | 8.73 | +26.57 | 0.537 | 0.332 |

**Findings.**

1. **The all-set g3 in §9.9's table is near-useless as a length diagnostic; the per-subset g3
   is what ranks methods.** Over all 19 evaluated checkpoints, `acc_chosen_longer` is
   strongly rank-correlated with **g3[cl]** (spearman **+0.720**, permutation p = 0.0006) but
   essentially uncorrelated with the all-set g3 (spearman +0.099). The reason is mechanical:
   `rejected″` tracks *chosen's* length (§9.8 finding 4), so the two subsets carry
   opposite-signed g3 which largely cancels when averaged.
   *Robustness:* the linear (pearson) correlation is only +0.306 (n.s.) because two
   degenerate high-displacement configs (`drdpo_bp025`, `drdpo_bp05`, with g3[cl] of −78 and
   −26) act as extreme leverage points. Excluding those two, pearson = +0.822 and spearman
   = +0.858 (p < 0.0001). The monotone relationship is solid; the linear fit is not.
2. **Semantics does not buy hard-subset accuracy.** g1[cl] is uncorrelated with
   `acc_chosen_longer` (spearman +0.02, n.s.). ORPO is the reductio: the largest g1 anywhere
   (32.19 on cl) with below-chance accuracy there (0.489). Whatever g1 measures, it is not
   what makes a model rank a longer-but-better summary correctly.
3. **g2 (syntax) is small and stable** (−1 to −3.5) for every non-degenerate method, in both
   subsets — the methods differ on length, not syntax, confirming §9.8 finding 2 across a
   much wider method pool.

---

## 2. Why did Dr.DPO win the round-6 metric? (task 2)

§9.9 finding 1 conjectured: *"its DRO down-weighting of high-loss (length-exploiting) pairs
incidentally trims the worst length subset."* Four checks — **the conjecture is wrong, and
the win is largely an artifact of the metric.**
Scripts: `scripts/analyze_drdpo.py`, `scripts/drdpo_weights.py`.

### 2a. The hard-subset gain is not statistically significant

Exact McNemar, paired on the same 2000 examples, vs DPO v3:

| method | subset | acc | Δ | flips (+/−) | p |
|---|---|---|---|---|---|
| Dr.DPO | chosen_longer | 0.607 → 0.614 | +0.007 | +51 / −44 | 0.538 **n.s.** |
| Dr.DPO | rejected_longer | 0.717 → 0.687 | **−0.030** | +42 / −68 | **0.017 \*** |
| Dr.DPO | all | 0.656 → 0.648 | −0.008 | +98 / −113 | 0.335 n.s. |

Dr.DPO's *only* significant effect is getting **worse** on the easy subset.

### 2b. The gap closes from the ceiling, not the floor

Decomposing each method's length-gap reduction vs DPO v3 into "floor lifted" / "ceiling cut":

| method | Δ acc cl | Δ acc rl | gap | share of gap closure from the floor |
|---|---|---|---|---|
| **`drdpo_i4b`** | **+0.066** | −0.045 | 0.001 | **59%** |
| I4 winner | +0.031 | −0.026 | 0.054 | 55% |
| `drdpo_rdpo` | +0.034 | −0.055 | 0.022 | 38% |
| **Dr.DPO** | **+0.007** | −0.030 | 0.073 | **18%** |
| R-DPO | +0.005 | −0.023 | 0.082 | 17% |
| `drdpo_accum` | +0.003 | −0.021 | 0.086 | 12% |

Both `worst_subset` and `length_acc_gap` improve when the *easy* subset is degraded, so they
reward gap compression from either side. Dr.DPO topped round 6 mostly by giving up 3 points
on rejected-longer.

### 2c. The DRO weighting is length-blind — the §9.9 mechanism does not exist

Reconstructed the actual KL-DRO gradient weights `w_i ∝ exp(−L_i/β′)` on 2000 real training
pairs under each converged checkpoint (`scripts/drdpo_weights.py`):

| policy | corr(L_i, \|y_w\|−\|y_l\|) | mean L[cl] | mean L[rl] | ESS/B at β′=1 | w[cl]/w[rl] |
|---|---|---|---|---|---|
| Dr.DPO | **−0.040** | 0.660 | 0.758 | 0.817 | **1.017** |
| DPO v3 | −0.107 | 0.507 | 0.644 | 0.865 | 1.076 |

1. **Loss barely tracks length** (r ≈ −0.04), so the weighting cannot be selecting
   length-exploiting pairs. The cl:rl weight ratio is 1.017 — within 2% of uniform at every
   β′ in {0.25 … 4}.
2. **Chosen-longer pairs have *lower* loss, not higher** (0.660 vs 0.758) — the opposite of
   the conjecture. **The reason is a scale mismatch that matters project-wide:** the DPO loss
   is on the **reference-relative** margin (policy − ref), while the eval ranks the **raw**
   margin (chosen_logp − rejected_logp). A pair can improve over the reference while
   remaining mis-ranked in raw terms, so *any* reweighting keyed on the loss is structurally
   blind to the length bias the benchmark measures.
3. **The down-weighting is real but uniform**: ESS/B = 0.82 at β′=1 — it discards ~18% of
   effective batch size, concentrating on easy pairs regardless of length.

### 2d. Dr.DPO buys semantics early and *loses* raw length-neutrality

In-training trajectory (n=256 slice), at 0/25/50/75/100% of training:

| run | g1 | g3 |
|---|---|---|
| Dr.DPO | 16.13 → 20.20 → 19.80 → 21.87 → **21.92** | −0.16 → −2.73 → −2.98 → −4.77 → **−4.60** |
| DPO v3 | 15.94 → 15.98 → 16.11 → 16.98 → **17.35** | −0.05 → −0.62 → +0.42 → −0.91 → **−1.13** |

The g1 advantage is banked in the first quarter and plateaus, while Dr.DPO's raw-logp length
gap grows ~4× worse than DPO v3's (full-test all-set |g3| 2.73 vs 1.73).

### Answer to task 2

**Dr.DPO is not a length method and does not fix the length bias.** It is a uniform
outlier-suppressor: at β′=1 it discards ~18% of effective batch size, keeping the
easily-separable pairs. That produces much stronger semantic separation (g1 21.95 vs 16.32)
early in training, but it degrades the easy length subset by a significant 3 points while
moving the hard subset by a non-significant +0.007. Because `worst_subset` and
`length_acc_gap` reward *compressing* the two subsets, Dr.DPO topped round 6 largely on a
metric artifact. The genuine levers are those acting on the **raw policy margin** (I4) or
conditioning on length **explicitly** (R-DPO's penalty) — which is what task 3 tested.

---

## 3. Tuning Dr.DPO for both subsets (task 3)

Eight configs, same headline recipe as round 6 (full `trl-lib/tldr-preference`, LR 5e-6,
2 epochs, batch 8 × grad-accum 2) so every number is comparable to
`metrics/round6_summary.csv`. Harness `scripts/run_round7.sh`; mechanisms in
`length_pref.py`. Results are in the §1 table (rows marked ‡).

| mechanism | flag | idea |
|---|---|---|
| β′ sweep | `--drdpo_beta_prime` | round 6 only tried β′=1.0; β′ sets how much batch is discarded |
| (B) stratified DRO | `--drdpo_stratify_length` | apply the dual *within* each length environment, weight both equally |
| (C) + R-DPO penalty | `--drdpo_length_penalty` | subtract α(\|y_w\|−\|y_l\|) from the logit inside the dual |
| (D) + I4 invariance | `--lambda_len_inv` | penalize the mean **raw policy margin** difference across length envs |
| (E) global normalizer | `--drdpo_buffer` | normalize the dual over a 128-loss FIFO, not the micro-batch of 8 |

### 3.1 Significance vs DPO v3 (exact McNemar, paired)

| config | acc cl | p | acc rl | p | acc all | p |
|---|---|---|---|---|---|---|
| **`drdpo_i4b`** | **+0.066** | **0.0000 \*\*\*** | −0.045 | 0.0001 \*\*\* | +0.011 | 0.21 n.s. |
| `drdpo_i4` | +0.083 | 0.0000 \*\*\* | −0.095 | 0.0000 \*\*\* | −0.002 | 0.91 n.s. |
| `drdpo_rdpo` | +0.034 | 0.003 \*\* | −0.055 | 0.0000 \*\*\* | −0.008 | 0.35 n.s. |
| `drdpo_bp05` | +0.028 | 0.068 n.s. | −0.031 | 0.064 n.s. | −0.001 | 0.96 n.s. |
| `drdpo_accum` | +0.003 | 0.87 n.s. | −0.020 | 0.14 n.s. | −0.007 | 0.42 n.s. |
| `drdpo_strat` | −0.016 | 0.060 n.s. | +0.045 | 0.0000 \*\*\* | **+0.012** | **0.044 \*** |

Head-to-heads for the winner:

| comparison | chosen_longer (binding) | rejected_longer | all |
|---|---|---|---|
| **`drdpo_i4b` vs I4 winner (0.638)** | 0.638 → **0.673** (+0.035) **p=0.005 \*\*** | 0.692 → 0.672 (−0.020) p=0.10 n.s. | 0.658 → 0.667 p=0.30 n.s. |
| **`drdpo_i4b` vs `drdpo_rdpo`** | 0.641 → 0.673 (+0.032) **p=0.006 \*\*** | 0.663 → 0.672 (+0.009) n.s. | 0.648 → 0.667 **p=0.021 \*** |
| `drdpo_accum` vs Dr.DPO (mech. E) | 0.614 → 0.610 n.s. | 0.687 → 0.697 n.s. | 0.648 → 0.649 p=1.00 n.s. |

`drdpo_i4b` is the only config that improves the binding subset significantly **without** a
significant loss elsewhere.

### 3.2 Findings

1. **(D) The λ sweep is the whole story, and λ=2.5e-4 is a new record.** The
   length-invariance penalty is sharply single-peaked when stacked on Dr.DPO:

   | λ | acc cl | acc rl | gap | worst |
   |---|---|---|---|---|
   | 0 (plain Dr.DPO) | 0.614 | 0.687 | 0.073 | 0.614 |
   | **2.5e-4** | **0.673** | **0.672** | **0.001** | **0.672** |
   | 5e-4 | 0.691 | 0.622 | 0.068 | 0.622 |

   λ=0 undershoots (rejected-longer easier), λ=5e-4 overshoots and *flips* the bias
   (chosen-longer easier), and the crossing at 2.5e-4 lands both subsets within 0.001.
   `worst` 0.672 beats the round-5 record of 0.638 significantly on the binding subset
   (p=0.005). Note λ=2.5e-4 was *predicted* from the two endpoints, not found by search.
2. **Why I4 works where DRO reweighting cannot.** §2c showed the loss-keyed weighting is
   blind to length because the loss is reference-relative while the eval ranks the raw
   margin. I4 penalizes exactly the raw-margin difference across length environments —
   the quantity the benchmark measures. Dr.DPO then contributes what it is genuinely good at
   (outlier suppression), and the two compose. `drdpo_i4b` reaches this with *less*
   likelihood displacement than DPO v3 (chosen_logp −94 vs −100), so the gain is not the
   §5.4 failure mode in disguise.
3. **(C) Dr.DPO + R-DPO is a solid second** — `worst` 0.641, gap 0.022, with a real floor
   lift over DPO v3 (+0.034, p=0.003). But it only ties the round-5 record (+0.003 vs I4 on
   the binding subset, p=0.85) and is dominated by `drdpo_i4b` on both subsets.
4. **β′ is single-peaked at 0.5, but low β′ is degenerate.** worst: 0.586 (β′=0.25) → 0.635
   (0.5) → 0.614 (1.0) → 0.592 (2.0). However `chosen_logp` collapses to −224 at β′=0.5 and
   **−400** at β′=0.25 (DPO v3: −100), with g1 inflating to 66 and 94 and g2 flipping sign.
   This is the §5.4 likelihood-displacement failure mode taken to an extreme; `bp05`'s 0.635
   should be read as degenerate, not as a usable method.
5. **β′=2 recovers vanilla DPO, confirming the implementation.** g1 16.16 vs DPO v3's 16.32,
   g3 −1.62 vs −1.73, chosen_logp −99 vs −100 — as theory requires (β′→∞ ⇒ batch mean). The
   refactored `_dro_dual` is also bit-identical to round 6's inlined dual on random inputs
   across all β′ and batch sizes.
6. **(E) The global normalizer is a clean null.** Widening the dual's normalizer from the
   8-example micro-batch to a 128-loss FIFO changed nothing measurable (`worst` 0.610 vs
   0.614; all subsets p ≥ 0.14). The per-micro-batch normalization was not the limitation —
   consistent with §2c, where the problem is *what* the weighting keys on, not the population
   it is normalized over.
7. **(B) Length-stratified DRO failed its purpose but produced the project's highest
   accuracy.** It was the top-ranked mechanism a priori (§2c says the weighting must become
   length-aware), yet it produced the *worst* length gap of any round-7 config (0.171) by
   pushing rejected-longer to 0.763. Equalizing the two environments' DRO does not equalize
   their accuracy. Unexpectedly it reaches **acc_all 0.668**, the highest anywhere and the
   only significant improvement over DPO v3's 0.656 (p=0.044) — against the standing
   §9.8/§9.9 position that ~0.66 is model/data-bound. Treat as provisional: p=0.044 would
   not survive correction for the ~8 configs compared here.
8. **The worst ⊥ g3 Pareto wall is substantially weakened.** §9.8 concluded that every
   config improving decision-invariance widened the raw-logp length gap (round-5 winner:
   worst 0.638 at g3 −3.4), and the one config keeping g3 clean fell back to 0.603.
   `drdpo_i4b` breaks that trade:

   | | worst | g3 (all) | g3[cl] |
   |---|---|---|---|
   | DPO v3 | 0.607 | −1.73 | −8.61 |
   | round-5 I4 winner | 0.638 | −3.36 | −9.30 |
   | **`drdpo_i4b`** | **0.672** | **−1.84** | **−4.59** |

   It achieves the best decision-level invariance in the project while holding all-set g3 at
   DPO v3's level, and *halves* g3 on the hard subset. The two forms of invariance are not as
   strictly anti-correlated as the round-5 sweep concluded — though not free either (g3[cl]
   is −4.59, not 0).

---

## Caveats

- **Single seed.** Every result is one training run per config. `drdpo_i4b`'s margin is large
  and significant on the binding subset (p=0.005), but a seed replication is the obvious
  confirmation before treating 0.672 as the new bar.
- **λ=2.5e-4 was predicted, not searched**, from the λ=0 and λ=5e-4 endpoints. The
  near-perfect balance (gap 0.001) is partly luck; a finer sweep would show how sharp the
  optimum is and whether the balance point is stable across seeds.
- **The I4-winner comparison is not compute-matched.** The round-5 winner is a
  *continue-train* of DPO v3, so it has seen strictly more optimization than the round-7
  configs, which train from base for 2 epochs. This makes `drdpo_i4b`'s win *stronger*, not
  weaker, but it is not a clean head-to-head.
- One hyperparameter setting per mechanism, except β′ (4 values) and λ (3 values).
  `drdpo_rdpo` used α=0.05 un-swept, inherited from round 6.
- The DRO weight reconstruction (§2c) evaluates losses under each **converged** checkpoint,
  not along the trajectory; it characterises the weighting regime at convergence.
- `drdpo_strat`'s accuracy result (§3.2.7) is marginal and uncorrected for multiple
  comparisons.
- No config's *overall* accuracy differs significantly from DPO v3 except `drdpo_strat`'s
  marginal +0.012, so the round-5/6 conclusion that acc_all is model/data-bound at ~0.66
  survives. Round 7's gain is in **balance**, not accuracy.

## Actions for `REPORT.md`

1. **Correct §9.9 finding 1** — its stated mechanism is refuted by §2c, and its implied
   hard-subset improvement is not significant (§2a).
2. **Adopt the per-subset g3 convention** (§1 finding 1); the all-set g3 in §9.9's table is
   near-uninformative about length robustness.
3. **Revise the §9.8 worst ⊥ g3 conclusion** — `drdpo_i4b` reaches worst 0.672 at g3 −1.84,
   which the round-5 Pareto framing said should not be possible (§3.2 finding 8).
4. **Update the standing record**: `worst_subset` 0.638 → **0.672** (`drdpo_i4b`).
5. Add §9.10 summarising this round.

## Suggested next steps

1. **Replicate `drdpo_i4b` on 2–3 seeds** — the highest-value follow-up by far.
2. **Finer λ sweep** around 2.5e-4 (1.5e-4, 2e-4, 3e-4, 3.5e-4) to map the balance point and
   check it is not knife-edged.
3. **Ablate: I4 at λ≈2.5e-4 on plain DPO from base**, to separate how much of the win comes
   from Dr.DPO's outlier suppression vs the I4 term alone.
4. `worst` is now within 0.005 of `acc_all` (0.672 vs 0.667) — the gap term is spent.
   Further progress requires higher accuracy, which §9.8/§9.9 argue is model/data-bound, so
   the next real lever is a larger base model, not another objective.

## Artifacts

- Task 1: `scripts/subset_gaps.py` → `metrics/subset_gaps.csv`
- Task 2: `scripts/analyze_drdpo.py`; `scripts/drdpo_weights.py` →
  `metrics/drdpo_weights_{drdpo,DPO_v3}.csv`
- Task 3: `scripts/run_round7.sh`, `scripts/round7_queue.sh`, `scripts/eval_round7.sh`,
  `scripts/round7_watch.sh`; mechanisms in `length_pref.py`
  (`--drdpo_stratify_length`, `--drdpo_length_penalty`, `--drdpo_buffer`, `--lambda_len_inv`)
- Data: `metrics/round7_summary.csv`, `metrics/round7_<label>_fulltest.csv`;
  trajectories `results/round7_drdpo_tuning/<label>/`;
  checkpoints `checkpoints/round7/<label>/final`
