# Follow-up Tasks — Round 8 — Support-Augmented KL-Dr.DPO (counterfactual orbit DRO)

*Date: 2026-08-26. One task: add a **KL-regularized DRO over each example's rejected
counterfactual orbit** (original / length-matched / length+syntax-matched) INSIDE the existing
Dr.DPO, leaving the outer batch aggregation untouched, and measure it. Full detail will land in
`REPORT.md` §9.11; this file is the frozen round-8 digest.*

> **Convention.** Frozen per-round digest; `REPORT.md` is the superset. Rounds 1–7:
> `FOLLOWUP_TASKS_ROUND{1..7}.md`.
>
> **Status: complete.** 6 configs trained (`Bojian92/tldr_preference_decomposed[train]`,
> n=3000, 2 epochs, LR 5e-6, batch 8×2) and evaluated on 256 decomposed validation examples.
> Unit tests: 34/34 pass (`test_counterfactual_kl_dro.py`).

---

## Method

For each example the three pairs (chosen ≻ rejected), (chosen ≻ rejected″) and
(chosen ≻ rejected′) share one label. With the file's existing loss-minimization convention
ℓ^k = −log σ(m^k), m^k = β·(s(x,y_w) − s(x,y_l^k)), s = log π_θ − log π_ref:

    p    = (1 − ε_L − ε_LS, ε_L, ε_LS) = (0.85, 0.10, 0.05)
    A_i  = τ_v · logsumexp_k( log p_k + ℓ_i^k / τ_v )          ← inner, NEW
    L    = existing_outer_aggregation({A_i})                   ← unchanged

`A_i` simply replaces `ℓ_i` in the outer aggregation. The three counterfactuals are never
flattened into the batch (that would triple each prompt's weight and mix within- with
between-example robustness). Fixed τ_v ⇒ this is **KL-regularized / KL-penalized** DRO, not a
constrained DRO with an explicit KL radius; no dual variable is optimized.

**Sign conventions differ between the two levels, on purpose.** Inner: `+ℓ/τ_v` ⇒ the *hardest*
transformation of an example dominates (τ_v→0 gives max_k ℓ^k). Outer: the published Dr.DPO
dual `−β' log E[exp(−L/β')]` ⇒ high-loss *examples* are DOWN-weighted. The outer sign was not
touched. Empirical check at the last step of `cf_drdpo_tau1`: mean A = 0.518 while the reported
loss = 0.459 < 0.518, i.e. the outer dual still sits below the mean (Jensen), as it always did.

---

## Headline

1. **The method does what it was designed to do — but only on the easy subset. Pooled
   worst-case-over-orbit accuracy improves ~+6 points; per subset that is +10 pts on
   rejected_longer and +3 examples (n=106) on the binding chosen_longer subset.** P(chosen beats **all three** rejected variants) goes 0.473/0.484 (Dr.DPO /
   DPO controls) → **0.535–0.547** (cf arms). The single biggest component is the
   length-matched pair: P(chosen > rejected″) 0.551/0.578 → **0.641–0.656** (+9 pts).
2. **Plain accuracy is untouched (~0.65) and so is the length-direction gap (~0.57).**
   The ~0.66 accuracy ceiling of §9.9/§9.10 survives round 8 as well, and the chosen-vs-rejected
   length bias — which nothing in this method addresses, since all three counterfactuals sit on
   the *rejected* side — is unchanged to 3 decimals. Consistent with §9.10's finding that the
   binding constraint is the base model, not the aggregation.
3. **τ_v barely matters here, and the reason is measurable.** The three per-branch losses end
   training within 0.14 of each other (0.531 / 0.395 / 0.448), so the inner adversary never
   moves far off nominal: counterfactual mass q^L+q^LS = 0.133 (τ_v=0.25) → 0.139 (1.0) →
   0.146 (4.0), against the nominal 0.150. The ordering is exactly right — a sharper adversary
   puts *more* mass on the hardest branch, which by end of training is the ORIGINAL pair
   (ℓ^0 = 0.53 is the largest), hence *less* counterfactual mass. τ_v=0.25 is very slightly
   worse on worst-orbit (0.535 vs 0.543) but the spread is within noise for a single seed.
4. **Pooled g3 flips strongly negative (−4.4 … −5.0 vs +0.24 for the controls); per subset the
   move is asymmetric rather than a uniform divergence — see the subset table.** Training explicitly
   pushes rejected″ down, so `rejected″ − rejected` becomes very negative. This is the intended
   mechanical consequence of putting rejected″ on the rejected side of a preference pair, but it
   moves g3 *away* from the project's "g3→0 invariance" target — the two goals are in direct
   conflict, and this round chose worst-case robustness. g1 also jumps to ~19.0–19.3 (from
   14.7–15.6), i.e. the SimPO-like semantic separation, without SimPO's length normalization.

## Results (last eval step, 256 decomposed validation examples)

| label | outer | τ_v | acc | acc_rpp | acc_rp | **worst_orbit** | len_gap | g1 | g2 | g3 |
|---|---|---|---|---|---|---|---|---|---|---|
| base_dpo | mean | — | 0.656 | 0.578 | 0.801 | 0.484 | 0.573 | 15.55 | −9.04 | +0.24 |
| base_drdpo | Dr.DPO | — | 0.648 | 0.551 | 0.777 | 0.473 | 0.575 | 14.70 | −8.48 | +0.26 |
| cf_dpo_tau1 | mean | 1.0 | 0.656 | 0.656 | 0.859 | **0.547** | 0.573 | 19.30 | −8.07 | −4.39 |
| cf_drdpo_tau025 | Dr.DPO | 0.25 | 0.652 | 0.641 | 0.848 | 0.535 | 0.566 | 19.00 | −7.89 | −4.46 |
| cf_drdpo_tau1 | Dr.DPO | 1.0 | 0.652 | 0.648 | 0.855 | 0.543 | 0.566 | 19.05 | −7.51 | −4.98 |
| cf_drdpo_tau4 | Dr.DPO | 4.0 | 0.652 | 0.652 | 0.855 | 0.543 | 0.566 | 19.21 | −7.40 | −5.04 |

`metrics/round8_summary.csv`; regenerate with `.venv/bin/python scripts/summarize_round8.py`.

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


**Caveats.** Single seed per config; 256 eval examples (±0.03 at 1σ on a 0.5 rate), so the τ_v
ordering and the cf_dpo-vs-cf_drdpo difference are NOT resolvable — only the ~+6 pt
control-vs-cf gap is comfortably outside that band. The cf arms see three rejected responses per
prompt and therefore ~2× the gradient signal per step of the controls at equal step count: the
comparison is step-matched, not information-matched, so part of the worst-orbit gain is simply
"trained on rejected″/rejected′ at all". A data-matched control (plain DPO on the 3-way expanded
set, i.e. counterfactuals as independent batch rows) is the obvious next arm and is exactly the
flattening the design forbids for the *method* — but it is the right *baseline*. `base_dpo` runs
through TRL's `DPOTrainer` while every other arm runs through `LengthPrefTrainer`; `base_drdpo`
is the trainer-matched control.

## Follow-ups

1. Add the information-matched control (3× expanded pairs, flat batch) to separate "inner DRO"
   from "more rejected data".
2. Seed replication (3 seeds) — nothing here except the control-vs-cf gap survives one seed.
3. τ_v only bites when the branch losses spread; try ε_L/ε_LS ≫ defaults, or a harder orbit
   (counterfactuals generated adversarially rather than fixed).
4. The chosen-side is untouched by construction (§8 of the spec). Pairing this with the round-7
   winner I4 (λ=2.5e-4), which *does* move the length gap, is the natural combination.
