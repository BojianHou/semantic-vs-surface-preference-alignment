# ESCALATION — the loop is converged; the DoD is infeasible on this data (human call needed)

**Status (iter 9):** 15/20 gates PASS. The 5 failing gates (C1, C2, C5, C6, C9) are
**jointly infeasible for any SIPO-family loss** on this dataset under the
contract-frozen **summed-logp** scorer — established across 45+ configs, 3 independent
loss-gradient paths, a coupling analysis, a gap-floor measurement, and a summed-vs-mean
counterfactual (independently reproduced by the evaluator in iter6/7).

**iter-9 update:** the iter-8 recommendation (Option A: length-normalize the scorer →
all_pass, zero retraining) was **checked and found insufficient.** A full mean-scale
scorecard (`option_a_scorecard.py`) shows Option A does **not** clear C6 (composite 0.07
init-anchored / 0.63 mixed, both < 0.8) and its C2/C5/C9 passes are **non-informative** —
the untrained `init` base model posts the same mean numbers as the trained winner. The DoD
`score ≥ 0.8` is therefore unreachable on **both** scales. See §2b.

The deliverable **iter3m** (summed composite **0.4498**) remains the best
all-feasible-gates-passing run and stands. What follows is the decision the loop cannot make
for itself — now sharpened to Option B (accept the achievable frontier).

---

## 1. The binding constraint, in one paragraph

`g2 = mean_ex( Σ_t logp(rp_t) − Σ_t logp(rpp_t) )`. The scorer sums over tokens.
`rp` (rejected_prime) and `rpp` (rejected_double_prime) are *word*-length-identical
paraphrases, but **token**-length-identical in only **24.6%** of examples (median
rp=37, rpp=36 tokens). So `g2` is dominated by a **~1-token summed-length tail**, not by
syntax. Closing summed `g2` to ~0 requires either (a) inflating the longer sequence's
per-token logp above the shorter's — which degrades the shared representation and
collapses `g1` (semantics) — or (b) length-normalizing the score. Path (a) is what all
three loss mechanisms hit; path (b) is a scorer change (contract non-goal), hence this
escalation. The identical mechanism drives `length_acc_gap` (C5): summed logp favors
short responses, so `acc_chosen_longer` (0.34) ≪ `acc_rejected_longer` (0.80).

## 2. Evidence stack (all reproduced this iteration)

| Probe | Result | Kills the hope that… |
|---|---|---|
| perpos ±detach (iter5) | g2 frozen (detach) / g1-collapse (no detach) | per-position invariance moves summed g2 |
| reward_diff-mean (iter3/4) | g2 frozen ~−9 | scalar reward-diff moves it |
| reward_diff-**sum** hard, detached, λ 30–100× (iter7) | g2 froze (7a: 0.2 nats/220 steps) or coupling **1.44** (7b: dg1 −1.34 / dg2 +0.93) | the *most direct* attack on the exact metric moves it |
| gap floor across 15 runs | 0.459 exact (acc_cl 0.340 / acc_rl 0.799), training-invariant | C5 is reachable by training |
| **summed-vs-mean counterfactual** (same iter3m ckpt) | see table below | the failures are policy failures |

**Counterfactual — same winning checkpoint, two scorings (reconfirmed iter8):**

| metric | SUMMED (frozen scorer) | MEAN (length-normalized) | gate under MEAN |
|---|---|---|---|
| \|g2\| syntax | 9.02 | **0.19** | C2 ≤2 **PASS**, C9 <6.8 **PASS** |
| length_acc_gap | 0.459 | **0.038** | C5 ≤0.2 **PASS** |
| acc | 0.602 | 0.586 | C4 ≥0.60 fail (marginal) |
| g1 semantics | 16.69 | 0.462 (mean units) | C1's 18 is a summed-scale number |

**Conclusion:** the model already ranks by length-neutral semantics
(mean acc_cl 0.566 ≈ acc_rl 0.604). C2/C5/C9 are **summed-logp scoring artifacts**.

## 2b. iter-9 correction — Option A does NOT reach all-pass (and why)

Iter 9 built a full mean-scale scorecard (`option_a_scorecard.py`, runnable, scorer
untouched) that iter 8 never produced: it re-derives the **composite** on the mean scale
and compares the winner against the **untrained init** base model. Two results overturn
the iter-8 "Option A = clean unblock" claim:

1. **C6 fails under Option A.** Honestly anchored to init, iter3m's mean-scale composite is
   **0.0748** (< 0.8) — *lower* than the summed 0.4498. It fails under any anchoring:
   init-relative anchors give syn/length/lensplit ≈ 0; the "mixed" anchoring (mean values
   into the summed anchors) instead collapses `sem` to 0 (mean g1 = 0.46 ≪ the 14 anchor)
   and still only reaches 0.63. There is no anchor set on disk that clears 0.8.
2. **The C2/C5/C9 "passes" are non-informative.** The UNTRAINED init model posts the SAME
   mean numbers as the trained winner:

   | MEAN scale | init (untrained) | iter3m (winner) | Δ |
   |---|---|---|---|
   | \|g2\| | 0.206 | 0.194 | +0.012 |
   | acc | 0.590 | 0.586 | −0.004 |
   | acc_cl / acc_rl | 0.566 / 0.604 | 0.566 / 0.604 | 0 / 0 |
   | length_acc_gap | 0.038 | 0.038 | 0.000 |
   | g1 | 0.447 | 0.462 | +0.015 |

   The `≤2` / `≤0.2` thresholds are **summed-scale numbers**; mean-logp gaps are inherently
   O(0.2), so init passes them too. Length-normalization does not reveal a *training* win —
   it reveals the base model was **already** mean-length-neutral, and SIPO barely moves any
   axis off init on that scale.

**Net:** length-normalization is real and correct as a *diagnosis* (it localizes C2/C5 to
the summed scale), but it is **not a path to all_pass** — it neither clears C6 nor
demonstrates a trained improvement. The DoD's `score ≥ 0.8` is unreachable on *both* scales.

## 3. The options (revised iter-9)

### Option A — length-normalize the scorer  (diagnosis only; DOES NOT reach all-pass)
Flips the *threshold* gates C2/C5/C9, but C6 still fails (composite 0.07–0.63) and the
passes are non-informative (init passes identically). Keep it only as evidence that the
summed scale is what drives C2/C5 — not as the unblock. Superseded by Option B for all_pass.

### Option B — accept the achievable frontier as the deliverable  (RECOMMENDED)
No scoring method lets SIPO clear `score ≥ 0.8` on this paraphrase-pair data (and *no
reference method does either*: SimPO reaches g1=19 only at |g2|=6.8, |g3|=5.1 — it fails
C2/C3/C5). The honest close is to relax the DoD to the documented summed-scale frontier and
record iter3m as the result: C1 g1 ≥ 16, C2 \|g2\| ≤ 9, C5 gap ≤ 0.46, C6 score ≥ 0.44,
C9 \|g2\| ≤ 9.1 → iter3m passes all 20. This states the true finding: **SIPO on
paraphrase-pair data plateaus at composite ≈ 0.45; the summed-logp semantic separation is
real (g1 16.7, and iter3m preserves it while baseline v0 collapses to 9.96) but the g2/gap
length-tail is irreducible.**

### Option C — change the setup, then re-run  (highest effort; only if 0.8 is mandatory)
The `score ≥ 0.8` target is only reachable by changing the measurement AND the data/metric,
then retraining: (a) redefine the syntax metric (per-position / symmetric next-token KL,
already `syn_mode="kl"` in `sipo.py`) with mean-scale-calibrated C2/C9, AND (b) re-run
SimPO/DPO/baseline WITH decomposed eval so the mean scale has real reference anchors (none
exist on disk today — only `results/sipo_search/v0` carries mean-logp CSVs). Requires a fresh
sweep and is out of the current in-scope search.

## 4. What the loop should NOT do
Run another training config. All three loss-gradient paths to g2 are closed; a 4th run
re-hits a proven wall (slop). The evaluator has set `recommend_restart=FALSE` for iters 4–8
for the same reason: the rubric's restart trigger is a stuck loss *form* whose remedy is
rebuilding the invariance mechanism — that rebuild was done (iter2) and re-probed (iter5,
iter7); the residual walls live in the frozen scorer + data, which no loss rebuild moves.

**Recommendation (revised): Option B.** iter-9 proved Option A does not reach all_pass
(C6 fails 0.07–0.63 on the mean scale; its threshold passes are non-informative because the
untrained init model matches the winner). The sound close is to accept the achievable
frontier — no SIPO or reference method clears `score ≥ 0.8` on this data — and record iter3m
(summed composite 0.4498) as the deliverable. Verify with:
`.venv/bin/python option_a_scorecard.py` and
`.venv/bin/python sipo_eval.py --run_dir results/sipo_search/iter3m/sipo`.
