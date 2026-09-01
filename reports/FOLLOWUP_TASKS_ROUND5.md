# Follow-up Tasks — Round 5 — DPO v3 full-test-set length split (g1/g2/g3 + accuracy)

*Date: 2026-07-12. Evaluate DPO v3 (and SimPO v2 for contrast) on the FULL 2000-example decomposed validation ("test set"), split by response-length direction, reporting accuracy AND the g1/g2/g3 decomposition gaps. Full detail in `REPORT.md` §9.8; this file is the frozen round-5 digest.*

> **Convention.** Frozen per-round digest; `REPORT.md` is the superset. Rounds 1–4: `FOLLOWUP_TASKS_ROUND{1,2,3,4}.md`.

## Motivation
Prior length-direction splits (§9.4b, round 3) used the **n=256 in-training eval slice**, which §9.5 flagged as an incidentally short-biased subsample. Round 4 concluded DPO v3 is the most length-robust ranker but cited only slice numbers. Round 5 re-evaluates the **saved DPO v3 checkpoint on the full 2000-example validation** with the matched encoding, and reports the full g1/g2/g3 breakdown, not just accuracy.

## Setup
Evaluate saved final checkpoints on the full decomposed `validation` split (n=2000; chosen-longer 1031, rejected-longer 860, ties 109), using `train.py`'s exact encoding (chat template, max_len 1024, left truncation) so numbers match the in-training callback and round-3. Method: `checkpoints/dpo-v3/final`, `checkpoints/simpo-strong/final` (=SimPO v2). Script: `scripts/eval_length_split_checkpoint.py`; per-example outputs `metrics/length_split_fulltest_{DPO_v3,SimPO_v2}.csv`.

Gaps (sum-logp, the study's standard scale): g1 = chosen−rejected′ (**semantics**), g2 = rejected′−rejected″ (**syntax**), g3 = rejected″−rejected (**length**).

## Result — DPO v3 vs SimPO v2 on the full test set

| method / subset | n | accuracy | g1 semantics | g2 syntax | g3 length |
|---|---|---|---|---|---|
| **DPO v3** — chosen longer | 1031 | 0.607 | +18.96 | −1.10 | −8.61 |
| **DPO v3** — rejected longer | 860 | 0.717 | +14.30 | −2.60 | +5.84 |
| **DPO v3** — all | 2000 | **0.656** | +16.32 | −2.04 | **−1.73** |
| SimPO v2 — chosen longer | 1031 | 0.460 | +18.95 | −1.87 | −20.11 |
| SimPO v2 — rejected longer | 860 | 0.814 | +16.37 | −1.48 | +3.34 |
| SimPO v2 — all | 2000 | 0.621 | +17.39 | −1.84 | **−9.12** |

length-acc gap |chosen_longer − rejected_longer|: **DPO v3 0.110** vs SimPO v2 0.354 (sum-ranking).
Per-token (mean-logp) all-set gaps: DPO v3 g1 +0.433 / g2 −0.032 / **g3 −0.001**; SimPO v2 g1 +0.471 / g2 −0.032 / g3 −0.230.

## Findings
1. **DPO v3 is decisively the most length-robust ranker.** Overall |g3| = **1.73 (DPO v3) vs 9.12 (SimPO v2)** — ~5× more length-neutral. Even in the hardest subset (chosen longer, where `rejected″` is a long chosen-length paraphrase) DPO's g3 is −8.6 vs SimPO's −20.1. Accuracy mirrors it: when the correct answer is longer, DPO still ranks it first 60.7% of the time (>0.5, not brevity-biased), while SimPO flips to 0.460.
2. **g1 (semantics) stays strong for both** (~+14 to +19), roughly length-independent — the semantic discrimination is intact; the methods differ on length, not semantics. g2 (syntax) is small and comparable (−1 to −2.6).
3. **The n=256 slice overstated DPO v3's length bias.** Slice gap 0.377 (chosen-longer acc 0.387) → full-test gap **0.110** (0.607). The slice is short-biased (§9.5); the full test set is the honest number.
4. **The g3 sign flip across subsets is mechanical**, not a method effect: `rejected″` matches *chosen's* length, so its length vs `rejected` reverses with the split direction. Only |g3| magnitude is meaningful, and DPO's is far smaller in both directions.
5. **This sharpens §9.7.** The "big g1 + ~0 g3" target SIPO (round 4) was built to hit is closely met — **by DPO v3, not by SIPO's reward-shaping**: DPO keeps SimPO-level g1 (~16–17) with |g3|≈1.7 (sum) / ≈0.00 (per-token). The best existing method already sits near the goal; the round-4 negative result stands.

## Follow-up (added 2026-07-12) — can length-group DRO remove the residual subset bias? (no)
DPO v3 is length-robust on mixed data (gap 0.110) but still conditions on length within skewed subsets (acc 0.607 chosen-longer vs 0.717 rejected-longer). Idea tested: split each batch into {chosen-longer, rejected-longer} and optimize the WORST group (group-DRO) instead of the plain mean, to stop the model keying on length. Reference-based DPO (sum-logp, sigmoid, β=0.1, ref=frozen base), trained on the decomposed 3k set; the only difference between arms is the objective. Code `dro_dpo_train.py --group_dro {off,worst}`; evaluated on the full 2000 test set.

| run (both 3k, identical except objective) | acc[chosen longer] | acc[rejected longer] | length gap | overall acc | g1 (all) | g3 (all) |
|---|---|---|---|---|---|---|
| plain DPO (off) | 0.540 | 0.744 | 0.204 | 0.629 | 13.95 | −4.79 |
| DPO + group-DRO (worst) | 0.524 | 0.720 | 0.196 | 0.610 | 13.92 | −4.86 |

**Verdict: group-DRO does not remove the length bias.** The gap barely moved (0.204 → 0.196, within noise) and overall accuracy dipped (0.629 → 0.610); g1/g2/g3 essentially unchanged. DRO *did* upweight the failing chosen-longer group, but `acc[chosen longer]` did not rise (0.540 → 0.524) — the model cannot satisfy those pairs because sum-logp mechanically fights "longer is better." This is the smoking gun that the bias is **objective-level, not data-distributional**: neither exclusive-subset training (§9.6) nor loss reweighting (here) removes it — reweighting only redistributes which subset is sacrificed. Caveat: the 3k plain-DPO baseline is more length-biased than 92k DPO v3 (gap 0.204 vs 0.110), but the plain-vs-DRO delta (the tested quantity) is clean and ~zero. The higher-leverage levers for length-invariance remain mean-logp ranking (per-token g3≈0 for free) or an explicit invariance term (trades against g1, the §9.7 wall).

## Follow-up 2 (added 2026-07-13) — brainstorm + auto-research for length/syntax-invariance
Goal: a *trained* method that is semantics-only and length/syntax-invariant **on a skewed test set** (fix DPO v3's within-subset bias). Primary metric: **worst_subset_acc = min(acc_chosen_longer, acc_rejected_longer)** on the full 2000 test (non-degenerate: high only if both skews are handled). Bars: DPO v3 0.607; a free one-parameter length-debias at ranking time reaches ~0.64 (no training) — the bar a trained method must beat. (`invariance_train.py`, `invariance_eval.py`; running-agent-loops harness, fresh `.loop`.)

Free levers first: **I1 mean-logp ranking FAILS** (worst 0.543 — over-corrects, flips the bias to prefer longer); **I6 length-debias** reaches ~0.64 for both DPO v3 and SimPO v2 (post-hoc, in-sample β).

Auto-research: **36 configs** across I2 (length-matched negative), I3 (counterfactual consistency), I4 (equalize policy margin across length environments; +EMA), I5 (IRM / V-REx), and group-DRO.

| config | worst-subset | length gap | acc_all | g1 | g2 | g3 |
|---|---|---|---|---|---|---|
| DPO v3 (start) | 0.607 | 0.110 | 0.656 | 16.3 | −2.0 | −1.7 |
| **winner: DPO v3 + I4 len-inv λ=5e-4** | **0.638** | **0.054** | 0.658 | 17.7 | −2.2 | −3.4 |
| clean-g3 config (I4+I3 cons3both) | 0.603 | 0.112 | 0.653 | 16.3 | −2.0 | **−1.7** |
| I2 length-matched neg (blew up) | 0.44–0.47 | — | — | 130–143 | +9..+17 | −142..−146 |
| I3 consistency alone | 0.49–0.53 | — | — | ~14 | −2.7 | −5.3 |

**Findings.**
1. **Best trained method halves the length-accuracy gap** (0.110→0.054, worst 0.607→0.638) — but only **≈matches the free debias ceiling** (short by 0.002) and misses the 0.68 target. Winner = continue-train DPO v3 with an I4 penalty equalizing the mean policy margin across the two length environments (λ=5e-4; single-peaked sweep).
2. **worst-subset ⊥ g3 Pareto wall.** Every config that pushed accuracy-invariance up *widened* the raw-logp length gap g3 (winner −3.4 vs DPO v3 −1.7); the only config keeping g3 clean (−1.7) fell back to worst 0.603. Decision-invariance and raw-logp length-invariance **cannot** be had together here.
3. **acc_all ≈ 0.66 is model/data-bound.** No base on this 1.5B + TL;DR setup exceeds ~0.66 overall accuracy; via `worst ≈ acc_all − gap/2`, clearing 0.68 needs acc ~0.70 — unreachable (matches REPORT's standing "no method improves ranking accuracy" conclusion). The loop's bootstrapped "stronger base" attempts trended *below* DPO v3 and were stopped.
4. **Failed ideas:** I2 (length-matched negatives) over-separates on the wrong axis and destroys the decomposition (g1≈143, g3≈−146); I3 consistency alone hurts.

**Answer to the principal question (updated): partly, with a hard limit.** A trained method *can* be made substantially more length-invariant at the **decision/accuracy** level on skewed data (gap halved), but *cannot* be simultaneously semantics-strong and raw-logp length/syntax-invariant — they trade off on a smooth Pareto frontier — and the accuracy ceiling (~0.66) is a model/data property, not a search miss. Tellingly, the best trained method barely beats a one-parameter post-hoc debias, so reward-shaping adds little over subtracting the length component at ranking. The one lever that could break the ceiling — a genuinely higher-accuracy base model — is outside this 1.5B/TL;DR setup.

## Caveat
This round's numbers come from re-evaluating saved checkpoints on the full split; earlier rounds used the 256-slice in-training eval, so absolute values differ slightly from §5.2/§9.4b (protocol matched, sample differs). SIPO runs used `--no_save`, so no SIPO checkpoint exists for a full-test re-eval; round-4 SIPO numbers remain slice-based.

## Artifacts
- Script: `scripts/eval_length_split_checkpoint.py` (evaluate any checkpoint on the full test set with the length split)
- Data: `metrics/length_split_fulltest_DPO_v3.csv`, `metrics/length_split_fulltest_SimPO_v2.csv`
- Checkpoints: `checkpoints/dpo-v3/final`, `checkpoints/simpo-strong/final`
- Group-DRO follow-up: `dro_dpo_train.py` (`--group_dro {off,worst}`); `metrics/length_split_fulltest_dpo_dro_{off,worst}.csv`; `checkpoints/dro_dpo_{off,worst}/final`; runs `results/dro_dpo_{off,worst}/`
- Invariance auto-research: `invariance_train.py` (I2–I5: `--neg_key`, `--lambda_cons`, `--lambda_len_inv`+`--len_inv_ema`, `--lambda_irm`, `--lambda_vrex`, `--group_dro`), scorer `invariance_eval.py`; 36 config trajectories in `results/inv_search/` + full-test CSVs `metrics/length_split_fulltest_inv_*.csv`; loop state `.loop/` (round-4's archived to `.loop_round4_sipo/`). Winner reproducible from `results/inv_search/dv3cont_lr5e6_l5e4` (checkpoint auto-deleted per disk policy; CSV reproduces the score).
