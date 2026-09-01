# Follow-up Tasks — Round 6 (added 2026-07-27)

**Question.** Do published preference-optimization methods that *claim* length
robustness / invariance actually reduce the TL;DR length bias — and can any beat
DPO v3, the most length-robust method so far (§9.8)?

**Scope.** Eight **offline** methods (RL methods Dr.GRPO / GSPO deferred to Round 7,
as they need a reward model + rollouts, an entirely separate stack). Trained on the
headline recipe — full `trl-lib/tldr-preference` (92,858 pairs), LR 5e-6, 2 epochs,
batch 8 × grad-accum 2, Qwen2.5-1.5B-Instruct — and evaluated on the **full
2000-example** decomposed `validation` split by length direction (same protocol as
§9.8). Tie Training runs 1 epoch over a 2× (strict+tie) mixture, matching
examples-seen. One hyperparameter setting per method (no sweeps).

**Metric focus.** `worst_subset = min(acc_chosen_longer, acc_rejected_longer)` (the
Round-5 headline metric), `length_acc_gap`, and the decomposition gaps
g1 (semantics, chosen − rejected′), g2 (syntax), g3 (length, rejected″ − rejected).

## Methods & implementation

| Method | Paper | Length claim | Implementation |
|---|---|---|---|
| ORPO | 2403.07691 | length-averaged odds-ratio | TRL `experimental.orpo` (native) |
| LD-DPO | TRL native | `ld_alpha` down-weights verbose tail | TRL DPO `ld_alpha=0.5` (native) |
| rDPO | 2403.00409 | label-noise robustness | TRL DPO `loss_type=robust`, ε=0.1 (native) |
| R-DPO | 2403.19159 | explicit `−α(\|y_w\|−\|y_l\|)` penalty | `length_pref.py`, α=0.05 |
| Dr.DPO | 2407.07880 | DRO down-weights outlier pairs | `length_pref.py`, β′=1.0 |
| SamPO | 2406.10957 | down-sampled per-token KL to `min(len)` | `length_pref.py` |
| LMPO | 2502.14643 | length-scaled probability margin | `length_pref.py`, β=2.5, γ-ratio 0.55, λ=0.2, k=5 |
| Tie | 2605.11134 | equal-utility length-ties, random label | `length_pref.py` + `scripts/build_tie_dataset.py`, α=0.5 |

Custom-loss formulas were taken from each paper's **authors' code** (verified). The
custom trainer's plain-`dpo` mode reproduces TRL's DPO init loss (0.693 = −ln 0.5),
so the variants inherit that faithfulness. Native methods use genuine TRL trainers.

## Results (full 2000-example decomposed test)

| method | acc | g1 | g2 | g3 | g3/tok | acc_cl | acc_rl | len_gap | **worst** |
|---|---|---|---|---|---|---|---|---|---|
| **Dr.DPO** | 0.648 | 21.95 | −2.19 | −2.73 | −0.018 | 0.614 | 0.687 | **0.073** | **0.614** |
| **R-DPO** | 0.649 | 15.38 | −2.04 | −3.10 | −0.047 | 0.612 | 0.694 | 0.082 | 0.612 |
| LD-DPO | 0.655 | 15.13 | −2.04 | −2.89 | −0.040 | 0.595 | 0.729 | 0.135 | 0.595 |
| rDPO | 0.653 | 15.80 | −2.02 | −2.32 | −0.019 | 0.587 | 0.735 | 0.148 | 0.587 |
| SamPO | 0.659 | 15.20 | −2.26 | −3.01 | −0.045 | 0.586 | 0.749 | 0.163 | 0.586 |
| Tie | 0.628 | 16.25 | −1.67 | −6.08 | −0.135 | 0.570 | 0.707 | 0.137 | 0.570 |
| ORPO | 0.611 | 30.59 | −1.01 | −23.94 | −0.657 | 0.489 | 0.759 | 0.270 | 0.489 |
| LMPO | 0.579 | 8.32 | 1.58 | −0.83 | 0.039 | 0.332 | 0.869 | 0.537 | 0.332 |
| *DPO v3 (ref, §9.8)* | 0.656 | 16.32 | −2.04 | −1.73 | −0.001 | 0.607 | 0.717 | 0.110 | 0.607 |
| *SimPO v2 (ref, §9.8)* | 0.621 | 17.39 | −1.84 | −9.12 | −0.230 | 0.460 | 0.814 | 0.354 | 0.460 |

Plot: `results/round6_length_robust/round6_comparison.png` (worst-subset vs |g3|).
Data: `metrics/round6_summary.csv`, `metrics/round6_<method>_fulltest.csv`.

## Findings

1. **Only Dr.DPO and R-DPO beat DPO v3's length robustness — and only marginally.**
   Dr.DPO: worst-subset 0.614 (vs DPO v3 0.607), length_acc_gap 0.073 (vs 0.110),
   overall acc 0.648, strongest healthy g1 (21.95). R-DPO (the purpose-built length
   penalty): worst 0.612, gap 0.082. Neither reaches the Round-5 *trained* invariance
   winner (DPO v3 + I4 len-inv, worst 0.638, §9.8). **Notably Dr.DPO is not a length
   method** — its DRO down-weighting of high-loss (often length-exploiting) pairs
   *incidentally* trims the worst length subset.
2. **LD-DPO, rDPO, SamPO ≈ DPO v3** (worst 0.586–0.595): no clear improvement. SamPO's
   token down-sampling did not translate into decision-level length invariance here.
3. **ORPO is strongly brevity-biased**: g3 = −23.94 and acc_chosen_longer = 0.489
   (below chance) despite the largest semantic separation (g1 = 30.59). Its
   length-averaged odds-ratio + monolithic SFT raise absolute likelihood but bake in
   a hard short-preference — **length averaging ≠ length invariance**. (Un-swept β/lr.)
4. **LMPO is the sharpest instance of the raw-logp ⊥ decision Pareto wall (§9.7/§9.8):**
   it attains the *smallest* raw-logp length gap (|g3| = 0.83, near DPO v3) yet the
   *worst* decision-level length bias (length_acc_gap 0.537; acc_chosen_longer 0.332)
   and weak semantics (g1 8.32). Length-neutral log-probs and length-invariant
   *decisions* are distinct — and here anti-correlated. (Possibly hparam-sensitive.)
5. **Tie Training** did not beat DPO v3 (worst 0.570, g3 −6.08). The length-tie pool is
   small (≈3000 decomposed-train pairs, upsampled to α=0.5) and low-diversity — the
   most likely limiter; a larger, quality-controlled tie pool is the obvious follow-up.
6. **The ~0.66 accuracy ceiling and the worst ⊥ g3 wall survive all eight published
   methods.** No off-the-shelf offline method escapes them, strongly corroborating the
   cumulative conclusion (§9.6/§9.8): the TL;DR length bias is **objective-level** and
   the accuracy ceiling is **model/data-bound** on 1.5B + TL;DR.

## Caveats
- One hyperparameter config per method; ORPO and LMPO in particular look under-tuned.
- Tie pool small/low-diversity (see finding 5).
- Dr.GRPO / GSPO (online RL) deferred to Round 7 (need a reward model + rollouts).

## Artifacts
- Code: `train.py` (+`--method orpo/rdpo/drdpo/sampo/lmpo/tie`, `--ld_alpha`,
  `--label_smoothing`), `length_pref.py`, `scripts/build_tie_dataset.py`.
- Run/eval: `scripts/run_round6.sh`, `scripts/eval_round6.sh`,
  `scripts/summarize_round6.py`, `scripts/plot_round6.py`.
- Checkpoints: `checkpoints/round6/<method>/final`; trajectories:
  `results/round6_length_robust/<method>/`.
