# Follow-up Tasks — Round 4 — SIPO + auto-research (HONEST NEGATIVE RESULT)

*Date: 2026-07-05. Goal: a new method (SIPO) that combines SimPO's semantic separation (g1~19) with DPO's length-neutrality (g3~0) — i.e. "learn preference by semantics, not length/syntax." Driven by an autonomous planner/generator/evaluator loop. Full detail in `REPORT.md` §9.7; this file is the frozen round-4 digest.*

> **Convention.** Frozen per-round digest; `REPORT.md` is the superset. Rounds 1–3: `FOLLOWUP_TASKS_ROUND{1,2,3}.md`.

## Principal scientific question
Does alignment learn preference by **semantics** rather than **length or syntax**? The decomposition chain answers it per-gap: g1=chosen−rejected′ (semantics), g2=rejected′−rejected″ (syntax), g3=rejected″−rejected (length). Target: g1 large, g2≈0, g3≈0.

## Method — SIPO (Semantics-Isolating Preference Optimization)
Reference-free, trained on the 3k 4-way decomposed set (`sipo.py`, `train.py --method sipo`):
```
L = -logσ(β(r_chosen − r_rejected′) − γ)      # semantic preference on the length+syntax-MATCHED pair
  + λ_syn·Inv(r_rejected′, r_rejected″)         # syntax invariance → g2→0
  + λ_len·Inv(r_rejected″, r_rejected)          # length invariance → g3→0
```
Invariance default `smooth_l1` (L2 on summed logps explodes: grad-norm ~2.5e5).

## Auto-research process
`running-agent-loops` harness: separate `claude -p` generator (picks config / edits `sipo.py`, trains on a B200) and evaluator (runs `sipo_eval.py`, scores vs a research rubric, can order a restart); state in `.loop/`. Ran **40+ configs over 10 iterations + 1 restart**, converged (identical evaluator verdict iters 4–10). It correctly refused to fake success and escalated (`ESCALATION.md`). The generator autonomously found real levers (decoupled invariance reward scale, per-position detach, an accuracy term).

## Result — SIPO did NOT achieve the goal (verified independently)

End-of-training, summed-logp scale (the panel-(d) target):

| run | g1 (want ~19) | g3 (want ~0) | g2 | acc | length-acc-gap | composite |
|---|---|---|---|---|---|---|
| SimPO v2 | **19.0** | −5.1 | −6.8 | .61 | .69 | — |
| **DPO v3** | 17.4 | **−1.1** | −7.1 | .60 | — | — |
| SIPO iter3m (best of 40) | 16.7 | −1.27 | **−9.0** | .60 | .459 | 0.4498 |
| SIPO baseline v0 | 10.0 | +3.74 | −8.2 | .60 | .669 | 0.1457 |

**SIPO iter3m is essentially a slightly-worse DPO v3** — DPO already sits nearest the target (higher g1 17.4>16.7, better |g3| 1.1<1.27) and *nothing beat it*. SIPO also regressed on syntax (g2 −9.0, worse than SimPO's −6.8). No config reached the aspirational bar (g1≥18, |g2|≤2, |g3|≤2, gap≤0.2, score≥0.8) — **and neither does SimPO or DPO**.

## Findings (both verified, not just loop-asserted)

1. **The target is provably infeasible in-scope: g1 ⊥ g3 tradeoff.** On the summed scale, pushing semantic separation up drags length-neutrality down (iter16: g1=22.6 → g3=−8.1; iter04: g1=20 → g3=−5.3; iter2r: g1=16.0 → g3=−0.4). SimPO's green (g1~19) and DPO's purple (g3~0) lie at opposite ends of one frontier; the target corner (top-right) is unreached by 40+ configs. Figure: `results/length_split/sipo_g1_g3_frontier.png`.
2. **The answer to the principal question is SCALE-DEPENDENT — and the summed metric manufactures the length/syntax story.** Per-token (mean logp), *every* method incl. the base model has tiny g2/g3 and semantics-dominant g1; the ~25× token multiplier inflates summed g2 to −9. iter3m: sum (g1 16.7, g2 −9.0, g3 −1.3) vs per-token (g1 +0.46, g2 −0.19, g3 −0.17), length-acc-gap 0.459 (sum) vs 0.038 (per-token).
3. **SIPO gives no per-token advantage.** Per-token profiles (avg last 20 evals): DPO v3 is best (g1/|g2| = 3.70, length-gap 0.018), SimPO 3.86/0.130, SIPO iter3m only 2.38/0.038, base 1.66/0.009. SIPO's per-token |g2| (0.20) is *worse* than SimPO/DPO (0.13–0.14). The apparent summed-scale "length win" is not backed by a per-token improvement.

## Honest caveats
- The loop's progress.md slightly oversold iter3m ("beats every reference on length" — true vs SimPO/baseline, but **not vs DPO v3**). Corrected here.
- `score≥0.8` is unreachable for any method on this paraphrase-pair data — a Pareto conflict + summed-logp metric artifact, not specifically a SIPO failure.

## Open decision (escalated; awaiting human)
- **Option B (default taken):** accept the frontier, record this negative result. ← done (this digest).
- **Option C (not run):** add a syntax metric not confounded by summed-logp, re-run SimPO/DPO/base on the decomposed eval for honest mean-scale anchors, and try a method that breaks the g1⊥g3 frontier (not just reward-shaping). Only path to a genuine `score≥0.8`.
- **Rethink:** the panel-(d) summed target may be the wrong objective.

## Artifacts
- Method: `sipo.py`, `train.py --method sipo`; scorer `sipo_eval.py`
- Runs: `results/sipo_search/v0/sipo`, `results/sipo_search/iter*/sipo` (40+); winner `results/sipo_search/iter3m/sipo`
- Figure: `results/length_split/sipo_g1_g3_frontier.png`
- Loop state: `.loop_round4_sipo/` (spec, contract, feature_list, rubric, log, progress, verdict); escalation `ESCALATION.md`
- Env: `.venv` (reconstructed)
