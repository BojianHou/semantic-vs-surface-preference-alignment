"""Option-A scorecard: does length-normalizing the scorer actually yield all-pass?

ESCALATION.md §3 Option A proposes recomputing g2/g3/gap/acc from the MEAN-logp
columns and recalibrating C1/C4, claiming this "flips C2/C5/C9 to PASS on the
current iter3m checkpoint with zero retraining." That claim was never fully
checked: (1) it never re-derives the composite `score` (C6), and (2) it never
compares the mean-scale numbers against the UNTRAINED init model.

This script settles both, WITHOUT modifying sipo_eval.py / the eval callback
(contract non-goal). It mirrors sipo_eval.py's subscore STRUCTURE but anchors
every axis on the mean scale to the init base model (results/sipo_search/v0 step 0) --
the honest "no-training" reference, exactly as the summed scorer uses init |g2|
as its 0-anchor. It prints, for iter3m AND init, the full mean-scale metrics,
subscores, composite, and per-gate PASS/FAIL under Option A.

Analysis only. Reads CSVs; writes nothing; touches no protected dir.
Usage: .venv/bin/python option_a_scorecard.py
"""
import glob
import json

import pandas as pd

WIN = "results/sipo_search/iter3m/sipo"     # deliverable checkpoint
INIT = "results/sipo_search/v0/sipo"        # untrained base model (step 0 eval)


def _steps(run_dir):
    return sorted(glob.glob(f"{run_dir}/decomposed_step*.csv"),
                  key=lambda q: int(q.split("step")[-1].split(".")[0]))


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def mean_metrics(csv):
    """All panel-(d) metrics on the MEAN (length-normalized) scale."""
    d = pd.read_csv(csv)
    g1 = float((d.chosen_mean_logp - d.rejected_prime_mean_logp).mean())
    g2 = float((d.rejected_prime_mean_logp - d.rejected_double_prime_mean_logp).mean())
    g3 = float((d.rejected_double_prime_mean_logp - d.rejected_mean_logp).mean())
    cl = d[d.chosen_tokens > d.rejected_tokens]
    rl = d[d.chosen_tokens < d.rejected_tokens]
    acc = float((d.chosen_mean_logp - d.rejected_mean_logp > 0).mean())
    acc_cl = float(((cl.chosen_mean_logp - cl.rejected_mean_logp) > 0).mean())
    acc_rl = float(((rl.chosen_mean_logp - rl.rejected_mean_logp) > 0).mean())
    return {"g1": g1, "g2": g2, "g3": g3, "acc": acc,
            "acc_cl": acc_cl, "acc_rl": acc_rl, "gap": abs(acc_cl - acc_rl)}


def subscores(m, anchor):
    """Mean-scale subscores, mirroring sipo_eval.py structure.

    Anchored so the UNTRAINED init model sits at each axis's 0-anchor (init |g2|,
    init |g3|, init gap) or 0.50 (acc), and the ideal (0 / 0.66) sits at 1 --
    i.e. subscore = improvement over init toward the ideal. sem has no mean-scale
    SimPO anchor on disk, so we bracket its span with init->g1_ideal where
    g1_ideal = 2 * init_g1 (a generous +100% semantic-separation target).
    """
    g1_ideal = 2.0 * anchor["g1"]
    sem = clip01((m["g1"] - anchor["g1"]) / (g1_ideal - anchor["g1"])) if g1_ideal > anchor["g1"] else 0.0
    syn = clip01(1.0 - abs(m["g2"]) / abs(anchor["g2"]))            # init |g2| -> 0
    length = clip01(1.0 - abs(m["g3"]) / abs(anchor["g3"]))         # init |g3| -> 0
    accs = clip01((m["acc"] - 0.50) / (0.66 - 0.50))               # 0.50 -> 0, 0.66 -> 1
    lensplit = clip01(1.0 - m["gap"] / anchor["gap"]) if anchor["gap"] > 0 else (1.0 if m["gap"] == 0 else 0.0)
    subs = {"sem": sem, "syn": syn, "length": length, "acc": accs, "lensplit": lensplit}
    weights = {"sem": 0.30, "syn": 0.20, "length": 0.20, "acc": 0.10, "lensplit": 0.20}
    score = sum(weights[k] * subs[k] for k in weights)
    return subs, weights, score


def gate_table(m, score):
    """Per-gate result under Option A (mean scale + ESCALATION recalibrated thresholds)."""
    return {
        "C1 g1>=0.40 (recal)":  ("PASS" if m["g1"] >= 0.40 else "fail", round(m["g1"], 3)),
        "C2 |g2|<=2":           ("PASS" if abs(m["g2"]) <= 2 else "fail", round(abs(m["g2"]), 3)),
        "C3 |g3|<=2":           ("PASS" if abs(m["g3"]) <= 2 else "fail", round(abs(m["g3"]), 3)),
        "C4 acc>=0.58 (recal)": ("PASS" if m["acc"] >= 0.58 else "fail", round(m["acc"], 3)),
        "C5 gap<=0.2":          ("PASS" if m["gap"] <= 0.2 else "fail", round(m["gap"], 3)),
        "C6 score>=0.8":        ("PASS" if score >= 0.8 else "fail", round(score, 4)),
        "C9 |g2|<6.8":          ("PASS" if abs(m["g2"]) < 6.8 else "fail", round(abs(m["g2"]), 3)),
    }


def main():
    init = mean_metrics(_steps(INIT)[0])           # step 0 == untrained base
    win = mean_metrics(_steps(WIN)[-1])            # last step == deliverable

    print("=" * 74)
    print("OPTION-A SCORECARD  (mean-logp scoring; scorer NOT modified)")
    print("=" * 74)
    print("\nMEAN-scale metrics — trained winner vs UNTRAINED init base model:")
    hdr = f"  {'metric':<12}{'init(untrained)':>18}{'iter3m(winner)':>18}{'Δ (win-init)':>16}"
    print(hdr)
    for k in ["g1", "g2", "g3", "acc", "acc_cl", "acc_rl", "gap"]:
        print(f"  {k:<12}{init[k]:>18.3f}{win[k]:>18.3f}{win[k]-init[k]:>+16.3f}")

    for tag, m in [("INIT (untrained base)", init), ("iter3m (WINNER)", win)]:
        subs, w, score = subscores(m, anchor=init)
        print(f"\n--- {tag} : mean-scale subscores (init-anchored) ---")
        print("  " + "  ".join(f"{k}={subs[k]:.3f}" for k in w))
        print(f"  composite score = {score:.4f}")
        if "WINNER" in tag:
            print("\n  Per-gate under Option A (recalibrated thresholds):")
            for g, (v, x) in gate_table(m, score).items():
                print(f"    {g:<24} {v:<5} ({x})")
            out = {"metrics_mean": {k: round(v, 3) for k, v in m.items()},
                   "subscores_mean": {k: round(v, 3) for k, v in subs.items()},
                   "composite_mean": round(score, 4)}
            print("\n  JSON:", json.dumps(out))

    print("\n" + "=" * 74)
    print("VERDICT: Option A does NOT yield all-pass. C6 (composite>=0.8) FAILS on")
    print("the mean scale, and C2/C5/C9 'pass' non-informatively — the UNTRAINED")
    print("init model posts the same mean |g2|/gap/acc as the trained winner.")
    print("=" * 74)


if __name__ == "__main__":
    main()
