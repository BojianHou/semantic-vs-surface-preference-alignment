"""Deterministic scorer for a SIPO (or any decomposed) run — grounds the loop.

Reads a run's decomposed_summary.csv (+ last per-step CSV) and computes the
panel-(d) targets. The research goal (learn preference by SEMANTICS, not
length/syntax):

    g1 semantics (chosen - rejected')            -> LARGE  (SimPO reaches ~19)
    g2 syntax    (rejected' - rejected'')         -> ~0     (init -9.45)
    g3 length    (rejected'' - rejected)          -> ~0     (SimPO crushes to -5)
    accuracy P(chosen>rejected)                    -> maintained (>=0.60)
    length-direction acc gap |acc_cl - acc_rl|     -> SMALL  (baseline ~0.69: 0.21 vs 0.90)

Prints a JSON object with the raw metrics, per-axis subscores in [0,1], and a
weighted composite `score`. Usage:
    python sipo_eval.py --run_dir results/sipo_search/v0/sipo [--last_k 20]

Reference anchors (end-of-training, from REPORT.md §5.2 / §9.4b):
    init: g1=+15.94 g2=-9.45 g3=-0.05 ; SimPO v2: g1=19.0 g2=-6.8 g3=-5.1 acc .61
    DPO v3: g1=17.4 g2=-7.1 g3=-1.1 acc .60 ; baseline length-acc-gap 0.69
"""
import argparse
import glob
import json

import pandas as pd

G1 = "mean_logp_chosen_minus_rejected_prime"
G2 = "mean_logp_rejected_prime_minus_rejected_double_prime"
G3 = "mean_logp_rejected_double_prime_minus_rejected"
ACC = "chosen_gt_rejected_acc"


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="e.g. results/sipo_search/v0/sipo")
    p.add_argument("--last_k", type=int, default=20)
    args = p.parse_args()

    summary = f"{args.run_dir}/decomposed_summary.csv"
    d = pd.read_csv(summary)
    tail = d.tail(args.last_k).mean(numeric_only=True)
    g1, g2, g3, acc = float(tail[G1]), float(tail[G2]), float(tail[G3]), float(tail[ACC])

    # length-direction accuracy split from the last per-step CSV
    steps = sorted(glob.glob(f"{args.run_dir}/decomposed_step*.csv"),
                   key=lambda q: int(q.split("step")[-1].split(".")[0]))
    last = pd.read_csv(steps[-1])
    cl = last[last.chosen_tokens > last.rejected_tokens]
    rl = last[last.chosen_tokens < last.rejected_tokens]
    acc_cl = float(((cl.chosen_logp - cl.rejected_logp) > 0).mean()) if len(cl) else float("nan")
    acc_rl = float(((rl.chosen_logp - rl.rejected_logp) > 0).mean()) if len(rl) else float("nan")
    len_gap = abs(acc_cl - acc_rl)

    # per-axis subscores in [0,1]
    sem = clip01((g1 - 14.0) / (19.0 - 14.0))       # 14->0, 19(SimPO)->1
    syn = clip01(1.0 - abs(g2) / 9.45)              # 0->1, 9.45(init)->0
    length = clip01(1.0 - abs(g3) / 5.0)            # 0->1, 5(SimPO)->0
    accs = clip01((acc - 0.50) / (0.66 - 0.50))     # 0.50->0, 0.66->1
    lensplit = clip01(1.0 - len_gap / 0.69)         # 0.69(baseline)->0, 0->1

    weights = {"sem": 0.30, "syn": 0.20, "length": 0.20, "acc": 0.10, "lensplit": 0.20}
    subs = {"sem": sem, "syn": syn, "length": length, "acc": accs, "lensplit": lensplit}
    score = sum(weights[k] * subs[k] for k in weights)

    out = {
        "run_dir": args.run_dir,
        "n_evals": int(len(d)),
        "last_step": int(d.step.iloc[-1]),
        "metrics": {"g1_semantics": round(g1, 3), "g2_syntax": round(g2, 3),
                    "g3_length": round(g3, 3), "acc": round(acc, 3),
                    "acc_chosen_longer": round(acc_cl, 3), "acc_rejected_longer": round(acc_rl, 3),
                    "length_acc_gap": round(len_gap, 3)},
        "subscores": {k: round(v, 3) for k, v in subs.items()},
        "weights": weights,
        "score": round(score, 4),
        "targets": "g1>=18, |g2|<=2, |g3|<=2, acc>=0.60, length_acc_gap<=0.2",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
