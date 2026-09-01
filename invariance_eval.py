"""Score an invariance run's full-test CSV against the goal.

Primary metric = worst-subset accuracy = min(acc_chosen_longer, acc_rejected_longer)
on the full test set (high only if BOTH length skews are handled -> semantics +
length-invariance at once, non-degenerate). Reports g1/g2/g3 + syntax/length gaps.

Usage: python invariance_eval.py --csv metrics/length_split_fulltest_<label>.csv
(the CSV is produced by scripts/eval_length_split_checkpoint.py)
"""
import argparse
import json

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()
    d = pd.read_csv(args.csv)
    cl = d[d.chosen_tokens > d.rejected_tokens]
    rl = d[d.chosen_tokens < d.rejected_tokens]

    def acc(x):
        return float(((x.chosen_logp - x.rejected_logp) > 0).mean())

    def gap(x, a, b):
        return float((x[f"{a}_logp"] - x[f"{b}_logp"]).mean())

    acl, arl, aall = acc(cl), acc(rl), acc(d)
    out = {
        "csv": args.csv,
        "n": int(len(d)),
        "worst_subset_acc": round(min(acl, arl), 4),      # PRIMARY (maximize)
        "acc_chosen_longer": round(acl, 4),
        "acc_rejected_longer": round(arl, 4),
        "acc_all": round(aall, 4),
        "length_acc_gap": round(abs(acl - arl), 4),
        "g1_semantics": round(gap(d, "chosen", "rejected_prime"), 3),
        "g2_syntax": round(gap(d, "rejected_prime", "rejected_double_prime"), 3),
        "g3_length": round(gap(d, "rejected_double_prime", "rejected"), 3),
        "targets": "maximize worst_subset_acc (bar: DPO v3 0.607, debias ceiling ~0.64); |g2|,|g3|->0, g1 large",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
