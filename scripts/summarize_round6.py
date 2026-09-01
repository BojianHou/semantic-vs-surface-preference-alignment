"""Summarize Round-6 length-robust runs: acc, g1/g2/g3, and the length-direction
accuracy split, from the full-test decomposed CSVs (metrics/round6_<label>_fulltest.csv,
produced by scripts/eval_round6.sh). Prints a comparison table and writes
metrics/round6_summary.csv.

Gaps (sum-logp, matching REPORT §9.8):
    g1 = chosen - rejected_prime           (semantics; want large)
    g2 = rejected_prime - rejected_double_prime  (syntax; want ~0)
    g3 = rejected_double_prime - rejected  (length; want ~0)
"""
import argparse
import glob
import os

import pandas as pd

# Reference anchors from REPORT §9.8 (full 2000 test), for context in the table.
REFERENCE_ROWS = [
    # label, acc, g1, g2, g3, acc_cl, acc_rl, len_gap, g3_pertoken
    ("DPO v3 (ref)", 0.656, 16.32, -2.04, -1.73, 0.607, 0.717, 0.110, -0.001),
    ("SimPO v2 (ref)", 0.621, 17.39, -1.84, -9.12, 0.460, 0.814, 0.354, -0.230),
]


def metrics_from_csv(path):
    df = pd.read_csv(path)
    g1 = (df.chosen_logp - df.rejected_prime_logp).mean()
    g2 = (df.rejected_prime_logp - df.rejected_double_prime_logp).mean()
    g3 = (df.rejected_double_prime_logp - df.rejected_logp).mean()
    g3_pt = (df.rejected_double_prime_mean_logp - df.rejected_mean_logp).mean()
    acc = ((df.chosen_logp - df.rejected_logp) > 0).mean()
    cl = df[df.chosen_tokens > df.rejected_tokens]
    rl = df[df.chosen_tokens < df.rejected_tokens]
    acc_cl = ((cl.chosen_logp - cl.rejected_logp) > 0).mean()
    acc_rl = ((rl.chosen_logp - rl.rejected_logp) > 0).mean()
    return dict(n=len(df), acc=acc, g1=g1, g2=g2, g3=g3, g3_pertoken=g3_pt,
                acc_chosen_longer=acc_cl, acc_rejected_longer=acc_rl,
                length_acc_gap=abs(acc_cl - acc_rl),
                worst_subset=min(acc_cl, acc_rl))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="metrics/round6_*_fulltest.csv")
    p.add_argument("--out", default="metrics/round6_summary.csv")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        label = os.path.basename(path).replace("round6_", "").replace("_fulltest.csv", "")
        m = metrics_from_csv(path)
        m["method"] = label
        rows.append(m)
    if not rows:
        print(f"no CSVs matched {args.glob} — run scripts/eval_round6.sh first")
        return

    df = pd.DataFrame(rows).set_index("method")
    cols = ["n", "acc", "g1", "g2", "g3", "g3_pertoken",
            "acc_chosen_longer", "acc_rejected_longer", "length_acc_gap", "worst_subset"]
    df = df[cols].sort_values("worst_subset", ascending=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out)

    # Pretty print with reference anchors appended.
    print(f"\nRound-6 length-robust methods — full 2000-example decomposed test\n")
    hdr = f"{'method':16s} {'acc':>5s} {'g1':>6s} {'g2':>6s} {'g3':>7s} {'g3/tok':>7s} " \
          f"{'acc_cl':>6s} {'acc_rl':>6s} {'lgap':>5s} {'worst':>6s}"
    print(hdr); print("-" * len(hdr))
    for m, r in df.iterrows():
        print(f"{m:16s} {r.acc:5.3f} {r.g1:6.2f} {r.g2:6.2f} {r.g3:7.2f} {r.g3_pertoken:7.3f} "
              f"{r.acc_chosen_longer:6.3f} {r.acc_rejected_longer:6.3f} "
              f"{r.length_acc_gap:5.3f} {r.worst_subset:6.3f}")
    print("-" * len(hdr))
    for (lab, acc, g1, g2, g3, cl, rl, gap, g3pt) in REFERENCE_ROWS:
        print(f"{lab:16s} {acc:5.3f} {g1:6.2f} {g2:6.2f} {g3:7.2f} {g3pt:7.3f} "
              f"{cl:6.3f} {rl:6.3f} {gap:5.3f} {min(cl, rl):6.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
