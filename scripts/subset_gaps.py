"""Per-length-subset decomposition gaps (g1/g2/g3) for every evaluated checkpoint.

REPORT §9.9's Round-6 table reports g1/g2/g3 on the FULL test only, and the
length split as accuracy alone. §9.8 reports per-subset gaps but for DPO v3 /
SimPO v2 only. This script closes that gap: it recomputes the §9.8-style
per-subset breakdown for every method that has a full-test per-example CSV.

Pure recompute — the CSVs written by scripts/eval_length_split_checkpoint.py
already carry all four objects' summed/mean logps and token counts, so no model
forward pass is needed.

Gaps (sum-logp, matching REPORT §9.8):
    g1 = chosen - rejected_prime                 (semantics; want large)
    g2 = rejected_prime - rejected_double_prime  (syntax;    want ~0)
    g3 = rejected_double_prime - rejected        (length;    want ~0)

Subsets by response-length direction (Qwen2.5 tokens), ties excluded from the
two subsets but kept in `all`:
    chosen_longer    chosen_tokens > rejected_tokens   (n=1031)
    rejected_longer  chosen_tokens < rejected_tokens   (n=860)

Usage (from project root):
    .venv/bin/python scripts/subset_gaps.py
    .venv/bin/python scripts/subset_gaps.py --out metrics/subset_gaps.csv
"""

import argparse
import glob
import os

import pandas as pd

# (label, path). Round-6 methods first, then the standing reference checkpoints.
SOURCES = [
    ("drdpo", "metrics/round6_drdpo_fulltest.csv"),
    ("rdpo", "metrics/round6_rdpo_fulltest.csv"),
    ("lddpo", "metrics/round6_lddpo_fulltest.csv"),
    ("rdpo_robust", "metrics/round6_rdpo_robust_fulltest.csv"),
    ("sampo", "metrics/round6_sampo_fulltest.csv"),
    ("tie", "metrics/round6_tie_fulltest.csv"),
    ("orpo", "metrics/round6_orpo_fulltest.csv"),
    ("lmpo", "metrics/round6_lmpo_fulltest.csv"),
    # References (REPORT §9.8): the round-5 leader, the SimPO contrast, and the
    # round-5 trained-invariance winner (DPO v3 + I4 len-inv lambda=5e-4).
    ("DPO_v3", "metrics/length_split_fulltest_DPO_v3.csv"),
    ("SimPO_v2", "metrics/length_split_fulltest_SimPO_v2.csv"),
    ("I4_winner", "metrics/length_split_fulltest_inv_dv3cont_lr5e6_l5e4.csv"),
]


def _gaps(df):
    """Decomposition gaps + absolute logps for one (sub)frame."""
    return dict(
        n=len(df),
        acc=((df.chosen_logp - df.rejected_logp) > 0).mean(),
        g1=(df.chosen_logp - df.rejected_prime_logp).mean(),
        g2=(df.rejected_prime_logp - df.rejected_double_prime_logp).mean(),
        g3=(df.rejected_double_prime_logp - df.rejected_logp).mean(),
        g1_pertoken=(df.chosen_mean_logp - df.rejected_prime_mean_logp).mean(),
        g2_pertoken=(df.rejected_prime_mean_logp - df.rejected_double_prime_mean_logp).mean(),
        g3_pertoken=(df.rejected_double_prime_mean_logp - df.rejected_mean_logp).mean(),
        # Absolute levels — exposes likelihood displacement alongside the gaps.
        chosen_logp=df.chosen_logp.mean(),
        rejected_logp=df.rejected_logp.mean(),
        chosen_tokens=df.chosen_tokens.mean(),
        rejected_tokens=df.rejected_tokens.mean(),
    )


def rows_for(label, path):
    df = pd.read_csv(path)
    subsets = {
        "chosen_longer": df[df.chosen_tokens > df.rejected_tokens],
        "rejected_longer": df[df.chosen_tokens < df.rejected_tokens],
        "all": df,
    }
    out = []
    for name, sub in subsets.items():
        r = _gaps(sub)
        r["method"], r["subset"] = label, name
        out.append(r)
    # worst_subset is a property of the method, not of a subset; attach to `all`.
    cl = out[0]["acc"]
    rl = out[1]["acc"]
    for r in out:
        r["worst_subset"] = min(cl, rl)
        r["length_acc_gap"] = abs(cl - rl)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="metrics/subset_gaps.csv")
    args = p.parse_args()

    # Round-7 Dr.DPO-tuning runs are auto-discovered so this stays current as the
    # sweep lands, without editing SOURCES for each new config.
    sources = list(SOURCES) + [
        (os.path.basename(p).replace("round7_", "").replace("_fulltest.csv", ""), p)
        for p in sorted(glob.glob("metrics/round7_*_fulltest.csv"))
    ]

    rows, missing = [], []
    for label, path in sources:
        if not os.path.exists(path):
            missing.append(path)
            continue
        rows.extend(rows_for(label, path))
    if not rows:
        print("no input CSVs found — run scripts/eval_round6.sh first")
        return
    for m in missing:
        print(f"SKIP (not found): {m}")

    df = pd.DataFrame(rows)
    cols = ["method", "subset", "n", "acc", "g1", "g2", "g3",
            "g1_pertoken", "g2_pertoken", "g3_pertoken",
            "chosen_logp", "rejected_logp", "chosen_tokens", "rejected_tokens",
            "length_acc_gap", "worst_subset"]
    df = df[cols]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # Pretty print, ordered by worst_subset (the round-5/6 headline metric).
    order = (df[df.subset == "all"].sort_values("worst_subset", ascending=False).method.tolist())
    print("\nPer-length-subset decomposition gaps — full 2000-example decomposed test")
    print("(cl = chosen longer, rl = rejected longer; sum-logp scale)\n")
    hdr = (f"{'method':13s} {'subset':16s} {'n':>5s} {'acc':>6s} {'g1':>7s} {'g2':>6s} "
           f"{'g3':>8s} {'g3/tok':>7s} {'chosen_lp':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for m in order:
        for _, r in df[df.method == m].iterrows():
            print(f"{r.method:13s} {r.subset:16s} {r.n:5d} {r.acc:6.3f} {r.g1:7.2f} "
                  f"{r.g2:6.2f} {r.g3:8.2f} {r.g3_pertoken:7.3f} {r.chosen_logp:10.1f}")
        print("-" * len(hdr))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
