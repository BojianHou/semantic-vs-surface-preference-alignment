"""Master audit / re-ranking of every evaluated checkpoint (reports/AUDIT_2026-08-29.md).

Scores all methods with the CORRECTED metric (scripts/decomposition_metrics.py):
content-only gaps, bias vs sensitivity vs consistency reported separately, and
prompt-clustered confidence intervals. Includes the untrained base model, without
which no claim about what *alignment* does is grounded.

Usage:
    .venv/bin/python scripts/audit_all_methods.py                 # legacy eval set
    .venv/bin/python scripts/audit_all_methods.py --eval v2       # rebuilt v2 eval set
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomposition_metrics import paired_delta, prompt_clusters, scorecard  # noqa: E402

# label -> metrics/ csv. Canonical set = the 19 of metrics/subset_gaps.csv + the base model.
SOURCES = [
    ("BASE (untrained)", "length_split_fulltest_BASE_untrained.csv"),
    ("DPO v3", "length_split_fulltest_DPO_v3.csv"),
    ("SimPO v2", "length_split_fulltest_SimPO_v2.csv"),
    ("I4 winner (r5)", "length_split_fulltest_inv_dv3cont_lr5e6_l5e4.csv"),
    ("Dr.DPO", "round6_drdpo_fulltest.csv"),
    ("R-DPO", "round6_rdpo_fulltest.csv"),
    ("LD-DPO", "round6_lddpo_fulltest.csv"),
    ("rDPO", "round6_rdpo_robust_fulltest.csv"),
    ("SamPO", "round6_sampo_fulltest.csv"),
    ("Tie", "round6_tie_fulltest.csv"),
    ("ORPO", "round6_orpo_fulltest.csv"),
    ("LMPO", "round6_lmpo_fulltest.csv"),
    ("drdpo_i4b (r7 'best')", "round7_drdpo_i4b_fulltest.csv"),
    ("drdpo_i4", "round7_drdpo_i4_fulltest.csv"),
    ("drdpo_rdpo", "round7_drdpo_rdpo_fulltest.csv"),
    ("drdpo_strat", "round7_drdpo_strat_fulltest.csv"),
    ("drdpo_accum", "round7_drdpo_accum_fulltest.csv"),
    ("drdpo_bp05", "round7_drdpo_bp05_fulltest.csv"),
    ("drdpo_bp2", "round7_drdpo_bp2_fulltest.csv"),
    ("drdpo_bp025", "round7_drdpo_bp025_fulltest.csv"),
]
BASE = "BASE (untrained)"


def load(fn, suffix=""):
    p = os.path.join("metrics", fn.replace(".csv", f"{suffix}.csv"))
    return pd.read_csv(p).sort_values("index").reset_index(drop=True) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["legacy", "v2"], default="legacy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = "" if a.eval == "legacy" else "_v2"

    frames, rows = {}, []
    for label, fn in SOURCES:
        d = load(fn, suffix)
        if d is None:
            print(f"(skip, not evaluated on {a.eval}: {fn})")
            continue
        frames[label] = d
    if not frames:
        raise SystemExit("no evaluations found")

    if a.eval == "legacy":
        idx = next(iter(frames.values()))["index"].to_numpy()
        clusters = prompt_clusters(index=idx)
    else:
        clusters = None  # v2 has one row per unique prompt by construction

    for label, d in frames.items():
        rows.append({"method": label, **scorecard(d, clusters)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)

    print("=" * 118)
    print(f"CORRECTED SCORECARD ({a.eval} eval, content component, "
          f"{df.n_clusters.iloc[0]} independent prompts / {df.n.iloc[0]} rows)")
    print("  bias = systematic preference (want 0 for g2/g3) | sens = per-example |effect| (want 0)")
    print("  consistency = mean/sd (want LARGE for g1, ~0 for g2/g3)")
    print("=" * 118)
    show = df[["method", "acc", "g1_bias", "g1_consistency", "g2_bias", "g2_sens",
               "g2_consistency", "g3_bias", "g3_sens", "g3_consistency"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 118)
    print("ROBUSTNESS — % of examples where a surface rewrite moves the score MORE than meaning does")
    print("  (this is the question 'does it judge by meaning?' asked per example; lower is better)")
    print("=" * 118)
    rb = df[["method", "pct_syntax_beats_semantics", "pct_length_beats_semantics",
             "pct_g1_positive"]].sort_values("pct_syntax_beats_semantics")
    print(rb.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    if BASE in frames:
        print("\n" + "=" * 118)
        print("PAIRED CHANGE vs the UNTRAINED model (same examples; * = 95% CI excludes 0)")
        print("  g1 bias: want POSITIVE.  g2/g3 sensitivity: want NEGATIVE (less surface-driven)")
        print("=" * 118)
        out = []
        for label, d in frames.items():
            if label == BASE:
                continue
            pd_ = paired_delta(d, frames[BASE], clusters)
            out.append({
                "method": label,
                "Δg1 bias": f"{pd_['g1']['d_bias']:+.2f}{'*' if pd_['g1']['bias_sig'] else ' '}",
                "Δg2 sens": f"{pd_['g2']['d_sens']:+.2f}{'*' if pd_['g2']['sens_sig'] else ' '}",
                "Δg3 sens": f"{pd_['g3']['d_sens']:+.2f}{'*' if pd_['g3']['sens_sig'] else ' '}",
            })
        print(pd.DataFrame(out).to_string(index=False))

    outp = a.out or f"metrics/audit_all_methods{suffix}.csv"
    df.to_csv(outp, index=False)
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
