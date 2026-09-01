"""Why does Dr.DPO win the worst-subset metric? (REPORT §9.9 finding 1 follow-up)

Round 6 crowned Dr.DPO on worst_subset = min(acc_chosen_longer, acc_rejected_longer)
but never asked *how* it gets there. This script runs three cheap, falsifiable
checks on the saved per-example full-test CSVs:

  A. Paired significance. Dr.DPO vs DPO v3 on the SAME 2000 examples, per length
     subset (exact McNemar on discordant pairs). Tests whether the +0.007 on the
     hard subset is real or noise.
  B. Gap accounting. Splits the change in worst_subset / length_acc_gap into the
     part won on the hard subset vs the part *given up* on the easy subset — a
     method can improve `worst` either by lifting the floor or lowering the ceiling.
  C. Cross-method structure. Over all evaluated methods, which decomposition gap
     predicts chosen-longer accuracy: g1 (semantics) or g3 (length)?

Usage (from project root, after scripts/subset_gaps.py):
    .venv/bin/python scripts/analyze_drdpo.py
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

# --- small numpy-only stats helpers -------------------------------------------
# scipy is not in .venv and installing into it mid-run (training jobs are live in
# this same env) is not worth the risk; these are exact / permutation-based.

_RNG = np.random.default_rng(42)


def binom_two_sided(k, n):
    """Exact two-sided binomial p-value against p=0.5 (symmetric, so 2x the tail)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def pearsonr(x, y, n_perm=20000):
    """Pearson r with a permutation p-value (no t-distribution needed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = float(np.corrcoef(x, y)[0, 1])
    perm = np.array([np.corrcoef(_RNG.permutation(x), y)[0, 1] for _ in range(n_perm)])
    p = float((np.abs(perm) >= abs(r)).mean())
    return r, p


def spearmanr(x, y):
    """Spearman rho = Pearson on ranks (average ranks for ties)."""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])

CSV = {
    "drdpo": "metrics/round6_drdpo_fulltest.csv",
    "DPO_v3": "metrics/length_split_fulltest_DPO_v3.csv",
    "I4_winner": "metrics/length_split_fulltest_inv_dv3cont_lr5e6_l5e4.csv",
}


def load_correct(path):
    """Per-example correctness (chosen ranked above rejected) + length direction."""
    df = pd.read_csv(path)
    return pd.DataFrame({
        "index": df["index"],
        "correct": (df.chosen_logp - df.rejected_logp) > 0,
        "cl": df.chosen_tokens > df.rejected_tokens,
        "rl": df.chosen_tokens < df.rejected_tokens,
    })


def mcnemar(a, b):
    """Exact McNemar on paired boolean correctness vectors. Returns (n01, n10, p)."""
    n01 = int((~a & b).sum())   # a wrong, b right
    n10 = int((a & ~b).sum())   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = binom_two_sided(n10, n)
    return n01, n10, p


def part_a():
    print("=" * 78)
    print("A. Is Dr.DPO's edge over DPO v3 statistically real? (exact McNemar, paired)")
    print("=" * 78)
    base = load_correct(CSV["DPO_v3"]).set_index("index")
    for label in ("drdpo", "I4_winner"):
        cur = load_correct(CSV[label]).set_index("index")
        joined = base.join(cur, lsuffix="_base", rsuffix="_cur")
        print(f"\n{label} vs DPO_v3")
        for subset, mask in (("chosen_longer", joined.cl_base),
                             ("rejected_longer", joined.rl_base),
                             ("all", pd.Series(True, index=joined.index))):
            s = joined[mask]
            a, b = s.correct_base.values, s.correct_cur.values
            n01, n10, p = mcnemar(a, b)
            d = b.mean() - a.mean()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {subset:16s} n={len(s):4d}  {a.mean():.3f} -> {b.mean():.3f} "
                  f"(delta {d:+.3f})  flips +{n01}/-{n10}  p={p:.3f} {sig}")


def part_b(gaps):
    print("\n" + "=" * 78)
    print("B. Where does the smaller length gap come from — floor lifted or ceiling cut?")
    print("=" * 78)
    piv = gaps[gaps.subset != "all"].pivot(index="method", columns="subset", values="acc")
    ref = piv.loc["DPO_v3"]
    print(f"\n{'method':13s} {'acc_cl':>7s} {'acc_rl':>7s} {'d_cl':>7s} {'d_rl':>7s} "
          f"{'gap':>6s} {'worst':>6s}  {'gap reduction from':s}")
    print("-" * 92)
    rows = []
    for m in piv.index:
        cl, rl = piv.loc[m, "chosen_longer"], piv.loc[m, "rejected_longer"]
        d_cl, d_rl = cl - ref.chosen_longer, rl - ref.rejected_longer
        gap, worst = abs(cl - rl), min(cl, rl)
        ref_gap = abs(ref.chosen_longer - ref.rejected_longer)
        dgap = ref_gap - gap
        if abs(dgap) < 1e-9:
            src = "—"
        else:
            # How much of the gap reduction is lifting the hard subset vs lowering the easy one.
            lift = max(d_cl, 0.0)
            cut = max(-d_rl, 0.0)
            tot = lift + cut
            src = (f"floor +{lift:.3f} / ceiling -{cut:.3f}  "
                   f"({100*lift/tot:.0f}% floor)" if tot > 0 else "neither")
        rows.append((m, cl, rl, d_cl, d_rl, gap, worst, src))
    for m, cl, rl, d_cl, d_rl, gap, worst, src in sorted(rows, key=lambda r: -r[6]):
        print(f"{m:13s} {cl:7.3f} {rl:7.3f} {d_cl:+7.3f} {d_rl:+7.3f} {gap:6.3f} "
              f"{worst:6.3f}  {src}")


def part_c(gaps):
    print("\n" + "=" * 78)
    print("C. Across methods, what predicts chosen-longer (hard-subset) accuracy?")
    print("=" * 78)
    cl = gaps[gaps.subset == "chosen_longer"].set_index("method")
    allg = gaps[gaps.subset == "all"].set_index("method")
    targets = {"acc_chosen_longer": cl.acc, "worst_subset": allg.worst_subset}
    preds = {
        "g1[cl] semantics": cl.g1,
        "g3[cl] length": cl.g3,
        "|g3|[cl]": cl.g3.abs(),
        "g3[all]": allg.g3,
        "|g3|[all]": allg.g3.abs(),
        "chosen_logp[cl] (displacement)": cl.chosen_logp,
    }
    for tname, tv in targets.items():
        print(f"\n  target = {tname}  (n={len(tv)} methods)")
        for pname, pv in preds.items():
            r, p = pearsonr(pv.loc[tv.index].values, tv.values)
            rho = spearmanr(pv.loc[tv.index].values, tv.values)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {pname:32s} pearson r={r:+.3f} (p={p:.4f}){star:4s} spearman={rho:+.3f}")

    # The DPO-family subset (exclude the two objective-level outliers) is the
    # honest comparison: ORPO/LMPO/SimPO use different reward scales entirely.
    fam = [m for m in cl.index if m not in ("orpo", "lmpo", "SimPO_v2")]
    print(f"\n  DPO-family only ({', '.join(fam)}):")
    for pname, pv in (("g1[cl]", cl.g1), ("g3[cl]", cl.g3)):
        r, p = pearsonr(pv.loc[fam].values, cl.acc.loc[fam].values)
        print(f"    acc_cl ~ {pname:10s} pearson r={r:+.3f} (p={p:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gaps", default="metrics/subset_gaps.csv")
    args = p.parse_args()
    if not os.path.exists(args.gaps):
        print(f"missing {args.gaps} — run scripts/subset_gaps.py first")
        return
    gaps = pd.read_csv(args.gaps)
    part_a()
    part_b(gaps)
    part_c(gaps)


if __name__ == "__main__":
    main()
