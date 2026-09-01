"""Round-8 scorecard: Support-Augmented KL-Dr.DPO vs the cf-off controls.

Reads every run under results/round8_cf_kl_dro/<label>/<method>/ and reports, at
the last eval step:

    acc            P(chosen > rejected)                     — the classic metric
    acc_rpp        P(chosen > rejected_double_prime)        — length-matched pair
    acc_rp         P(chosen > rejected_prime)               — length+syntax matched
    worst_orbit    P(chosen beats ALL THREE)                — what the inner DRO targets
    acc_cl/acc_rl  P(chosen > rejected) split by length direction, and their gap
    g1/g2/g3       the decomposition gaps (see sipo_eval.py)

Usage:  .venv/bin/python scripts/summarize_round8.py [--root results/round8_cf_kl_dro]
"""
import argparse
import glob
import os

import pandas as pd

G1 = "mean_logp_chosen_minus_rejected_prime"
G2 = "mean_logp_rejected_prime_minus_rejected_double_prime"
G3 = "mean_logp_rejected_double_prime_minus_rejected"


def last_step_csv(run_dir):
    steps = sorted(glob.glob(f"{run_dir}/decomposed_step*.csv"),
                   key=lambda q: int(q.split("step")[-1].split(".")[0]))
    return steps[-1] if steps else None


def subset_gaps(run_dir):
    """Per-length-subset g1/g2/g3 + accuracies (REPORT §9.10 finding 3: the pooled
    gaps are NOT interpretable — the two subsets carry opposite-signed g3 that
    cancels on averaging). Subsets match scripts/subset_gaps.py; ties excluded."""
    path = last_step_csv(run_dir)
    if path is None:
        return []
    d = pd.read_csv(path)
    out = []
    for name, f in (("all", d),
                    ("chosen_longer", d[d.chosen_tokens > d.rejected_tokens]),
                    ("rejected_longer", d[d.chosen_tokens < d.rejected_tokens])):
        b_r = f.chosen_logp > f.rejected_logp
        b_pp = f.chosen_logp > f.rejected_double_prime_logp
        b_p = f.chosen_logp > f.rejected_prime_logp
        out.append(dict(
            subset=name, n=len(f),
            acc=float(b_r.mean()), acc_rpp=float(b_pp.mean()), acc_rp=float(b_p.mean()),
            worst_orbit=float((b_r & b_pp & b_p).mean()),
            g1=float((f.chosen_logp - f.rejected_prime_logp).mean()),
            g2=float((f.rejected_prime_logp - f.rejected_double_prime_logp).mean()),
            g3=float((f.rejected_double_prime_logp - f.rejected_logp).mean()),
        ))
    return out


def score(run_dir):
    path = last_step_csv(run_dir)
    if path is None:
        return None
    d = pd.read_csv(path)
    beats_r = d.chosen_logp > d.rejected_logp
    beats_rpp = d.chosen_logp > d.rejected_double_prime_logp
    beats_rp = d.chosen_logp > d.rejected_prime_logp
    cl = d[d.chosen_tokens > d.rejected_tokens]
    rl = d[d.chosen_tokens < d.rejected_tokens]
    acc_cl = float(((cl.chosen_logp - cl.rejected_logp) > 0).mean()) if len(cl) else float("nan")
    acc_rl = float(((rl.chosen_logp - rl.rejected_logp) > 0).mean()) if len(rl) else float("nan")
    row = {
        "step": int(path.split("step")[-1].split(".")[0]),
        "n": len(d),
        "acc": float(beats_r.mean()),
        "acc_rpp": float(beats_rpp.mean()),
        "acc_rp": float(beats_rp.mean()),
        "worst_orbit": float((beats_r & beats_rpp & beats_rp).mean()),
        "acc_cl": acc_cl,
        "acc_rl": acc_rl,
        "len_gap": abs(acc_cl - acc_rl),
    }
    summ = os.path.join(run_dir, "decomposed_summary.csv")
    if os.path.exists(summ):
        tail = pd.read_csv(summ).tail(1).iloc[0]
        row.update(g1=float(tail[G1]), g2=float(tail[G2]), g3=float(tail[G3]))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/round8_cf_kl_dro")
    p.add_argument("--out", default="metrics/round8_summary.csv")
    args = p.parse_args()

    rows, subs = [], []
    for label in sorted(os.listdir(args.root)):
        for method_dir in sorted(glob.glob(os.path.join(args.root, label, "*"))):
            if not os.path.isdir(method_dir):
                continue
            r = score(method_dir)
            if r:
                rows.append({"label": label, "method": os.path.basename(method_dir), **r})
            subs += [{"label": label, **g} for g in subset_gaps(method_dir)]
    df = pd.DataFrame(rows)
    cols = ["label", "method", "step", "acc", "acc_rpp", "acc_rp", "worst_orbit",
            "acc_cl", "acc_rl", "len_gap", "g1", "g2", "g3"]
    df = df[[c for c in cols if c in df.columns]]
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")

    sdf = pd.DataFrame(subs)
    for sub in ("chosen_longer", "rejected_longer"):
        print(f"\n=== {sub} ===")
        print(sdf[sdf.subset == sub].drop(columns="subset").to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    sub_out = args.out.replace(".csv", "_by_subset.csv")
    sdf.to_csv(sub_out, index=False)
    print(f"\n-> {sub_out}")


if __name__ == "__main__":
    main()
