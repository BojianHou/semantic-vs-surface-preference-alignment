"""Counterfactual: rescore a decomposed run under length-normalized (MEAN) logp.

The frozen scorer `sipo_eval.py` computes g2 (syntax) and length_acc_gap from
SUMMED response log-probs. This script recomputes the SAME metrics on the SAME
per-example eval CSV using per-token MEAN log-probs (length normalization, i.e.
what SimPO uses) to quantify how much of each "wall" is a scoring artifact.

Analysis only — does NOT train, does NOT touch sipo_eval.py / the eval callback.
Usage: .venv/bin/python analyze_summed_vs_mean_scoring.py --run_dir results/sipo_search/iter3m/sipo
"""
import argparse
import glob

import pandas as pd


def split_gap(d, chosen_col, rej_col):
    cl = d[d.chosen_tokens > d.rejected_tokens]
    rl = d[d.chosen_tokens < d.rejected_tokens]
    acc = float((d[chosen_col] - d[rej_col] > 0).mean())
    acc_cl = float(((cl[chosen_col] - cl[rej_col]) > 0).mean())
    acc_rl = float(((rl[chosen_col] - rl[rej_col]) > 0).mean())
    return acc, acc_cl, acc_rl, abs(acc_cl - acc_rl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="results/sipo_search/iter3m/sipo")
    args = p.parse_args()

    f = sorted(glob.glob(f"{args.run_dir}/decomposed_step*.csv"),
               key=lambda q: int(q.split("step")[-1].split(".")[0]))[-1]
    d = pd.read_csv(f)
    print(f"file={f}  n={len(d)}")

    same_tok = (d.rejected_prime_tokens == d.rejected_double_prime_tokens).mean()
    print(f"\n[rp vs rpp token-length identical]: {same_tok:.3f} of examples "
          f"(median rp={d.rejected_prime_tokens.median():.0f} rpp={d.rejected_double_prime_tokens.median():.0f})")

    g2_sum = (d.rejected_prime_logp - d.rejected_double_prime_logp).mean()
    g2_mean = (d.rejected_prime_mean_logp - d.rejected_double_prime_mean_logp).mean()
    print(f"\nG2 syntax (rp-rpp):  SUMMED {g2_sum:+.3f} (|g2|={abs(g2_sum):.2f})  "
          f"MEAN {g2_mean:+.3f} (|g2|={abs(g2_mean):.2f})")

    acc_s, cl_s, rl_s, gap_s = split_gap(d, "chosen_logp", "rejected_logp")
    acc_m, cl_m, rl_m, gap_m = split_gap(d, "chosen_mean_logp", "rejected_mean_logp")
    print(f"\nSUMMED (scorer): acc={acc_s:.3f} acc_cl={cl_s:.3f} acc_rl={rl_s:.3f} gap={gap_s:.3f}")
    print(f"MEAN (len-norm): acc={acc_m:.3f} acc_cl={cl_m:.3f} acc_rl={rl_m:.3f} gap={gap_m:.3f}")

    print("\nUnder length-normalized scoring the SAME checkpoint would give:")
    print(f"  C2 |g2|<=2 : {'PASS' if abs(g2_mean) <= 2 else 'fail'} ({abs(g2_mean):.2f})")
    print(f"  C9 |g2|<6.8: {'PASS' if abs(g2_mean) < 6.8 else 'fail'} ({abs(g2_mean):.2f})")
    print(f"  C5 gap<=0.2: {'PASS' if gap_m <= 0.2 else 'fail'} ({gap_m:.3f})")
    print(f"  C4 acc>=0.60: {'PASS' if acc_m >= 0.60 else 'fail'} ({acc_m:.3f})")


if __name__ == "__main__":
    main()
