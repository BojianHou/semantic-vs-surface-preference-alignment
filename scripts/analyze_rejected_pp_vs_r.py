"""Task 2: why is rejected'' not preferred over rejected?

Hypothesis: rejected'' is LONGER than rejected, and (length-normalized) SimPO
prefers shorter responses -> it assigns rejected'' a lower score, making the
`rejected'' - rejected` gap negative.

We test this directly on the per-example eval CSVs (response lengths are fixed
across steps, so we read the final eval; averaged over the last K for stability):

  - lengths: mean tokens of rejected'' vs rejected, and how often rpp is longer
  - the gap itself, in SUM logp (the decomposition plot's units) and in
    MEAN/per-token logp (SimPO's reward units)
  - corr(length-diff, gap): negative => longer rpp -> lower score (length penalty)
  - gap conditioned on rpp-longer vs rpp-shorter subsets

Compared across: init (earliest eval), len-norm SimPO v2, sum-logp SimPO.
"""
import glob
import math

import numpy as np
import pandas as pd

LAST_K = 20

RUNS = [
    ("init (step0)", "results/core_v2_aggressive/simpo", "first"),
    ("len-norm v2", "results/core_v2_aggressive/simpo", "last"),
    ("sum-logp", "results/ablation_simpo_sumlogp/simpo", "last"),
]


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def files_for(base, which):
    fs = sorted(glob.glob(base + "/decomposed_step*.csv"),
                key=lambda p: int(p.split("step")[-1].split(".")[0]))
    return fs[:1] if which == "first" else fs[-LAST_K:]


def analyze(base, which):
    accum = {}
    dfs = [pd.read_csv(f) for f in files_for(base, which)]
    rows = []
    for d in dfs:
        len_diff = d.rejected_double_prime_tokens - d.rejected_tokens  # rpp - r
        gap_sum = d.rejected_double_prime_logp - d.rejected_logp        # sum-logp gap
        gap_mean = d.rejected_double_prime_mean_logp - d.rejected_mean_logp  # per-token gap
        longer = len_diff > 0
        rows.append({
            "mean_tok_rpp": d.rejected_double_prime_tokens.mean(),
            "mean_tok_r": d.rejected_tokens.mean(),
            "mean_len_diff(rpp-r)": len_diff.mean(),
            "pct_rpp_longer": 100 * longer.mean(),
            "gap_sum(rpp-r)": gap_sum.mean(),
            "gap_meanlogp(rpp-r)": gap_mean.mean(),
            "pct_rpp_preferred_sum": 100 * (gap_sum > 0).mean(),
            "corr(len_diff, gap_sum)": corr(len_diff, gap_sum),
            "corr(len_diff, gap_mean)": corr(len_diff, gap_mean),
            "gap_sum | rpp_longer": gap_sum[longer].mean(),
            "gap_sum | rpp_shorter": gap_sum[~longer].mean(),
        })
    return pd.DataFrame(rows).mean()


out = pd.DataFrame({name: analyze(base, which) for name, base, which in RUNS}).round(3)
pd.set_option("display.width", 200)
print("Task 2 — rejected'' vs rejected (avg over last %d evals; init = first eval)\n" % LAST_K)
print(out.to_string())
out.to_csv("rejected_pp_vs_r_summary.csv")
print("\nwrote rejected_pp_vs_r_summary.csv")
print(
    "\nRead:\n"
    "  pct_rpp_longer > 50           => rejected'' is typically longer than rejected\n"
    "  gap_sum(rpp-r) < 0            => model scores rejected'' BELOW rejected (not preferred)\n"
    "  corr(len_diff, gap*) < 0      => the longer rpp is, the lower its score => length penalty\n"
    "  gap_sum|rpp_longer << gap_sum|rpp_shorter => penalty concentrated on the longer cases"
)
