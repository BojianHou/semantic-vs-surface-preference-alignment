"""
Length-bias analysis: summed-logp SimPO vs length-normalized SimPO (v2).

SimPO's reward is beta * logp(y|x). With length normalization (mean log-prob,
v2) the per-token average is used; with summed log-prob the raw sum is used.
Because logp is accumulated negatively over tokens, the summed variant
mechanically penalizes longer sequences. This script quantifies how each
variant's preference margin relates to response length.

For each run we use the final-step per-example eval CSV (n=256) and also
average key stats over the last K steps for stability.

Outputs a small table to stdout and writes length_bias_summary.csv.
"""
import glob
import math
import numpy as np
import pandas as pd


def pearsonr(x, y):
    """Pearson r and a two-sided p-value (t-approx), no scipy dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    # two-sided p via normal approx on Fisher z (fine for n=256)
    r_clip = min(max(r, -0.999999), 0.999999)
    z = np.arctanh(r_clip) * np.sqrt(n - 3)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return r, float(p)

RUNS = {
    "simpo_lennorm_v2": "results/core_v2_aggressive/simpo",
    "simpo_sumlogp": "results/ablation_simpo_sumlogp/simpo",
}
LAST_K = 20  # average per-example metrics over last K eval steps for stability


def load_step(path):
    df = pd.read_csv(path)
    # length delta: how much longer chosen is vs rejected (true rejected)
    df["len_delta"] = df["chosen_tokens"] - df["rejected_tokens"]
    df["chosen_longer"] = df["len_delta"] > 0
    # margin in SUMMED-logp space (raw SimPO sum-logp reward space)
    df["margin_sum"] = df["logp_chosen_minus_rejected"]
    # margin in MEAN-logp space (length-normalized SimPO reward space, v2)
    df["margin_mean"] = df["chosen_mean_logp"] - df["rejected_mean_logp"]
    return df


def analyze_run(name, d):
    files = sorted(glob.glob(d + "/decomposed_step*.csv"))
    last = load_step(files[-1])
    cl = last[last["chosen_longer"]]
    cs = last[~last["chosen_longer"]]

    out = {"run": name}
    for space, col in [("sum", "margin_sum"), ("mean", "margin_mean")]:
        r, p = pearsonr(last["len_delta"], last[col])
        acc_all = float((last[col] > 0).mean())
        acc_long = float((cl[col] > 0).mean()) if len(cl) else np.nan
        acc_short = float((cs[col] > 0).mean()) if len(cs) else np.nan
        # last-K averaged correlation for stability
        rs = [
            pearsonr(s["len_delta"], s[col])[0]
            for f in files[-LAST_K:]
            for s in [load_step(f)]
            if s["len_delta"].std() > 0 and s[col].std() > 0
        ]
        out[f"acc[{space}]"] = round(acc_all, 4)
        out[f"corr[{space}]"] = round(r, 4)
        out[f"corr[{space}]_lastK"] = round(float(np.mean(rs)), 4) if rs else np.nan
        out[f"acc_longer[{space}]"] = round(acc_long, 4)
        out[f"acc_shorter[{space}]"] = round(acc_short, 4)
        out[f"acc_gap[{space}]"] = round(acc_long - acc_short, 4)

    out["n_chosen_longer"] = int(len(cl))
    out["n_chosen_shorter"] = int(len(cs))
    out["mean_chosen_tokens"] = round(float(last["chosen_tokens"].mean()), 1)
    out["mean_rejected_tokens"] = round(float(last["rejected_tokens"].mean()), 1)
    return out


rows = [analyze_run(n, d) for n, d in RUNS.items()]
out = pd.DataFrame(rows)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
# transpose for readability: metrics as rows, runs as columns
disp = out.set_index("run").T
print(disp.to_string())
out.to_csv("length_bias_summary.csv", index=False)
print("\nwrote length_bias_summary.csv")

print(
    "\nLegend: [sum]=summed-logp reward space, [mean]=length-normalized (per-token) reward space.\n"
    "  corr > 0          => model prefers the LONGER response (length bias)\n"
    "  corr ~ 0          => length-neutral\n"
    "  corr < 0          => model prefers the SHORTER response (anti-length / over-correction)\n"
    "  acc_gap > 0       => higher pairwise accuracy when the GOOD answer is longer\n"
    "Each model is most fairly read in ITS OWN training space: v2 -> [mean], sumlogp -> [sum]."
)
