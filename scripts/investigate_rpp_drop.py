"""Investigate WHY the len-norm SimPO `rejected'' - rejected` (sum-logp) curve drops.

Decomposes the SUM-logp gap into a length term and a per-token term, tracks the
absolute per-token (mean) log-probs over training, and uses the sum-logp run as a
control (same response lengths, normalization OFF).

  gap_sum(rpp - r) = len_rpp*mean_rpp - len_r*mean_r
                   ~= (mean_rpp - mean_r)*avg_len   [per-token term]
                      + mean*(len_rpp - len_r)      [length term]
"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAST_K = 20
RUNS = [("len-norm v2", "results/core_v2_aggressive/simpo", "#1f77b4"),
        ("sum-logp", "results/ablation_simpo_sumlogp/simpo", "#d62728")]


def files(base):
    return sorted(glob.glob(base + "/decomposed_step*.csv"),
                  key=lambda p: int(p.split("step")[-1].split(".")[0]))


def row_metrics(d):
    return {
        "len_rpp": d.rejected_double_prime_tokens.mean(),
        "len_r": d.rejected_tokens.mean(),
        "mean_rpp": d.rejected_double_prime_mean_logp.mean(),   # per-token logp
        "mean_r": d.rejected_mean_logp.mean(),
        "mean_chosen": d.chosen_mean_logp.mean(),
        "gap_sum": (d.rejected_double_prime_logp - d.rejected_logp).mean(),
        "gap_mean": (d.rejected_double_prime_mean_logp - d.rejected_mean_logp).mean(),
    }


def traj(base):
    rows = []
    for f in files(base):
        step = int(f.split("step")[-1].split(".")[0])
        rows.append({"step": step, **row_metrics(pd.read_csv(f))})
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


print("=== init -> end decomposition of the rejected'' - rejected SUM-logp gap ===\n")
trajs = {}
for name, base, _ in RUNS:
    t = traj(base)
    trajs[name] = t
    init = t.iloc[0]
    end = t.tail(LAST_K).mean(numeric_only=True)
    d_mean_rpp = end.mean_rpp - init.mean_rpp
    d_mean_r = end.mean_r - init.mean_r
    avg_len = (end.len_rpp + end.len_r) / 2
    pertoken_term = (end.gap_mean - init.gap_mean) * avg_len   # change driven per-token
    length_term = (end.gap_sum - init.gap_sum) - pertoken_term
    print(f"[{name}]")
    print(f"  gap_sum:   {init.gap_sum:+.2f} -> {end.gap_sum:+.2f}   (Δ {end.gap_sum-init.gap_sum:+.2f})")
    print(f"  gap_mean (per-token): {init.gap_mean:+.4f} -> {end.gap_mean:+.4f}   (Δ {end.gap_mean-init.gap_mean:+.4f})")
    print(f"  abs mean_logp rpp: {init.mean_rpp:+.3f} -> {end.mean_rpp:+.3f}  (Δ {d_mean_rpp:+.3f})")
    print(f"  abs mean_logp r:   {init.mean_r:+.3f} -> {end.mean_r:+.3f}  (Δ {d_mean_r:+.3f})")
    print(f"  abs mean_logp chosen: {init.mean_chosen:+.3f} -> {end.mean_chosen:+.3f}")
    print(f"  Δgap_sum decomposition:  per-token term {pertoken_term:+.2f}  + length term {length_term:+.2f}")
    print(f"  (lengths ~fixed: rpp={end.len_rpp:.1f}, r={end.len_r:.1f} tok)\n")

# ---- plot: gap_sum, gap_mean, and absolute per-token logp of rpp & r ----
smooth = 15
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for name, _, color in RUNS:
    t = trajs[name]
    axes[0].plot(t.step, t.gap_sum.rolling(smooth, min_periods=1, center=True).mean(), color=color, lw=1.8, label=name)
    axes[1].plot(t.step, t.gap_mean.rolling(smooth, min_periods=1, center=True).mean(), color=color, lw=1.8, label=name)
    axes[2].plot(t.step, t.mean_rpp.rolling(smooth, min_periods=1, center=True).mean(), color=color, lw=1.8, ls="-", label=f"{name} rej''")
    axes[2].plot(t.step, t.mean_r.rolling(smooth, min_periods=1, center=True).mean(), color=color, lw=1.4, ls=":", label=f"{name} rej")
for ax, ttl in zip(axes, ["rejected'' − rejected (SUM logp)\n[the red curve in the original]",
                          "rejected'' − rejected (PER-TOKEN logp)\n[length removed]",
                          "absolute per-token logp\nsolid=rej''  dotted=rej"]):
    ax.axhline(0, color="gray", ls="--", lw=0.7)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Why rejected'' − rejected drops: per-token suppression under length normalization "
             "(sum-logp control does the opposite)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "results/ablation_simpo_sumlogp/rpp_drop_mechanism.png"
fig.savefig(out, dpi=130)
print(f"Saved {out}")
