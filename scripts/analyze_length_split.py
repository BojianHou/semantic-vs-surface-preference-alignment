"""Task 3: split the eval set by length direction and recompute ALL metrics.

Subset A ("chosen longer"):   chosen_tokens >  rejected_tokens
Subset B ("rejected longer"): chosen_tokens <  rejected_tokens   (ties dropped)

Response lengths are fixed across steps, so subset membership is constant; we can
build per-subset trajectories of every metric. We report:
  - a final-step table (avg over last K evals) of all 6 metrics x {A,B} per run
  - per-run 6-panel trajectory plots, one line per subset

Metrics (same as the cross-method panels):
  sum_C_R, mean_C_R, acc_sum, C_Rp (chosen-rejected'),
  Rp_Rpp (rejected'-rejected''), Rpp_R (rejected''-rejected)
"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAST_K = 20
RUNS = [
    ("len-norm v2", "results/core_v2_aggressive/simpo"),
    ("sum-logp", "results/ablation_simpo_sumlogp/simpo"),
]
PANELS = [
    ("sum_C_R", "chosen − rejected (sum logp)", None),
    ("mean_C_R", "chosen − rejected (mean logp) [objective]", None),
    ("acc_sum", "accuracy: P(chosen > rejected)", (0, 1)),
    ("C_Rp", "chosen − rejected′ (sum logp)", None),
    ("Rp_Rpp", "rejected′ − rejected″ (sum logp)", None),
    ("Rpp_R", "rejected″ − rejected (sum logp)", None),
]
SUBSETS = [
    ("chosen_longer", "#2ca02c"),   # chosen_tokens > rejected_tokens
    ("rejected_longer", "#9467bd"),  # chosen_tokens < rejected_tokens
]


def metrics(d):
    return {
        "sum_C_R": (d.chosen_logp - d.rejected_logp).mean(),
        "mean_C_R": (d.chosen_mean_logp - d.rejected_mean_logp).mean(),
        "acc_sum": ((d.chosen_logp - d.rejected_logp) > 0).mean(),
        "C_Rp": (d.chosen_logp - d.rejected_prime_logp).mean(),
        "Rp_Rpp": (d.rejected_prime_logp - d.rejected_double_prime_logp).mean(),
        "Rpp_R": (d.rejected_double_prime_logp - d.rejected_logp).mean(),
    }


def split(d):
    a = d[d.chosen_tokens > d.rejected_tokens]
    b = d[d.chosen_tokens < d.rejected_tokens]
    return {"chosen_longer": a, "rejected_longer": b}


def trajectory(base):
    files = sorted(glob.glob(base + "/decomposed_step*.csv"),
                   key=lambda p: int(p.split("step")[-1].split(".")[0]))
    out = {s: [] for s, _ in SUBSETS}
    for f in files:
        step = int(f.split("step")[-1].split(".")[0])
        sub = split(pd.read_csv(f))
        for s, _ in SUBSETS:
            row = {"step": step, **metrics(sub[s])}
            out[s].append(row)
    return {s: pd.DataFrame(rows).sort_values("step").reset_index(drop=True) for s, rows in out.items()}


def final_counts(base):
    last = pd.read_csv(sorted(glob.glob(base + "/decomposed_step*.csv"),
                              key=lambda p: int(p.split("step")[-1].split(".")[0]))[-1])
    sub = split(last)
    return {s: len(sub[s]) for s in sub}, len(last)


# ---- final-step table (avg over last K) ----
records = []
trajs = {}
for name, base in RUNS:
    tr = trajectory(base)
    trajs[name] = tr
    counts, n = final_counts(base)
    for s, _ in SUBSETS:
        tail = tr[s].tail(LAST_K).mean(numeric_only=True)
        rec = {"run": name, "subset": s, "n": counts[s]}
        rec.update({k: round(float(tail[k]), 3) for k, _, _ in PANELS})
        records.append(rec)

table = pd.DataFrame(records)
pd.set_option("display.width", 220)
print("Task 3 — all metrics split by length direction (avg over last %d evals)\n" % LAST_K)
print(table.to_string(index=False))
table.to_csv("length_split_metrics.csv", index=False)
print("\nwrote length_split_metrics.csv")

# ---- per-run 6-panel trajectory plots ----
smooth = 15
for name, base in RUNS:
    tr = trajs[name]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, ylim) in zip(axes.flat, PANELS):
        for s, color in SUBSETS:
            t = tr[s]
            ax.plot(t.step, t[col], color=color, alpha=0.18, linewidth=0.7)
            ax.plot(t.step, t[col].rolling(smooth, min_periods=1, center=True).mean(),
                    color=color, linewidth=1.8, label=s)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if col == "acc_sum":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.legend(fontsize=8)
    fig.suptitle(f"Length-split metrics — SimPO {name}\n"
                 f"green: chosen_tokens>rejected_tokens · purple: chosen<rejected · n=256 · rolling({smooth})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    tag = name.replace(" ", "_").replace("-", "")
    out = f"results/ablation_simpo_sumlogp/length_split_{tag}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
