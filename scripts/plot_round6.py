"""Plot Round-6 length-robust methods vs DPO v3 / SimPO v2 anchors.

Left: worst-subset accuracy (higher = more length-robust decisions).
Right: |g3| raw-logp length gap (lower = more length-neutral log-probs).
Shows the worst-subset ⊥ |g3| dissociation (e.g. LMPO: tiny |g3|, worst worst-subset).

Usage: .venv/bin/python scripts/plot_round6.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("metrics/round6_summary.csv").set_index("method")
# anchors
anchors = {"DPO v3": (0.607, 1.73, 0.110), "SimPO v2": (0.460, 9.12, 0.354)}

order = df.sort_values("worst_subset", ascending=False).index.tolist()
worst = [df.loc[m, "worst_subset"] for m in order]
absg3 = [abs(df.loc[m, "g3"]) for m in order]
labels = list(order)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#2c7fb8"] * len(order)

ax[0].barh(labels, worst, color=colors)
for name, (w, _, _) in anchors.items():
    ax[0].axvline(w, ls="--", lw=1.2, color="crimson" if "SimPO" in name else "green")
    ax[0].text(w, len(order) - 0.5, name, rotation=90, va="top", fontsize=8,
               color="crimson" if "SimPO" in name else "green")
ax[0].set_xlabel("worst-subset accuracy  min(acc_chosen_longer, acc_rejected_longer)")
ax[0].set_title("Length-robust DECISIONS (higher better)")
ax[0].invert_yaxis()

ax[1].barh(labels, absg3, color="#d95f0e")
for name, (_, g, _) in anchors.items():
    ax[1].axvline(g, ls="--", lw=1.2, color="crimson" if "SimPO" in name else "green")
    ax[1].text(g, len(order) - 0.5, name, rotation=90, va="top", fontsize=8,
               color="crimson" if "SimPO" in name else "green")
ax[1].set_xlabel("|g3| = |chosen-length paraphrase − rejected| (sum logp)")
ax[1].set_title("Length-neutral LOG-PROBS (lower better)")
ax[1].invert_yaxis()

fig.suptitle("Round 6 — off-the-shelf length-robust preference methods (Qwen2.5-1.5B, TL;DR, full 2000 test)")
fig.tight_layout()
out = "results/round6_length_robust/round6_comparison.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
