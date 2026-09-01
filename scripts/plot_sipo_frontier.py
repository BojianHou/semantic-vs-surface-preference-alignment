"""Round 4 money figure: the g1 (semantics) vs g3 (length) Pareto frontier.

Shows that no method reaches the target corner (g1 high AND g3 ~0) — SimPO's
"green" (g1~19) and DPO's "purple" (g3~0) cannot be had simultaneously with this
loss family on the summed-logp scale. Scatters every SIPO config + SimPO/DPO/base.
"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def end_gaps(base, k=20):
    fs = sorted(glob.glob(base + "/decomposed_step*.csv"),
                key=lambda p: int(p.split("step")[-1].split(".")[0]))
    if not fs:
        return None
    dfs = [pd.read_csv(f) for f in fs[-k:]]
    g1 = sum((d.chosen_logp - d.rejected_prime_logp).mean() for d in dfs) / len(dfs)
    g3 = sum((d.rejected_double_prime_logp - d.rejected_logp).mean() for d in dfs) / len(dfs)
    acc = sum((( d.chosen_logp - d.rejected_logp) > 0).mean() for d in dfs) / len(dfs)
    return g1, g3, acc


refs = {"SimPO v2": ("results/core_v2_aggressive/simpo", "#2ca02c"),
        "DPO v3": ("results/core_dpo_v3/dpo", "#1f77b4"),
        "baseline v0": ("results/sipo_search/v0/sipo", "#7f7f7f")}

fig, ax = plt.subplots(figsize=(8, 6))

# SIPO configs
xs, ys = [], []
for base in glob.glob("results/sipo_search/iter*/sipo"):
    r = end_gaps(base)
    if r:
        xs.append(r[0]); ys.append(r[1])
ax.scatter(xs, ys, c="#ff7f0e", s=40, alpha=0.7, label="SIPO configs (loop)", zorder=3)

# references
for name, (base, color) in refs.items():
    r = end_gaps(base)
    if r:
        ax.scatter([r[0]], [r[1]], c=color, s=160, marker="*", edgecolor="k",
                   linewidth=0.6, label=name, zorder=5)

# winner
w = end_gaps("results/sipo_search/iter3m/sipo")
if w:
    ax.scatter([w[0]], [w[1]], facecolor="none", edgecolor="red", s=260, linewidth=2.2,
               label="SIPO iter3m (best)", zorder=6)

# target corner
ax.axvspan(18, 24, ymin=0, ymax=1, color="green", alpha=0.05)
ax.axhspan(-2, 2, color="green", alpha=0.05)
ax.add_patch(plt.Rectangle((18, -2), 6, 4, fill=True, color="green", alpha=0.12, zorder=0))
ax.annotate("TARGET\n(g1≥18 & |g3|≤2)\n— unreached by any method",
            xy=(20.5, 0), fontsize=9, ha="center", va="center", color="darkgreen")

ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("g1 = chosen − rejected′  (semantics, sum logp)  →  larger better")
ax.set_ylabel("g3 = rejected″ − rejected  (length, sum logp)  →  0 is best")
ax.set_title("The g1⊥g3 Pareto tradeoff: SimPO-green + DPO-purple can't be had at once\n"
             "(summed-logp scale; each orange dot = one auto-research config)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
out = "results/length_split/sipo_g1_g3_frontier.png"
import os
os.makedirs("results/length_split", exist_ok=True)
fig.savefig(out, dpi=130)
print(f"Saved {out}")
print(f"winner iter3m: g1={w[0]:.2f} g3={w[1]:.2f} acc={w[2]:.3f}")
