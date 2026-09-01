"""Round 3: does the TRAINING-data length direction flip the brevity heuristic?

Compares three models on the SAME decomposed val set (n=256), each sliced by
eval-set length direction (chosen_tokens vs rejected_tokens):

  chosen_longer arm   : trained only on pairs where chosen is the longer response
  rejected_longer arm : trained only on pairs where rejected is the longer response
  mixed baseline      : the existing full-92k SimPO run (results/core_v2_aggressive/simpo)

Round-2 (analyze_length_split.py) showed the mixed baseline ranks by brevity:
acc ~0.21 when chosen is longer vs ~0.90 when chosen is shorter. The question
here: if we train ONLY on chosen-longer pairs, does the model learn to prefer
longer (raising chosen-longer-subset accuracy) — i.e. is the heuristic a
data artifact — or does brevity-ranking persist regardless (objective bias)?

Reuses the metric/split definitions from analyze_length_split.py.
"""
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAST_K = 20
RUNS = [
    ("chosen_longer arm", "results/length_arm_chosen_longer/simpo", "#2ca02c"),
    ("rejected_longer arm", "results/length_arm_rejected_longer/simpo", "#9467bd"),
    ("mixed baseline", "results/core_v2_aggressive/simpo", "#7f7f7f"),
]
PANELS = [
    ("sum_C_R", "chosen − rejected (sum logp)", None),
    ("mean_C_R", "chosen − rejected (mean logp) [objective]", None),
    ("acc_sum", "accuracy: P(chosen > rejected)", (0, 1)),
    ("C_Rp", "chosen − rejected′ (sum logp)", None),
    ("Rp_Rpp", "rejected′ − rejected″ (sum logp)", None),
    ("Rpp_R", "rejected″ − rejected (sum logp)", None),
]
SUBSETS = ["chosen_longer", "rejected_longer", "all"]


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
    return {
        "chosen_longer": d[d.chosen_tokens > d.rejected_tokens],
        "rejected_longer": d[d.chosen_tokens < d.rejected_tokens],
        "all": d,
    }


def step_files(base):
    return sorted(glob.glob(base + "/decomposed_step*.csv"),
                  key=lambda p: int(p.split("step")[-1].split(".")[0]))


def trajectory(base):
    out = {s: [] for s in SUBSETS}
    for f in step_files(base):
        step = int(f.split("step")[-1].split(".")[0])
        sub = split(pd.read_csv(f))
        for s in SUBSETS:
            out[s].append({"step": step, **metrics(sub[s])})
    return {s: pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
            for s, rows in out.items()}


def main():
    records, trajs = [], {}
    for name, base, _ in RUNS:
        files = step_files(base)
        if not files:
            print(f"[skip] no CSVs yet for {name} ({base})")
            continue
        tr = trajectory(base)
        trajs[name] = tr
        last = pd.read_csv(files[-1])
        counts = {s: len(split(last)[s]) for s in SUBSETS}
        for s in SUBSETS:
            tail = tr[s].tail(LAST_K).mean(numeric_only=True)
            rec = {"run": name, "subset": s, "n": counts[s]}
            rec.update({k: round(float(tail[k]), 3) for k, _, _ in PANELS})
            records.append(rec)

    if not records:
        print("No results to analyze yet.")
        return

    table = pd.DataFrame(records)
    pd.set_option("display.width", 220)
    print("Round 3 — length-direction accuracy by TRAINING arm (avg over last %d evals)\n" % LAST_K)
    print(table.to_string(index=False))
    table.to_csv("arms_length_split_metrics.csv", index=False)
    print("\nwrote arms_length_split_metrics.csv")

    # Headline: acc(chosen_longer subset) vs acc(rejected_longer subset) per arm.
    print("\n=== headline: does training arm flip the brevity heuristic? ===")
    for name, _, _ in RUNS:
        rows = {r["subset"]: r for r in records if r["run"] == name}
        if "chosen_longer" in rows and "rejected_longer" in rows:
            print(f"  {name:22s}  acc[chosen_longer]={rows['chosen_longer']['acc_sum']:.3f}  "
                  f"acc[rejected_longer]={rows['rejected_longer']['acc_sum']:.3f}  "
                  f"acc[all]={rows['all']['acc_sum']:.3f}")

    # Comparison plot: accuracy trajectory per arm, one line per eval-length subset.
    smooth = 15
    present = [(n, b, c) for n, b, c in RUNS if n in trajs]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5), squeeze=False)
    for ax, (name, _, _) in zip(axes[0], present):
        tr = trajs[name]
        for s, color in [("chosen_longer", "#2ca02c"), ("rejected_longer", "#9467bd"), ("all", "#333333")]:
            t = tr[s]
            ax.plot(t.step, t["acc_sum"], color=color, alpha=0.18, linewidth=0.7)
            ax.plot(t.step, t["acc_sum"].rolling(smooth, min_periods=1, center=True).mean(),
                    color=color, linewidth=1.8, label=s)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_title(f"trained on: {name}", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("P(chosen > rejected)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="eval subset")
    fig.suptitle("Round 3 — eval accuracy split by response-length direction, per training arm\n"
                 "green: chosen longer · purple: rejected longer · black: all · n=256",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = "results/length_split/arms_accuracy_by_length.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
