"""Plot the effect of SimPO length-normalization on decomposed-logp trajectories.

Same 6-panel cross-comparison layout as plot_trends.py's cross-method figure,
but instead of SimPO-vs-DPO it overlays two SimPO runs that differ ONLY in the
reward normalization:

  * len-norm  (v2, results/core_v2_aggressive/simpo)        -- average log-prob  [SimPO default]
  * sum-logp  (results/ablation_simpo_sumlogp/simpo)      -- summed  log-prob  [--no-simpo_average_log_prob]

This isolates how length normalization reshapes the chosen-vs-rejected margin
and each decomposition component (chosen-rejected', rejected'-rejected'', etc.).

Usage:
    python plot_normalization_effect.py
"""

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# (display label, csv dir, color)
RUNS = [
    ("len-norm (v2)", "results/core_v2_aggressive/simpo", "#1f77b4"),
    ("sum-logp", "results/ablation_simpo_sumlogp/simpo", "#d62728"),
]

PANELS = [
    ("sum_C_R", "chosen − rejected (sum logp)", None),
    ("mean_C_R", "chosen − rejected (mean logp) [SimPO objective]", None),
    ("acc_sum", "accuracy: P(chosen > rejected)", (0, 1)),
    ("C_Rp", "chosen − rejected′ (sum logp)", None),
    ("Rp_Rpp", "rejected′ − rejected″ (sum logp)", None),
    ("Rpp_R", "rejected″ − rejected (sum logp)", None),
]


def load_trajectory(base):
    files = sorted(
        glob.glob(f"{base}/decomposed_step*.csv"),
        key=lambda p: int(p.split("step")[-1].split(".")[0]),
    )
    rows = []
    for f in files:
        step = int(f.split("step")[-1].split(".")[0])
        d = pd.read_csv(f)
        rows.append(
            {
                "step": step,
                "sum_C_R": (d.chosen_logp - d.rejected_logp).mean(),
                "mean_C_R": (d.chosen_mean_logp - d.rejected_mean_logp).mean(),
                "acc_sum": ((d.chosen_logp - d.rejected_logp) > 0).mean(),
                "C_Rp": (d.chosen_logp - d.rejected_prime_logp).mean(),
                "Rp_Rpp": (d.rejected_prime_logp - d.rejected_double_prime_logp).mean(),
                "Rpp_R": (d.rejected_double_prime_logp - d.rejected_logp).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def _line(ax, traj, col, color, label, smooth):
    ax.plot(traj.step, traj[col], color=color, alpha=0.20, linewidth=0.7)
    ax.plot(
        traj.step,
        traj[col].rolling(smooth, min_periods=1, center=True).mean(),
        color=color,
        linewidth=1.8,
        label=label,
    )


def main():
    smooth = 15
    trajs = []
    for label, base, color in RUNS:
        if not glob.glob(f"{base}/decomposed_step*.csv"):
            raise SystemExit(f"No eval CSVs under {base}/")
        trajs.append((label, load_trajectory(base), color))

    last_step = max(t.step.max() for _, t, _ in trajs)
    subtitle = (
        f"Qwen2.5-1.5B-Instruct · SimPO length-norm vs sum-logp · up to step {last_step} · "
        f"n=256 every 10 steps · rolling({smooth})"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, ylim) in zip(axes.flat, PANELS):
        for label, traj, color in trajs:
            _line(ax, traj, col, color, label, smooth)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if col == "acc_sum":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.legend(fontsize=8)
    fig.suptitle("SimPO normalization effect: length-normalized vs summed log-prob\n" + subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "results/ablation_simpo_sumlogp/normalization_effect.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
