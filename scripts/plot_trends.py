"""Plot decomposed-logp eval trajectories for the SimPO and DPO runs.

Reads <result_dir>/<method>/decomposed_step*.csv (one per eval, fixed slice),
computes per-step metrics, and renders two figures:

  1. cross-method:  one panel per metric, SimPO vs DPO overlaid.
  2. within-method: one panel per method, all decomposition gaps overlaid
                    (+ accuracy on a twin axis) so the chain can be compared
                    against itself.

Usage:
    python plot_trends.py --result_dir results          # v1 (conservative)
    python plot_trends.py --result_dir results/core_v2_aggressive   # v2 (moderate LR)
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHOD_COLORS = {"simpo": "#d62728", "dpo": "#1f77b4"}

# Decomposition gaps (sum logp), with display label and a distinct color for the
# within-method overlay.
GAPS = [
    ("C_Rp", "chosen − rejected′", "#1b9e77"),
    ("Rp_Rpp", "rejected′ − rejected″", "#d95f02"),
    ("Rpp_R", "rejected″ − rejected", "#7570b3"),
    ("sum_C_R", "chosen − rejected (total)", "#111111"),
]


def load_trajectory(method, base):
    files = sorted(
        glob.glob(f"{base}/{method}/decomposed_step*.csv"),
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


def plot_cross_method(trajs, out, smooth, subtitle):
    panels = [
        ("sum_C_R", "chosen − rejected (sum logp)", None),
        ("mean_C_R", "chosen − rejected (mean logp) [SimPO objective]", None),
        ("acc_sum", "accuracy: P(chosen > rejected)", (0, 1)),
        ("C_Rp", "chosen − rejected′ (sum logp)", None),
        ("Rp_Rpp", "rejected′ − rejected″ (sum logp)", None),
        ("Rpp_R", "rejected″ − rejected (sum logp)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, ylim) in zip(axes.flat, panels):
        for m, color in METHOD_COLORS.items():
            if m in trajs:
                _line(ax, trajs[m], col, color, m.upper(), smooth)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if col == "acc_sum":
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.legend(fontsize=8)
    fig.suptitle("Cross-method: SimPO vs DPO per metric\n" + subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def plot_within_method(trajs, out, smooth, subtitle):
    methods = list(trajs)
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 6), squeeze=False)
    for ax, m in zip(axes.flat, methods):
        traj = trajs[m]
        for col, label, color in GAPS:
            lw = 2.4 if col == "sum_C_R" else 1.7
            ax.plot(
                traj.step,
                traj[col].rolling(smooth, min_periods=1, center=True).mean(),
                color=color,
                linewidth=lw,
                label=label,
            )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{m.upper()} — decomposition gaps (sum logp)", fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel("Δ log-prob (nats)")
        ax.grid(alpha=0.3)
        # accuracy on a twin axis for context
        ax2 = ax.twinx()
        ax2.plot(
            traj.step,
            traj["acc_sum"].rolling(smooth, min_periods=1, center=True).mean(),
            color="#999999",
            linewidth=1.2,
            ls=":",
            label="P(chosen>rejected) [right]",
        )
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("accuracy", color="#777777")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    fig.suptitle("Within-method: decomposition gaps compared against each other\n" + subtitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="results/core_v1_conservative")
    ap.add_argument("--smooth", type=int, default=15, help="rolling-mean window (evals)")
    args = ap.parse_args()

    trajs = {}
    for m in METHOD_COLORS:
        if glob.glob(f"{args.result_dir}/{m}/decomposed_step*.csv"):
            trajs[m] = load_trajectory(m, args.result_dir)
    if not trajs:
        raise SystemExit(f"No eval CSVs under {args.result_dir}/<method>/")

    last_step = max(t.step.max() for t in trajs.values())
    subtitle = (
        f"Qwen2.5-1.5B-Instruct · {args.result_dir} · up to step {last_step} · "
        f"n=256 every 10 steps · rolling({args.smooth})"
    )
    plot_cross_method(trajs, f"{args.result_dir}/decomposed_cross_method.png", args.smooth, subtitle)
    plot_within_method(trajs, f"{args.result_dir}/decomposed_within_method.png", args.smooth, subtitle)


if __name__ == "__main__":
    main()
