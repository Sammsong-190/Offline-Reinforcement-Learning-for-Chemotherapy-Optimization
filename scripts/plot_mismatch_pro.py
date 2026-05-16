#!/usr/bin/env python3
"""
顶刊风 QC mismatch：Boxplot + Stripplot，中间双向箭头标 Gap（pandas + seaborn）。
依赖: pip install pandas seaborn
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="results/qc_mismatch_elderly.csv")
    p.add_argument("--png", default="figures/mismatch_plot_pro.png")
    p.add_argument("--pdf", default="figures/mismatch_plot_pro.pdf")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("请安装: pip install pandas seaborn", file=sys.stderr)
        return 1

    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        print(f"找不到: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    for col in ("mean_qc_predicted", "true_cost_rate"):
        if col not in df.columns:
            print(f"CSV 缺少列: {col}", file=sys.stderr)
            return 1

    n = len(df)
    plot_data = pd.DataFrame({
        "Value": pd.concat(
            [df["mean_qc_predicted"], df["true_cost_rate"]], ignore_index=True
        ),
        "Metric": (
            [r"Predicted Risk ($Q_C$)" + "\n(Agent's Internal Belief)"] * n
            + ["True Cost Rate\n(Actual Env Dynamics)"] * n
        ),
    })

    for style in ("seaborn-v0_8-ticks", "seaborn-ticks", "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Helvetica Neue"],
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#333333",
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c44e52", "#1f77b4"]  # crimson family, steel blue

    # hue=Metric 与 x 相同仅为满足 seaborn ≥0.14 的 palette 着色；legend 关闭
    sns.boxplot(
        data=plot_data,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors,
        width=0.4,
        dodge=False,
        legend=False,
        boxprops=dict(alpha=0.3, edgecolor="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=colors,
        dodge=False,
        legend=False,
        alpha=0.6,
        jitter=0.15,
        size=6,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    mean_qc = float(df["mean_qc_predicted"].mean())
    mean_true = float(df["true_cost_rate"].mean())
    y_lo, y_hi = min(mean_true, mean_qc), max(mean_true, mean_qc)
    ax.annotate(
        "",
        xy=(0.5, y_lo),
        xytext=(0.5, y_hi),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=2),
    )

    if mean_true > 1e-12:
        ratio = mean_qc / mean_true
        gap_text = f"Massive Cognitive Gap\n(~{ratio:.0f}× Overestimation)"
    else:
        gap_text = "Massive Cognitive Gap\n(∞ vs. ~0 true rate)"

    ax.text(
        0.53,
        (y_lo + y_hi) / 2,
        gap_text,
        va="center",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="none", alpha=0.8),
    )

    ax.set_ylabel(
        "Risk Value / Constraint Violation Rate",
        fontweight="bold",
        labelpad=10,
    )
    ax.set_xlabel("")
    ax.set_title(
        "Cognitive Mismatch: OOD Over-pessimism in Safe Offline RL",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for out, kind in ((args.png, "png"), (args.pdf, "pdf")):
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white", format=kind
        )
        print(f"Saved {out_path.resolve()}")

    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
