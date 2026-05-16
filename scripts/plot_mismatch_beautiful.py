#!/usr/bin/env python3
"""
实验 C：Elderly Frail 等队列的 QC mismatch — Violin + Strip（pandas + seaborn + matplotlib）。

用法:
  python scripts/plot_mismatch_beautiful.py -i results/qc_mismatch_elderly.csv
  pip install pandas seaborn
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i", "--input",
        default="results/qc_mismatch_elderly.csv",
        help="evaluate_mismatch.py --cohort-id elderly_frail 生成",
    )
    p.add_argument("-o", "--output", default="figures/beautiful_mismatch_plot.png")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print("请安装: pip install pandas seaborn", file=sys.stderr)
        return 1

    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        print(
            f"缺少 {csv_path}，先运行: bash scripts/run_mismatch_elderly_plot.sh",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(csv_path)
    if "mean_qc_predicted" not in df.columns or "true_cost_rate" not in df.columns:
        print("CSV 需含: mean_qc_predicted, true_cost_rate", file=sys.stderr)
        return 1

    y1 = df["mean_qc_predicted"].dropna()
    y2 = df["true_cost_rate"].dropna()
    if len(y1) == 0 or len(y2) == 0:
        print("无有效数值", file=sys.stderr)
        return 1

    plot_data = pd.DataFrame({
        "Value": pd.concat([y1, y2], ignore_index=True),
        "Metric": (
            [r"Predicted risk ($Q_C$), episode mean"] * len(y1)
            + ["True environment cost rate, per step"] * len(y2)
        ),
    })

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            pass
    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=plot_data,
        x="Metric",
        y="Value",
        palette=["#ff9999", "#99ff99"],
        inner=None,
        ax=ax,
        alpha=0.6,
    )
    sns.stripplot(
        data=plot_data,
        x="Metric",
        y="Value",
        color="black",
        alpha=0.4,
        jitter=0.12,
        size=5,
        ax=ax,
    )
    ax.set_title(
        "Cognitive mismatch in safe offline RL (Elderly Frail cohort)",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    ax.set_ylabel("Risk / cost rate (shared scale)", fontsize=12)
    ax.set_xlabel("")

    fig.text(
        0.5,
        0.02,
        "Gap: high $Q_C$ vs. narrow low true violation rate on rollout.",
        ha="center",
        fontsize=11,
        bbox={"facecolor": "orange", "alpha": 0.12, "pad": 6},
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
