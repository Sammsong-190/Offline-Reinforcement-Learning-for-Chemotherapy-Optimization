#!/usr/bin/env python3
"""
奖励敏感性实验柱状图：从 evaluate 导出的 CSV 读取 policy, avg_dose 或 frac_toxicity_death
例: 合并 default 与 high_incentive 两次评估的 CSV 后:
  python scripts/plot_reward_bar.py -i results/reward_compare.csv -o figures/reward_sensitivity_bars.png --metric avg_dose
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="results/eval_results.csv")
    p.add_argument("-o", "--output", default="figures/reward_sensitivity_bars.png")
    p.add_argument(
        "--metric",
        choices=["avg_dose", "frac_toxicity_death"],
        default="avg_dose",
    )
    p.add_argument("--title", default="Reward sensitivity (high-incentive vs baseline)")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib")
        return 1

    path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if not path.exists():
        print(f"Not found: {path}")
        return 1

    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("Empty CSV")
        return 1

    policies = [r["policy"] for r in rows]
    vals = []
    for r in rows:
        if args.metric == "avg_dose":
            vals.append(float(r["avg_dose"]))
        else:
            k = "frac_toxicity_death"
            if k not in r or r[k] == "":
                print(f"Missing {k} for policy {r.get('policy')}")
                return 1
            vals.append(float(r[k]) * 100.0)

    ylabel = "Average dose" if args.metric == "avg_dose" else "Toxicity death rate (%)"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policies))
    ax.bar(x, vals, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
