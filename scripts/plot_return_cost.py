#!/usr/bin/env python3
"""
SCI 核心图: Return vs Cost 散点图
X 轴: Average Test Cost Rate (越低越好)
Y 轴: Average Test Return (越高越好)
Safe CQL 应位于左上角 (高收益、低风险)
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="results/eval_results.csv")
    p.add_argument("-o", "--output", default="results/return_vs_cost.png")
    p.add_argument("--title", default="Return vs Constraint Violation Rate")
    args = p.parse_args()

    path = ROOT / args.input
    if not path.exists():
        print(f"File not found: {path}")
        print("Run: python scripts/evaluate.py -o results/eval_results.csv")
        return 1

    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("Empty CSV")
        return 1

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("pip install matplotlib")
        return 1

    policies = [r["policy"] for r in rows]
    returns = [float(r["return_mean"]) for r in rows]
    return_stds = [float(r.get("return_std", 0)) for r in rows]
    costs = [float(r["constraint_violation_rate_pct"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"SafeCQL": "green", "BC": "blue", "CQL": "red", "Expert": "orange", "Random": "gray"}
    for i, name in enumerate(policies):
        c = colors.get(name, "black")
        ax.errorbar(
            costs[i], returns[i],
            yerr=return_stds[i] if return_stds[i] else None,
            fmt="o", capsize=3, color=c, label=name, markersize=10,
        )
        ax.annotate(name, (costs[i], returns[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Constraint Violation Rate (%)", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title(args.title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
