#!/usr/bin/env python3
"""
实验 A：按 ε 的折线 + 跨 seed 的误差带（Return 与可选 Avg dose）。
输入: aggregate_eval_multi_seed.py 生成的 results/multi_seed_agg.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="results/multi_seed_agg.csv")
    ap.add_argument("-o", "--output", default="figures/epsilon_shaded_return.png")
    ap.add_argument("--title", default="SafeCQL: Return vs ε (mean ± SD across seeds)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib")
        return 1

    path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("Empty CSV")
        return 1

    eps = np.array([float(r["epsilon"]) for r in rows])
    order = np.argsort(eps)
    eps = eps[order]
    ret = np.array([float(rows[i]["return_mean"]) for i in order])
    ret_sd = np.array([float(rows[i]["return_std_across_seeds"]) for i in order])
    dose = np.array([float(rows[i]["avg_dose_mean"]) for i in order])
    dose_sd = np.array([float(rows[i]["avg_dose_std_across_seeds"]) for i in order])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, ret, "o-", color="C0", label="Return (mean)")
    ax.fill_between(eps, ret - ret_sd, ret + ret_sd, color="C0", alpha=0.25, label="±1 SD (seeds)")
    ax.set_xlabel(r"$\epsilon$ (cost limit)")
    ax.set_ylabel("Average Return")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax2 = ax.twinx()
    ax2.plot(eps, dose, "s--", color="C1", label="Avg dose (mean)")
    ax2.fill_between(eps, dose - dose_sd, dose + dose_sd, color="C1", alpha=0.2)
    ax2.set_ylabel("Average dose", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="best")

    fig.tight_layout()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
