#!/usr/bin/env python3
"""
实验 B：相变阶梯图 — 双 Y 轴：左 Dose（阶梯/柱），右 Return（折线）；X=ε，可选对数坐标。
输入 CSV 列与 multi_seed_agg 相同（或 dense 扫描聚合后的表）。
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="results/epsilon_dense_agg.csv")
    ap.add_argument("-o", "--output", default="figures/phase_transition_dual_y.png")
    ap.add_argument("--log-x", action="store_true", help="X 轴对数刻度（突出低 ε 区）")
    ap.add_argument("--title", default="Phase transition: dose vs return across ε")
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
    if not path.exists():
        print(f"Missing {path}; run aggregate_eval_multi_seed.py on dense checkpoints first.")
        return 1

    rows = list(csv.DictReader(open(path)))
    if not rows:
        return 1

    eps = np.array([float(r["epsilon"]) for r in rows])
    order = np.argsort(eps)
    eps = eps[order]
    ret = np.array([float(rows[i]["return_mean"]) for i in order])
    dose = np.array([float(rows[i]["avg_dose_mean"]) for i in order])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = eps
    if args.log_x:
        ax1.set_xscale("symlog", linthresh=0.01)
    ax1.plot(x, dose, drawstyle="steps-mid", color="steelblue", linewidth=2.5, label="Avg dose (step)")
    ax1.set_xlabel(r"$\epsilon$")
    ax1.set_ylabel("Average dose", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, ret, "o-", color="darkorange", linewidth=2, markersize=8, label="Return")
    ax2.set_ylabel("Average Return", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title(args.title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
