#!/usr/bin/env python3
"""
SCI Figure B: Safety-Performance Pareto Front
X: Constraint Violation Rate (%)  |  Y: Average Return
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", nargs="+", default=["results/sensitivity.csv", "results/eval_results.csv"])
    p.add_argument("-o", "--output", default="figures/pareto_front.png")
    p.add_argument("--title", default="Figure B: Safety-Performance Pareto Front")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return 1

    rows = []
    for fpath in args.input:
        for base in [ROOT, Path.cwd()]:
            path = base / fpath if not Path(fpath).is_absolute() else Path(fpath)
            if path.exists():
                with open(path) as f:
                    for r in csv.DictReader(f):
                        rows.append(r)
                break

    if not rows:
        print("No CSV found. Run: python scripts/evaluate_sensitivity.py -o results/sensitivity.csv")
        return 1

    seen = set()
    unique = []
    for r in rows:
        if r["policy"] not in seen:
            seen.add(r["policy"])
            unique.append(r)

    policies = [r["policy"] for r in unique]
    returns = [float(r["return_mean"]) for r in unique]
    return_stds = [float(r.get("return_std", 0)) for r in unique]
    costs = [float(r["constraint_violation_rate_pct"]) for r in unique]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"SafeCQL_ε=0.01": "darkgreen", "SafeCQL_ε=0.1": "green", "SafeCQL_ε=0.5": "lightgreen",
              "SafeCQL": "green", "CQL": "red", "BC": "blue", "Expert": "orange", "Random": "gray"}
    for i, name in enumerate(policies):
        c = colors.get(name, "black")
        ax.errorbar(costs[i], returns[i], yerr=return_stds[i] if return_stds[i] else None,
                    fmt="o", capsize=3, color=c, markersize=10)
        ax.annotate(name, (costs[i], returns[i]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Constraint Violation Rate (%)", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title(args.title)
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
