#!/usr/bin/env python3
"""
SCI Figure B: Safety-Performance Pareto Front
X: Constraint Violation Rate (%)  |  Y: Average Return
将 SafeCQL_ε=* 点按 ε 升序连成折线（机制旋钮）；CQL/BC 等为孤立散点。
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EPS_RE = re.compile(r"SafeCQL_ε=([0-9.]+(?:e[+-]?[0-9]+)?)", re.I)


def _parse_eps(name: str):
    m = EPS_RE.search(name)
    return float(m.group(1)) if m else None


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

    def is_safecql_sweep(name: str) -> bool:
        return _parse_eps(name) is not None

    sweep_pts = [(r["policy"], r) for r in unique if is_safecql_sweep(r["policy"])]
    sweep_pts.sort(key=lambda x: _parse_eps(x[0]))
    other = [r for r in unique if not is_safecql_sweep(r["policy"])]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "SafeCQL_ε=0.01": "darkgreen", "SafeCQL_ε=0.1": "green", "SafeCQL_ε=0.5": "lightgreen",
        "SafeCQL": "green", "CQL": "red", "BC": "blue", "Expert": "orange", "Random": "gray",
    }

    if sweep_pts:
        xs = [float(r["constraint_violation_rate_pct"]) for _, r in sweep_pts]
        ys = [float(r["return_mean"]) for _, r in sweep_pts]
        yerr = [float(r.get("return_std") or 0) for _, r in sweep_pts]
        ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color="darkgreen", capsize=3,
                    markersize=8, linewidth=2, label="SafeCQL (ε sweep)")
        for name, r in sweep_pts:
            c = colors.get(name, "darkgreen")
            x = float(r["constraint_violation_rate_pct"])
            y = float(r["return_mean"])
            ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)

    for r in other:
        name = r["policy"]
        c = colors.get(name, "black")
        x = float(r["constraint_violation_rate_pct"])
        y = float(r["return_mean"])
        ye = float(r.get("return_std") or 0)
        ax.errorbar(x, y, yerr=ye if ye else None, fmt="s", capsize=3, color=c, markersize=9)
        ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Constraint Violation Rate (%)", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    if sweep_pts:
        ax.legend(loc="best")
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
