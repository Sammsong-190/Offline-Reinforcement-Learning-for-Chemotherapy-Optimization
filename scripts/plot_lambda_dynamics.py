#!/usr/bin/env python3
"""
SCI Figure A: Lagrange Multiplier Dynamics
X: Training Steps  |  Y: Lambda (λ)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42, help="与训练 --seed 一致")
    p.add_argument("-o", "--output", default="results/lambda_dynamics.png")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return 1

    data = {}
    s = args.seed
    for limit in [0.01, 0.1, 0.5]:
        path = ROOT / args.dir / f"safe_cql_limit{limit}_seed{s}_lambda.json"
        if not path.exists():
            path = ROOT / args.dir / f"safe_cql_limit{limit}_lambda.json"
        if path.exists():
            with open(path) as f:
                data[limit] = json.load(f)

    if not data:
        print("No lambda logs. Re-run: bash scripts/run_cost_sensitivity.sh")
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {0.01: "Strict (ε=0.01)", 0.1: "Moderate (ε=0.1)", 0.5: "Loose (ε=0.5)"}
    for limit in [0.01, 0.1, 0.5]:
        if limit not in data:
            continue
        hist = data[limit]["history"]
        steps = [h["step"] for h in hist]
        lambdas = [h["lambda"] for h in hist]
        ax.plot(steps, lambdas, label=labels[limit], linewidth=1.5)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Lagrange Multiplier (λ)", fontsize=12)
    ax.set_title("Figure A: Lagrange Multiplier Dynamics")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
