#!/usr/bin/env python3
"""
实验 C：并排小提琴图 — 左：episode 平均 Q_C(s,a)；右：真实逐步违规率 true_cost_rate。
也可 --scatter 画每步 episode 散点对比。
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="results/qc_mismatch_episodes.csv")
    ap.add_argument("-o", "--output", default="figures/qc_mismatch_violin.png")
    ap.add_argument("--scatter", action="store_true")
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
        print("Empty CSV; run scripts/evaluate_mismatch.py")
        return 1

    pred, true_rate = [], []
    for r in rows:
        if r.get("mean_qc_predicted") in ("", None):
            continue
        pred.append(float(r["mean_qc_predicted"]))
        true_rate.append(float(r.get("true_cost_rate") or 0.0))
    pred = np.array(pred)
    true_rate = np.array(true_rate)
    if len(pred) == 0:
        print("No mean_qc_predicted in CSV")
        return 1

    if args.scatter:
        fig, ax = plt.subplots(figsize=(8, 5))
        ep = np.arange(len(pred))
        ax.scatter(ep, pred, alpha=0.45, s=14, label=r"Mean $Q_C$ / episode")
        ax.scatter(ep, true_rate, alpha=0.45, s=14, label="True cost rate / step")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.legend()
        ax.set_title(r"Predicted $Q_C$ vs true violation rate (per episode)")
        ax.grid(True, alpha=0.3)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        ax1.violinplot([pred], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax1.set_xticks([1])
        ax1.set_xticklabels([r"$Q_C$ pred."])
        ax1.set_ylabel(r"Episode mean $Q_C(s,a)$")
        ax1.set_title("Critic believes risk is high")
        ax2.violinplot([true_rate], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["True rate"])
        ax2.set_ylabel("Fraction of steps with c=1")
        ax2.set_title("Environment: near-zero violations")
        fig.suptitle("Mismatch: conservative critic vs safe rollout")
        fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
