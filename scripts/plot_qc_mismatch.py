#!/usr/bin/env python3
"""
实验 C：QC / safety critic 可视化。

- 默认：并排小提琴图（episode 平均 Q_C vs 真实逐步违规率）。
- --scatter：按 episode 索引 overlay 散点。
- --calibration：x = 预测 Q_C episode 均值，y = true violation rate，每点一条轨迹
  （TCGA held-out CSV 下每点一名患者）；可选最小二乘趋势线。
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
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument(
        "--title",
        default="",
        help="图总标题；--calibration 时默认 Safety critic calibration on held-out TCGA twins",
    )
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

    nunique_pred = len(np.unique(np.round(pred, 8)))
    nunique_true = len(np.unique(np.round(true_rate, 8)))
    if nunique_pred <= 1 and nunique_true <= 1:
        print(
            "Warning: all points identical (variance 0). "
            "Default env rollouts may be deterministic; "
            "re-run evaluate_mismatch.py with --tcga-clinical for per-patient spread.",
            file=sys.stderr,
        )

    if args.calibration:
        title = args.title or "Safety critic calibration on held-out TCGA twins"
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(pred, true_rate, alpha=0.55, s=28, edgecolors="k", linewidths=0.3)
        valid = np.isfinite(pred) & np.isfinite(true_rate)
        px, ty = pred[valid], true_rate[valid]
        if px.size >= 2 and np.std(px) > 1e-12:
            coef = np.polyfit(px, ty, 1)
            xs = np.linspace(float(np.min(px)), float(np.max(px)), 50)
            ax.plot(xs, np.poly1d(coef)(xs), "r-", lw=1.8, alpha=0.85, label="OLS fit")
            ax.legend(loc="best", fontsize=9)
        ax.set_xlabel(r"Episode mean $Q_C(s,a)$ (predicted)", fontsize=11)
        ax.set_ylabel("True per-step violation rate", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, max(1.02, float(np.max(true_rate)) * 1.05 + 0.02))
    elif args.scatter:
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
        mean_v = float(np.mean(true_rate))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        ax1.violinplot([pred], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax1.set_xticks([1])
        ax1.set_xticklabels([r"$Q_C$ pred."])
        ax1.set_ylabel(r"Episode mean $Q_C(s,a)$")
        ax1.set_title("Predicted safety value (episode mean)")
        ax2.violinplot([true_rate], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["True rate"])
        ax2.set_ylabel("Fraction of steps with cost = 1")
        ax2.set_title(f"Observed violation rate (mean = {mean_v:.3f})")
        fig.suptitle("Safety critic vs environment rollout")
        fig.tight_layout()

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
