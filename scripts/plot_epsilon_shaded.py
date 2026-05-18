#!/usr/bin/env python3
"""
按 ε 聚合结果作图：默认三栏子图 (Return, Violation rate, Avg dose)，
各面板 mean ± SD across seeds，适合论文主图。

输入: aggregate_eval_multi_seed.py 生成的 *agg.csv（需含 violation_pct_* 列）。

兼容: --legacy 保留旧版「双 y 轴 + return/dose 双阴影」。
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
    ap.add_argument(
        "--title",
        default="SafeCQL on held-out TCGA digital twins",
        help="总标题（legacy 模式下为单图标题）",
    )
    ap.add_argument(
        "--subtitle",
        default="",
        help="图底部说明（例: mean ± SD across 21 seeds; 93 held-out patients）",
    )
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="旧版：单图双 y 轴，return 与 dose 均带阴影",
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
        print("Empty CSV")
        return 1

    eps = np.array([float(r["epsilon"]) for r in rows])
    order = np.argsort(eps)
    eps = eps[order]
    row_sel = [rows[i] for i in order]
    ret = np.array([float(r["return_mean"]) for r in row_sel])
    ret_sd = np.array([float(r["return_std_across_seeds"]) for r in row_sel])
    dose = np.array([float(r["avg_dose_mean"]) for r in row_sel])
    dose_sd = np.array([float(r["avg_dose_std_across_seeds"]) for r in row_sel])

    has_viol = all("violation_pct_mean" in r for r in row_sel)
    viol = (
        np.array([float(r["violation_pct_mean"]) for r in row_sel])
        if has_viol
        else None
    )
    viol_sd = (
        np.array([float(r["violation_pct_std_across_seeds"]) for r in row_sel])
        if has_viol
        else None
    )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.legacy:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(eps, ret, "o-", color="C0", label="Return (mean)")
        ax.fill_between(
            eps, ret - ret_sd, ret + ret_sd, color="C0", alpha=0.25, label="±1 SD (seeds)"
        )
        ax.set_xlabel(r"Cost limit $\epsilon$")
        ax.set_ylabel("Average return")
        ax.set_title(args.title)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(eps, dose, "s--", color="C1", label="Avg dose (mean)")
        ax2.fill_between(eps, dose - dose_sd, dose + dose_sd, color="C1", alpha=0.2)
        ax2.set_ylabel("Average dose", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")

        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="best")
        if args.subtitle:
            fig.text(0.5, 0.01, args.subtitle, ha="center", fontsize=9, wrap=True)
        fig.tight_layout()
        if args.subtitle:
            fig.subplots_adjust(bottom=0.12)
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
        return 0

    # --- 三栏论文图 ---
    if not has_viol:
        print(
            "CSV 缺少 violation_pct_mean；请用新版 aggregate 的 agg.csv，或改用 --legacy",
            file=sys.stderr,
        )
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    panels = [
        (axes[0], ret, ret_sd, "Average return", "C0", "(A)"),
        (axes[1], viol, viol_sd, "Violation rate (%)", "C2", "(B)"),
        (axes[2], dose, dose_sd, "Average dose", "C1", "(C)"),
    ]

    for ax, y, y_sd, ylabel, color, tag in panels:
        ax.plot(eps, y, "o-", color=color, markersize=6, linewidth=1.8)
        ax.fill_between(eps, y - y_sd, y + y_sd, color=color, alpha=0.22)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.text(
            0.02,
            0.98,
            tag,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )

    axes[1].set_xlabel(r"Cost limit $\epsilon$")
    fig.suptitle(args.title, fontsize=12, y=1.02)

    if args.subtitle:
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        fig.text(0.5, 0.02, args.subtitle, ha="center", fontsize=9, style="italic")
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
