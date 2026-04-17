#!/usr/bin/env python3
"""
SCI 核心图: Return vs Constraint Violation Rate 散点图
X: 违规率 (%) 越低越好  |  Y: Average Return 越高越好
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 策略名 → 颜色（区分度高、色盲友好倾向）
POLICY_STYLE: dict[str, tuple[str, str]] = {
    "SafeCQL": ("#1B5E20", "o"),
    "CQL": ("#C62828", "s"),
    "BC": ("#1565C0", "D"),
    "Expert": ("#EF6C00", "^"),
    "Random": ("#546E7A", "X"),
    "IQL": ("#6A1B9A", "P"),
    "Fixed0.0": ("#00897B", "v"),
    "Fixed0.5": ("#43A047", "p"),
    "Fixed1.0": ("#F9A825", "h"),
    "Fixed2.0": ("#D84315", "8"),
}


def _style_for(name: str) -> tuple[str, str]:
    """color, marker; 未知策略用 tab10 色 + 稳定 hash 标记"""
    key = name.strip()
    if key in POLICY_STYLE:
        return POLICY_STYLE[key]
    h = abs(hash(key))
    return f"C{h % 10}", ["o", "s", "^", "D", "v", "P", "X", "h", "8", "p"][h % 10]


def _annotate_offsets(n: int) -> list[tuple[float, float]]:
    """环形分散，减轻文字重叠"""
    out = []
    for i in range(max(n, 1)):
        ang = 2 * math.pi * i / max(n, 1) + 0.35
        out.append((22 * math.cos(ang), 22 * math.sin(ang)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="results/eval_results.csv")
    p.add_argument("-o", "--output", default="figures/return_vs_cost.png")
    p.add_argument("--title", default="Return vs Constraint Violation Rate")
    p.add_argument("--no-labels", action="store_true", help="仅图例、不画旁注（最整洁）")
    args = p.parse_args()

    path = ROOT / args.input
    if not path.exists():
        print(f"File not found: {path}")
        print("Run: python scripts/evaluate.py -o results/eval_results.csv")
        return 1

    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        print("Empty CSV")
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    policies = [r["policy"].strip() for r in rows]
    returns = np.array([float(r["return_mean"]) for r in rows])
    return_stds = np.array([float(r.get("return_std") or 0) for r in rows])
    costs = np.array([float(r["constraint_violation_rate_pct"]) for r in rows])

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.patch.set_facecolor("white")

    offsets = _annotate_offsets(len(policies))

    for i, name in enumerate(policies):
        color, marker = _style_for(name)
        ec = "white"
        lw = 1.2
        yerr = return_stds[i] if return_stds[i] > 1e-9 else None
        ax.plot(
            costs[i],
            returns[i],
            marker=marker,
            color=color,
            markersize=11,
            markeredgecolor=ec,
            markeredgewidth=lw,
            linestyle="none",
            label=name,
            zorder=3 + i,
        )
        if yerr is not None:
            ax.errorbar(
                costs[i],
                returns[i],
                yerr=yerr,
                fmt="none",
                ecolor=color,
                capsize=4,
                capthick=1.2,
                elinewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
        if not args.no_labels:
            ox, oy = offsets[i % len(offsets)]
            ax.annotate(
                name,
                (costs[i], returns[i]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=8.5,
                color="#263238",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.92),
                zorder=5 + i,
            )

    ax.set_xlabel("Constraint Violation Rate (%)", fontsize=12)
    ax.set_ylabel("Average Return", fontsize=12)
    ax.set_title(args.title, fontsize=13, pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axhline(y=0, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.axvline(x=0, color="0.5", linestyle=":", linewidth=0.9, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        title="Policy",
        title_fontsize=10,
    )
    leg.get_frame().set_edgecolor("0.75")

    fig.tight_layout()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
