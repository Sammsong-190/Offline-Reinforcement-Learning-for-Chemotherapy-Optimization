#!/usr/bin/env python3
"""
医学期刊风格图：Kaplan-Meier（按 cohort 双子图）+ Expert vs SafeCQL 平均剂量柱状图。

Usage:
  python scripts/plot_journal_figures.py \\
    --safe-cql-ckpt checkpoints/safe_cql_limit0.1_seed42.pt \\
    -o figures \\
    --n-ep 40 --seeds 42 123 456
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.patient_cohorts import PatientGenerator  # noqa: E402
from src.evaluation import (  # noqa: E402
    Agent,
    ExpertAgent,
    PyTorchAgent,
    TERMINAL_DEATH_REASONS,
    _rollout_one,
)


def _gather_rollouts(
    agent: Agent,
    patient_ctx: dict,
    n_episodes: int,
    seeds: list[int],
) -> list[dict]:
    rows = []
    for i, seed in enumerate(seeds):
        for ep in range(n_episodes):
            ep_seed = seed + ep * 1000 + i
            m = _rollout_one(
                agent,
                params=None,
                seed=ep_seed,
                randomize_patient=False,
                patient_ctx=patient_ctx,
            )
            rows.append(m)
    return rows


def _km_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    durations = np.array([r["survival_steps"] for r in rows], dtype=float)
    events = np.array(
        [1 if r["termination_reason"] in TERMINAL_DEATH_REASONS else 0 for r in rows],
        dtype=int,
    )
    return durations, events


def plot_km_two_cohorts(
    cohort_ids: tuple[str, ...],
    policy_data: dict[str, dict[str, list[dict]]],
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """
    policy_data[cohort_id][policy_name] = list of rollout dicts
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"Expert": "#C44E52", "SafeCQL": "#4C72B0"}
    for ax, cid in zip(axes, cohort_ids):
        for name in ("Expert", "SafeCQL"):
            rows = policy_data[cid][name]
            d, e = _km_arrays(rows)
            kmf = KaplanMeierFitter(label=name)
            kmf.fit(d, e)
            kmf.plot_survival_function(ax=ax, color=colors[name], linewidth=2.2)
        ax.set_xlabel("Time (simulation steps)", fontsize=11)
        ax.set_ylabel("Survival probability", fontsize=11)
        lab = cid.replace("_", " ").title()
        ax.set_title(f"{title_prefix}{lab}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", framealpha=0.95)
        ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def plot_avg_dose_bars(
    cohort_ids: tuple[str, ...],
    policy_data: dict[str, dict[str, list[dict]]],
    out_path: Path,
) -> None:
    """分组柱状图：每个 cohort 两根柱 Expert / SafeCQL。"""
    means = {cid: {} for cid in cohort_ids}
    stds = {cid: {} for cid in cohort_ids}
    for cid in cohort_ids:
        for name in ("Expert", "SafeCQL"):
            doses = [r["avg_dose"] for r in policy_data[cid][name]]
            means[cid][name] = float(np.mean(doses))
            stds[cid][name] = float(np.std(doses)) if len(doses) > 1 else 0.0

    labels = [cid.replace("_", " ").title() for cid in cohort_ids]
    x = np.arange(len(labels))
    width = 0.35
    colors = {"Expert": "#C44E52", "SafeCQL": "#4C72B0"}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, name in enumerate(("Expert", "SafeCQL")):
        vals = [means[cid][name] for cid in cohort_ids]
        errs = [stds[cid][name] for cid in cohort_ids]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            label=name,
            color=colors[name],
            capsize=4,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average dose (per step)", fontsize=11)
    ax.set_title("Behavior policy vs SafeCQL: mean dose by cohort", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--safe-cql-ckpt",
        default=None,
        help="SafeCQL 权重路径（默认 checkpoints/safe_cql_limit0.1_seed42.pt）",
    )
    parser.add_argument("--n-ep", type=int, default=40, help="每 seed 的 episode 数")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument(
        "-o",
        "--output-dir",
        default="figures",
        help="输出目录",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="KM 子图标题前缀",
    )
    args = parser.parse_args()

    ckpt = args.safe_cql_ckpt or str(ROOT / "checkpoints" / "safe_cql_limit0.1_seed42.pt")
    if not Path(ckpt).exists():
        print(f"Missing SafeCQL checkpoint: {ckpt}", file=sys.stderr)
        print("Train first: python scripts/train.py --algo safe_cql --data ...", file=sys.stderr)
        return 1

    cohort_ids = ("young_strong", "elderly_frail")
    expert = ExpertAgent()
    safe_cql = PyTorchAgent(ckpt, "safe_cql")

    gen = PatientGenerator(rng=np.random.default_rng(42))
    policy_data: dict[str, dict[str, list[dict]]] = {
        cid: {"Expert": [], "SafeCQL": []} for cid in cohort_ids
    }

    for cid in cohort_ids:
        ctx = gen.from_cohort(cid, jitter=0.0)
        policy_data[cid]["Expert"] = _gather_rollouts(
            expert, ctx, args.n_ep, args.seeds
        )
        policy_data[cid]["SafeCQL"] = _gather_rollouts(
            safe_cql, ctx, args.n_ep, args.seeds
        )

    out_dir = Path(args.output_dir)
    plot_km_two_cohorts(
        cohort_ids,
        policy_data,
        out_dir / "km_survival_expert_vs_safecql.png",
        title_prefix=args.prefix,
    )
    plot_avg_dose_bars(
        cohort_ids,
        policy_data,
        out_dir / "bar_avg_dose_expert_vs_safecql.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
