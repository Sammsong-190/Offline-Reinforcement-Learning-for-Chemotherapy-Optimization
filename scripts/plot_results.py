#!/usr/bin/env python3
"""
从 results/*.csv 读取汇总指标并出图；KM / 箱线图 / 轨迹 需环境 rollout（与 CSV 互补）。

- CSV：柱状图（各 cohort×policy 的 avg_dose、survival 等汇总）
- Rollout（默认开启）：Kaplan-Meier、SafeCQL 剂量箱线图、典型 young_strong 轨迹 (T, I, dose)

Usage:
  python scripts/plot_results.py --results-dir results --fig-dir figures
  python scripts/plot_results.py --no-rollout   # 仅根据已有 CSV 画汇总柱图
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.chemo_env import DT, MAX_STEPS, X0  # noqa: E402
from env.patient_cohorts import PatientGenerator  # noqa: E402
from env.robust import set_seed  # noqa: E402
from src.evaluation import (  # noqa: E402
    Agent,
    ExpertAgent,
    PyTorchAgent,
    TERMINAL_DEATH_REASONS,
    _rollout_one,
    step_ode,
    termination_info,
)


def _f(x: Any, default: float | None = None) -> float | None:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_results_csvs(results_dir: Path, pattern: str = "*.csv") -> list[dict]:
    """合并 results 目录下所有 CSV 行（含 cohort/policy 的虚拟试验表）。"""
    rows: list[dict] = []
    for path in sorted(results_dir.glob(pattern)):
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["_file"] = path.name
                rows.append(row)
    return rows


def plot_csv_summary_bars(rows: list[dict], out_path: Path) -> bool:
    """
    从汇总 CSV 画 grouped bar：需存在 cohort, policy, avg_dose（或同名变体）。
    """
    sub = [r for r in rows if r.get("cohort") and r.get("policy") and _f(r.get("avg_dose")) is not None]
    if len(sub) < 2:
        print("  [skip] CSV 汇总柱图：未找到足够的 cohort/policy/avg_dose 行")
        return False

    cohorts = sorted({r["cohort"] for r in sub})
    policies = sorted({r["policy"] for r in sub})
    if len(policies) > 8:
        policies = policies[:8]

    x = np.arange(len(cohorts))
    width = 0.8 / max(len(policies), 1)
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(policies)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pi, pol in enumerate(policies):
        vals = []
        for c in cohorts:
            m = [r for r in sub if r["cohort"] == c and r["policy"] == pol]
            if m:
                vals.append(_f(m[0].get("avg_dose"), 0.0))
            else:
                vals.append(0.0)
        offset = (pi - (len(policies) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=pol, color=cmap[pi])

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in cohorts], rotation=15, ha="right")
    ax.set_ylabel("avg_dose (from CSV summary)")
    ax.set_title("Average dose by cohort & policy (results/*.csv)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")
    return True


def gather_rollouts(
    agent: Agent,
    patient_ctx: dict,
    n_episodes: int,
    seeds: list[int],
) -> list[dict]:
    rows = []
    for i, seed in enumerate(seeds):
        for ep in range(n_episodes):
            ep_seed = seed + ep * 1000 + i
            rows.append(
                _rollout_one(
                    agent,
                    params=None,
                    seed=ep_seed,
                    randomize_patient=False,
                    patient_ctx=patient_ctx,
                )
            )
    return rows


def km_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    d = np.array([r["survival_steps"] for r in rows], dtype=float)
    e = np.array(
        [1 if r["termination_reason"] in TERMINAL_DEATH_REASONS else 0 for r in rows],
        dtype=int,
    )
    return d, e


def plot_km(
    cohort_ids: tuple[str, ...],
    data: dict[str, dict[str, list[dict]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"Expert": "#C44E52", "SafeCQL": "#4C72B0"}
    for ax, cid in zip(axes, cohort_ids):
        for name in ("Expert", "SafeCQL"):
            d, ev = km_arrays(data[cid][name])
            kmf = KaplanMeierFitter(label=name)
            kmf.fit(d, ev)
            kmf.plot_survival_function(ax=ax, color=colors[name], linewidth=2.2)
        ax.set_xlabel("Time (simulation steps)")
        ax.set_ylabel("Survival probability")
        ax.set_title(cid.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def plot_dose_boxplot(
    cohort_ids: tuple[str, ...],
    safe_rows: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """SafeCQL 在各 cohort 的每局 avg_dose 箱线图。"""
    data, labels = [], []
    for cid in cohort_ids:
        doses = [r["avg_dose"] for r in safe_rows[cid]]
        if doses:
            data.append(doses)
            labels.append(cid.replace("_", " ").title())
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#4C72B0", alpha=0.55)
    ax.set_ylabel("Average dose (per episode)")
    ax.set_title("SafeCQL: dose distribution by cohort")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def rollout_trace(
    agent: Agent,
    patient_ctx: dict,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """单条轨迹：T, I, dose 随 step（用于 SafeCQL 典型 young_strong）。"""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    dyn_params = patient_ctx.get("params")
    sde_sigma = float(patient_ctx.get("sde_sigma", 0.0))
    ctx = patient_ctx

    x = np.array(X0, dtype=np.float32)
    ts, ims, doses = [], [], []

    for _step in range(MAX_STEPS):
        a = agent.get_action(x)
        doses.append(float(a))
        x = step_ode(x, a, DT, dyn_params, sde_sigma=sde_sigma, rng=rng)
        ts.append(float(x[1]))
        ims.append(float(x[2]))
        done, _ = termination_info(x, ctx)
        if done:
            break

    steps = np.arange(len(ts))
    return {
        "step": steps,
        "T": np.array(ts),
        "I": np.array(ims),
        "dose": np.array(doses),
    }


def plot_trajectory(tr: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax0, ax1, ax2 = axes
    s = tr["step"]
    ax0.plot(s, tr["T"], color="#E377C2", lw=1.8)
    ax0.set_ylabel("Tumor T")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax1.plot(s, tr["I"], color="#2CA02C", lw=1.8)
    ax1.set_ylabel("Immune I")
    ax1.grid(True, alpha=0.3)
    ax2.step(s, tr["dose"], where="post", color="#4C72B0", lw=1.5)
    ax2.set_ylabel("Dose")
    ax2.set_xlabel("Simulation step")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot paper figures from results/*.csv + optional rollouts")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results", help="CSV 目录")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures", help="图输出目录")
    ap.add_argument("--csv-pattern", default="*.csv", help="glob 相对 results-dir")
    ap.add_argument("--safe-cql-ckpt", default=None, help="SafeCQL 权重")
    ap.add_argument("--n-ep", type=int, default=40)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--no-rollout", action="store_true", help="不跑环境，仅尝试从 CSV 画汇总柱图")
    ap.add_argument("--trajectory-seed", type=int, default=42, help="典型轨迹随机种子")
    args = ap.parse_args()

    res_dir = args.results_dir
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results_csvs(res_dir, args.csv_pattern)
    print(f"Loaded {len(rows)} rows from {res_dir}/{args.csv_pattern}")
    plot_csv_summary_bars(rows, fig_dir / "csv_summary_avg_dose.png")

    if args.no_rollout:
        print("Done (--no-rollout).")
        return 0

    ckpt = args.safe_cql_ckpt or str(ROOT / "checkpoints" / "safe_cql_limit0.1_seed42.pt")
    if not Path(ckpt).exists():
        print(f"Skip rollout plots: checkpoint not found: {ckpt}", file=sys.stderr)
        return 0

    cohort_ids = ("young_strong", "elderly_frail")
    expert = ExpertAgent()
    safe = PyTorchAgent(ckpt, "safe_cql")
    gen = PatientGenerator(rng=np.random.default_rng(42))

    policy_data: dict[str, dict[str, list[dict]]] = {
        cid: {"Expert": [], "SafeCQL": []} for cid in cohort_ids
    }
    safe_only: dict[str, list[dict]] = {cid: [] for cid in cohort_ids}

    for cid in cohort_ids:
        ctx = gen.from_cohort(cid, jitter=0.0)
        policy_data[cid]["Expert"] = gather_rollouts(expert, ctx, args.n_ep, args.seeds)
        policy_data[cid]["SafeCQL"] = gather_rollouts(safe, ctx, args.n_ep, args.seeds)
        safe_only[cid] = policy_data[cid]["SafeCQL"]

    plot_km(cohort_ids, policy_data, fig_dir / "km_survival_expert_vs_safecql.png")
    plot_dose_boxplot(cohort_ids, safe_only, fig_dir / "box_dose_safecql_by_cohort.png")

    ctx_ys = gen.from_cohort("young_strong", jitter=0.0)
    tr = rollout_trace(safe, ctx_ys, seed=args.trajectory_seed)
    plot_trajectory(
        tr,
        fig_dir / "trajectory_safecql_young_strong.png",
        title="Typical young_strong rollout: SafeCQL (T, I, dose)",
    )

    print(f"All figures under {fig_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
