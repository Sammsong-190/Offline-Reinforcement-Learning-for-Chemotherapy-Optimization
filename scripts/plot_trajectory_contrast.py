#!/usr/bin/env python3
"""
灾难轨迹 vs SafeCQL 成功轨迹（同一虚拟患者上下文、同一随机种子）。
三行：Tumor T、Immune I（含安全阈值红线）、Dose；两列：SafeCQL | CQL（或 Expert）。

Usage:
  python scripts/plot_trajectory_contrast.py -o figures/trajectory_contrast_young_strong.png
  python scripts/plot_trajectory_contrast.py --baseline expert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from env.chemo_env import DT, MAX_STEPS, X0, I_SAFE, step_ode, termination_info  # noqa: E402
from env.patient_cohorts import PatientGenerator  # noqa: E402
from env.robust import set_seed  # noqa: E402
from src.evaluation import D3RLPyAgent, ExpertAgent, PyTorchAgent  # noqa: E402


def rollout_trace(agent, patient_ctx: dict, seed: int = 42):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    dyn_params = patient_ctx.get("params")
    sde_sigma = float(patient_ctx.get("sde_sigma", 0.0))
    ctx = patient_ctx
    i_thr = float(ctx.get("i_safe", I_SAFE))

    x = np.array(X0, dtype=np.float32)
    ts, ims, doses = [], [], []

    for _ in range(MAX_STEPS):
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
        "i_safe": i_thr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="young_strong", help="固定亚群 patient_ctx")
    ap.add_argument(
        "--baseline",
        choices=["cql", "expert"],
        default="cql",
        help="右列失败轨迹：CQL 或 Expert",
    )
    ap.add_argument("--seed", type=int, default=42, help="两列共用同一 seed，保证可比")
    ap.add_argument("--safe-cql-ckpt", default=None)
    ap.add_argument("--cql-path", default=None, help="默认 项目根/cql_model.d3")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "figures" / "trajectory_contrast_safecql_vs_cql.png")
    args = ap.parse_args()

    ckpt = args.safe_cql_ckpt or str(ROOT / "checkpoints" / "safe_cql_limit0.1_seed42.pt")
    if not Path(ckpt).exists():
        print(f"Missing SafeCQL checkpoint: {ckpt}", file=sys.stderr)
        return 1

    cql_path = args.cql_path or str(ROOT / "cql_model.d3")
    if args.baseline == "cql" and not Path(cql_path).exists():
        print(f"Missing CQL model: {cql_path}", file=sys.stderr)
        return 1

    gen = PatientGenerator(rng=np.random.default_rng(42))
    ctx = gen.from_cohort(args.cohort, jitter=0.0)

    safe_agent = PyTorchAgent(ckpt, "safe_cql")
    if args.baseline == "cql":
        baseline_agent = D3RLPyAgent(cql_path)
        bl_name = "CQL"
    else:
        baseline_agent = ExpertAgent(epsilon=0.0)
        bl_name = "Expert"

    tr_s = rollout_trace(safe_agent, ctx, seed=args.seed)
    tr_b = rollout_trace(baseline_agent, ctx, seed=args.seed)
    i_line = tr_s["i_safe"]

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 7), sharex="col")
    titles = ("SafeCQL", bl_name)
    traces = (tr_s, tr_b)
    colors_t = ("#E377C2", "#E377C2")
    colors_i = ("#2CA02C", "#2CA02C")
    colors_d = ("#4C72B0", "#C44E52")

    for col, (title, tr, ct, ci, cd) in enumerate(zip(titles, traces, colors_t, colors_i, colors_d)):
        s = tr["step"]
        axes[0, col].plot(s, tr["T"], color=ct, lw=1.8)
        axes[0, col].set_ylabel("Tumor $T$")
        axes[0, col].set_title(title)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(s, tr["I"], color=ci, lw=1.8)
        axes[1, col].axhline(i_line, color="darkred", ls="--", lw=1.2, label=r"$I_{\mathrm{safe}}$")
        axes[1, col].set_ylabel("Immune $I$")
        if col == 1:
            axes[1, col].legend(loc="upper right", fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].step(s, tr["dose"], where="post", color=colors_d[col], lw=1.5)
        axes[2, col].set_ylabel("Dose")
        axes[2, col].set_xlabel("Simulation step")
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle(
        f"Trajectory contrast ({args.cohort.replace('_', ' ')}, same patient_ctx & seed={args.seed})",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
