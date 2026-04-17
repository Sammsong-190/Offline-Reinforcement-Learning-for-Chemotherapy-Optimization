#!/usr/bin/env python3
"""
SafeCQL 训练动态：λ、估计 Cost 风险 (Q_C(π))、奖励 Q 拟合损失。
需训练时使用 --log-lambda N 生成 checkpoints/*_lambda.json。

说明：离线训练日志中无「回合 Return」；用 Q_R loss 反映价值网络拟合进度。
Usage:
  python scripts/plot_training_dynamics.py --lambda-json checkpoints/safe_cql_limit0.1_seed42_lambda.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lambda-json",
        type=Path,
        default=None,
        help="如 checkpoints/safe_cql_limit0.1_seed42_lambda.json",
    )
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "figures" / "training_dynamics_safecql.png")
    args = ap.parse_args()

    path = args.lambda_json
    if path is None:
        cand = ROOT / "checkpoints" / "safe_cql_limit0.1_seed42_lambda.json"
        path = cand if cand.exists() else None
    if path is None or not Path(path).exists():
        print(
            "未找到 lambda 日志。请用以下命令重新训练并记录：\n"
            "  python scripts/train.py --algo safe_cql --data <your.npz> --seed 42 --log-lambda 1000",
            file=sys.stderr,
        )
        return 1

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    hist = payload.get("history", [])
    cost_limit = float(payload.get("cost_limit", 0.1))
    if not hist:
        print("history 为空", file=sys.stderr)
        return 1

    steps = [h["step"] for h in hist]
    lambdas = [h["lambda"] for h in hist]
    risks = [h.get("current_risk", 0) for h in hist]
    qr = [h.get("qr_loss", 0) for h in hist]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(steps, lambdas, color="#4C72B0", lw=1.4)
    ax0.set_ylabel(r"Lagrangian $\lambda$")
    ax0.set_title("SafeCQL training dynamics")
    ax0.grid(True, alpha=0.3)

    ax1.plot(steps, risks, color="#C44E52", lw=1.4, label=r"$\mathbb{E}[Q_C](\pi)$ (est.)")
    ax1.axhline(cost_limit, color="gray", ls="--", lw=1.2, label=f"Cost budget $\\varepsilon$={cost_limit}")
    ax1.set_ylabel("Cost / risk")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, qr, color="#2CA02C", lw=1.2)
    ax2.set_ylabel(r"$Q_R$ loss (batch)")
    ax2.set_xlabel("Training step")
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.01,
        0.02,
        "Episode return is not logged (offline RL); $Q_R$ loss reflects reward critic learning.",
        transform=ax2.transAxes,
        fontsize=8,
        color="0.35",
        va="bottom",
    )

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
