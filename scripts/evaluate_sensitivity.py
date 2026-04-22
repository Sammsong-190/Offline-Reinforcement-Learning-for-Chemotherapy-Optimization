#!/usr/bin/env python3
"""
实验二：评估不同 cost_limit (ε) 的 SafeCQL + 可选 CQL/BC 基线 → sensitivity.csv
checkpoint 命名与 train.py 一致: checkpoints/safe_cql_limit{cost_limit}_seed{seed}.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_LIMITS = [0.0, 0.1, 0.3, 0.5, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="results/sensitivity.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-ep", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--limits", nargs="+", type=float, default=DEFAULT_LIMITS,
                   help=f"与训练时 --cost-limit 一致；默认 {DEFAULT_LIMITS}")
    p.add_argument("--seed", type=int, default=42, help="匹配 checkpoint 文件名中的 seed")
    p.add_argument("--with-cql", action="store_true",
                   help="若存在 checkpoints/cql_model.d3 则加入评估")
    p.add_argument("--cql-path", default=None, help="CQL 权重路径（覆盖默认）")
    p.add_argument("--with-bc", action="store_true",
                   help="若存在 bc_policy.pt（项目根或 checkpoints）则加入")
    p.add_argument("--bc-path", default=None)
    args = p.parse_args()

    from src.evaluation import D3RLPyAgent, Evaluator, PyTorchAgent

    agents = {}
    ckdir = ROOT / args.checkpoint_dir

    for limit in args.limits:
        path = ckdir / f"safe_cql_limit{limit}_seed{args.seed}.pt"
        if path.exists():
            label = f"SafeCQL_ε={limit}"
            agents[label] = PyTorchAgent(str(path), "safe_cql")
        else:
            print(f"Skip (not found): {path.name}")

    cql = Path(args.cql_path) if args.cql_path else (ckdir / "cql_model.d3")
    if not cql.is_absolute():
        cql = ROOT / cql
    if args.with_cql:
        if cql.exists():
            agents["CQL"] = D3RLPyAgent(str(cql))
        else:
            print(f"Skip CQL (not found): {cql}")

    if args.with_bc:
        candidates = [ROOT / "bc_policy.pt", ckdir / "bc_policy.pt"]
        if args.bc_path:
            candidates.insert(0, Path(args.bc_path))
        found_bc = False
        for cand in candidates:
            pth = Path(cand)
            if not pth.is_absolute():
                pth = ROOT / pth
            if pth.exists():
                agents["BC"] = PyTorchAgent(str(pth), "bc")
                found_bc = True
                break
        if not found_bc:
            print("Skip BC (bc_policy.pt not found)")

    if not agents:
        legacy = ROOT / "safe_cql_model.pt"
        if legacy.exists():
            agents["SafeCQL"] = PyTorchAgent(str(legacy), "safe_cql")
            print("Using safe_cql_model.pt from project root")
        else:
            print("No Safe CQL models. Run: bash scripts/run_constraint_sweep.sh")
            return 1

    evaluator = Evaluator()
    results = evaluator.evaluate_all(agents, n_episodes=args.n_ep, seeds=args.seeds)

    for name, m in results.items():
        print(f"{name:22} Return={m['return_mean']:.2f}±{m['return_std']:.2f}  "
              f"Violation={m['constraint_violation_rate_pct']:.1f}%  Dose={m['avg_dose']:.2f}")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    Evaluator.save_csv(results, str(out))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
