#!/usr/bin/env python3
"""
SCI 灵敏度分析: 评估三组 cost_limit 模型，输出 sensitivity.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="results/sensitivity.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--n-ep", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    args = p.parse_args()

    from src.evaluation import Evaluator, PyTorchAgent

    agents = {}
    seed = 42
    for limit in [0.01, 0.1, 0.5]:
        path = ROOT / args.checkpoint_dir / f"safe_cql_limit{limit}_seed{seed}.pt"
        if path.exists():
            agents[f"SafeCQL_ε={limit}"] = PyTorchAgent(str(path), "safe_cql")
        else:
            legacy = ROOT / args.checkpoint_dir / f"safe_cql_limit{limit}.pt"
            if legacy.exists():
                agents[f"SafeCQL_ε={limit}"] = PyTorchAgent(str(legacy), "safe_cql")
            else:
                print(f"Skip (not found): {path.name}")

    if not agents and (ROOT / "safe_cql_model.pt").exists():
        agents["SafeCQL"] = PyTorchAgent(str(ROOT / "safe_cql_model.pt"), "safe_cql")
        print("Using safe_cql_model.pt from project root")

    if not agents:
        print("No Safe CQL models. Run: bash scripts/run_cost_sensitivity.sh")
        return 1

    evaluator = Evaluator()
    results = evaluator.evaluate_all(agents, n_episodes=args.n_ep, seeds=args.seeds)

    for name, m in results.items():
        print(f"{name:20} Return={m['return_mean']:.2f}±{m['return_std']:.2f}  "
              f"Violation={m['constraint_violation_rate_pct']:.1f}%")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    Evaluator.save_csv(results, str(out))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
