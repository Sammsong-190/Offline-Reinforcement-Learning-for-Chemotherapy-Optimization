#!/usr/bin/env python3
"""
统一评估入口 - 策略模式
使用 src.evaluation.Evaluator + Agent 接口
SCI 格式: Return, Constraint Violation Rate, Survival Rate
多种子，保存 CSV 供 notebooks 画 Return vs Cost 图
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None,
                        help="e.g. Expert BC SafeCQL CQL. Default: all available")
    parser.add_argument("--n_ep", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Multi-seed for SCI (单一种子不被认可)")
    parser.add_argument("--ood", action="store_true", help="OOD: randomize patient params")
    parser.add_argument("--output", "-o", default="results/eval_results.csv")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    from src.evaluation import Evaluator, build_agents

    agents = build_agents(ROOT)
    if args.policies:
        agents = {k: v for k, v in agents.items() if k in args.policies}

    evaluator = Evaluator()
    results = evaluator.evaluate_all(agents, n_episodes=args.n_ep, seeds=args.seeds,
                                     randomize_patient=args.ood)

    for name, m in results.items():
        print(f"{name:12} Return={m['return_mean']:.2f}±{m['return_std']:.2f}  "
              f"Violation={m['constraint_violation_rate_pct']:.1f}%  Survival={m['survival_pct']:.0f}%  "
              f"TumorClear={m['tumor_clear_pct']:.0f}%  Dose={m['avg_dose']:.2f}")

    if not args.no_csv and results:
        Evaluator.save_csv(results, args.output)


if __name__ == "__main__":
    main()
