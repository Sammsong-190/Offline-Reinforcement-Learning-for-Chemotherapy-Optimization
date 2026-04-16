#!/usr/bin/env python3
"""
统一评估入口 - 策略模式
使用 src.evaluation.Evaluator + Agent 接口
SCI 格式: Return, Constraint Violation Rate, Survival Rate
多种子，保存 CSV 供 notebooks 画 Return vs Cost 图
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _print_block(results: dict) -> None:
    for name, m in results.items():
        line = (
            f"{name:12} Return={m['return_mean']:.2f}±{m['return_std']:.2f}  "
            f"Violation={m['constraint_violation_rate_pct']:.1f}%  Survival={m['survival_pct']:.0f}%  "
            f"TumorClear={m['tumor_clear_pct']:.0f}%  Dose={m['avg_dose']:.2f}"
        )
        if "survival_steps_mean" in m:
            line += f"  surv_steps={m['survival_steps_mean']:.1f}±{m['survival_steps_std']:.1f}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=None,
                        help="e.g. Expert BC SafeCQL CQL. Default: all available")
    parser.add_argument("--n_ep", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Multi-seed for SCI (单一种子不被认可)")
    parser.add_argument("--ood", action="store_true", help="OOD: randomize patient params")
    parser.add_argument(
        "--cohort",
        action="store_true",
        help="每局从虚拟队列随机抽样 patient_ctx（异质性 + survival_steps / frac_* 用于 KM）",
    )
    parser.add_argument(
        "--cohort-id",
        nargs="+",
        default=None,
        metavar="ID",
        help="虚拟临床试验：固定亚群 patient_ctx（可多个，如 elderly_frail young_strong）。与 --cohort 互斥。",
    )
    parser.add_argument(
        "--safe-cql-ckpt",
        default=None,
        help="SafeCQL 权重路径（默认 checkpoints/safe_cql_limit0.1_seed42.pt）",
    )
    parser.add_argument("--output", "-o", default="results/eval_results.csv")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    if args.cohort and args.cohort_id:
        parser.error("请只使用 --cohort 或 --cohort-id 之一")

    from env.patient_cohorts import COHORT_IDS, PatientGenerator
    from src.evaluation import Evaluator, build_agents
    from src.evaluation import PyTorchAgent

    agents = build_agents(ROOT)
    ckpt = args.safe_cql_ckpt or str(ROOT / "checkpoints" / "safe_cql_limit0.1_seed42.pt")
    if Path(ckpt).exists():
        agents["SafeCQL"] = PyTorchAgent(ckpt, "safe_cql")
    if args.policies:
        agents = {k: v for k, v in agents.items() if k in args.policies}

    evaluator = Evaluator()

    if args.cohort_id:
        bad = [c for c in args.cohort_id if c not in COHORT_IDS]
        if bad:
            parser.error(f"未知 cohort: {bad}，可选: {list(COHORT_IDS)}")

        all_rows = []
        out_base = Path(args.output)
        for cid in args.cohort_id:
            gen = PatientGenerator(rng=np.random.default_rng(42))
            ctx = gen.from_cohort(cid, jitter=0.0)
            print(f"\n=== Virtual trial cohort: {cid}  ({ctx.get('label', '')}) ===")
            results = evaluator.evaluate_all(
                agents,
                n_episodes=args.n_ep,
                seeds=args.seeds,
                randomize_patient=False,
                patient_ctx=ctx,
                cohort_sample=False,
            )
            _print_block(results)
            for pol, m in results.items():
                row = {"cohort": cid, "policy": pol}
                row.update(m)
                all_rows.append(row)

        if not args.no_csv and all_rows:
            out_base.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["cohort", "policy"] + [
                k for k in all_rows[0].keys() if k not in ("cohort", "policy")
            ]
            for r in all_rows[1:]:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(out_base, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
                w.writeheader()
                w.writerows(all_rows)
            print(f"\nSaved {out_base.resolve()}")
        return

    results = evaluator.evaluate_all(
        agents,
        n_episodes=args.n_ep,
        seeds=args.seeds,
        randomize_patient=args.ood,
        cohort_sample=args.cohort,
    )

    _print_block(results)

    if not args.no_csv and results:
        Evaluator.save_csv(results, args.output)


if __name__ == "__main__":
    main()
