#!/usr/bin/env python3
"""
实验 C：逐条 rollout 记录 Q_C(s,a) 均值 vs 真实逐步违规率，供 calibration / mismatch 图。

默认：同一 patient_ctx / X0 且动力学确定性时，多 episode 可能完全重复（CSV 各列相同），
此时应使用 --tcga-clinical 在 held-out digital twin 上每人一条轨迹。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="SafeCQL .pt（须含 q_c）")
    ap.add_argument("-o", "--output", default="results/qc_mismatch_episodes.csv")
    ap.add_argument("--n-ep", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument(
        "--tcga-clinical",
        type=Path,
        default=None,
        help="clinical.tsv：与 aggregate_eval_multi_seed.py 相同 train/eval 划分，"
        "输出每人一行（held-out twin）",
    )
    ap.add_argument("--train-frac", type=float, default=0.72)
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="与 build_tcga_twin_dataset.py / aggregate_eval_multi_seed.py 一致",
    )
    ap.add_argument(
        "--cohort-id",
        default=None,
        metavar="ID",
        help="young_strong | elderly_frail | refractory_tumor；缺省为 default 动力学",
    )
    args = ap.parse_args()

    from src.evaluation import Evaluator, PyTorchAgent
    from env.patient_cohorts import COHORT_IDS, PatientGenerator

    if args.cohort_id is not None and args.tcga_clinical is not None:
        print("请勿同时指定 --cohort-id 与 --tcga-clinical", file=sys.stderr)
        return 1

    if args.cohort_id is not None and args.cohort_id not in COHORT_IDS:
        print(f"未知 cohort: {args.cohort_id}，可选: {COHORT_IDS}", file=sys.stderr)
        return 1

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        print(f"Not found: {ckpt}")
        return 1

    agent = PyTorchAgent(str(ckpt), "safe_cql")
    ev = Evaluator()

    if args.tcga_clinical is not None:
        from env.tcga_twins import load_tcga_twins_from_clinical, train_eval_split_twins

        twins_all = load_tcga_twins_from_clinical(Path(args.tcga_clinical))
        _train, eval_twins = train_eval_split_twins(
            twins_all, args.train_frac, args.split_seed
        )
        if not eval_twins:
            print("TCGA split produced empty eval set; check train-frac / clinical path.")
            return 1
        rows = ev.mismatch_rollouts_on_twins(
            agent, eval_twins, base_seed=args.base_seed
        )
        eval_mode = "tcga_heldout"
        cohort_label = "tcga_heldout"
        print(
            f"[TCGA] {len(eval_twins)} held-out patients "
            f"(train_frac={args.train_frac}, split_seed={args.split_seed})"
        )
    else:
        patient_ctx = None
        if args.cohort_id is not None:
            gen = PatientGenerator(rng=np.random.default_rng(42))
            patient_ctx = gen.from_cohort(args.cohort_id, jitter=0.0)
        rows = ev.episode_rollouts(
            agent,
            n_episodes=args.n_ep,
            base_seed=args.base_seed,
            patient_ctx=patient_ctx,
        )
        eval_mode = "default_x0" if args.cohort_id is None else f"cohort_{args.cohort_id}"
        cohort_label = args.cohort_id or "default"

    out = []
    for m in rows:
        out.append({
            "eval_mode": eval_mode,
            "cohort": cohort_label,
            "case_id": m.get("case_id", ""),
            "episode": m["episode"],
            "return": m["return"],
            "mean_qc_predicted": m.get("mean_qc_predicted", ""),
            "max_qc_predicted": m.get("max_qc_predicted", ""),
            "true_cost_rate": m.get("true_cost_rate", ""),
            "constraint_violation_rate_pct": m["constraint_violation_rate_pct"],
            "avg_dose": m["avg_dose"],
        })

    outp = ROOT / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Saved {outp} ({len(out)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
