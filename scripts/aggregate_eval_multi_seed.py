#!/usr/bin/env python3
"""
实验 A：对多个 seed × ε 的 checkpoint 做评估，输出长表 + 按 ε 聚合的均值/标准差（跨 seed）。
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT_RE = re.compile(r"safe_cql_limit([0-9.]+)_seed(\d+)\.pt$")


def find_checkpoints(ckpt_dir: Path):
    out = []
    for p in sorted(ckpt_dir.glob("safe_cql_limit*_seed*.pt")):
        m = CKPT_RE.search(p.name)
        if m:
            out.append((float(m.group(1)), int(m.group(2)), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--data-tag", default="", help="写入 CSV 备注列")
    ap.add_argument("-o", "--output-runs", default="results/multi_seed_runs.csv")
    ap.add_argument("-a", "--output-agg", default="results/multi_seed_agg.csv")
    ap.add_argument("--n-ep", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="若省略则扫描目录中全部 seed")
    ap.add_argument("--limits", nargs="+", type=float, default=None,
                    help="若省略则扫描目录中全部 limit")
    args = ap.parse_args()

    from src.evaluation import Evaluator, PyTorchAgent

    ckpt_dir = ROOT / args.checkpoint_dir
    found = find_checkpoints(ckpt_dir)
    if not found:
        print(f"No checkpoints in {ckpt_dir}")
        return 1

    limits = set(args.limits) if args.limits else None
    seeds_f = set(args.seeds) if args.seeds else None
    rows = []
    evaluator = Evaluator()
    for lim, seed, path in found:
        if limits is not None and lim not in limits:
            continue
        if seeds_f is not None and seed not in seeds_f:
            continue
        agent = PyTorchAgent(str(path), "safe_cql")
        # 每个 checkpoint 用 n_ep 条不同子种子 rollout（避免 evaluate 里多 episode 共用同一 seed）
        sub_seeds = [seed * 10000 + k for k in range(args.n_ep)]
        m = evaluator.evaluate_agent(agent, n_episodes=1, seeds=sub_seeds)
        rows.append({
            "epsilon": lim,
            "seed": seed,
            "checkpoint": path.name,
            "return_mean": m["return_mean"],
            "return_std_episodes": m["return_std"],
            "avg_dose": m["avg_dose"],
            "constraint_violation_rate_pct": m["constraint_violation_rate_pct"],
            "data_tag": args.data_tag,
        })

    if not rows:
        print("No matching checkpoints after filters.")
        return 1

    out_runs = ROOT / args.output_runs
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_runs} ({len(rows)} runs)")

    # 按 epsilon 聚合（跨 seed）
    by_eps = {}
    for r in rows:
        e = r["epsilon"]
        by_eps.setdefault(e, []).append(r)

    agg = []
    for e in sorted(by_eps.keys()):
        grp = by_eps[e]
        rm = [x["return_mean"] for x in grp]
        ad = [x["avg_dose"] for x in grp]
        viol = [x["constraint_violation_rate_pct"] for x in grp]
        agg.append({
            "epsilon": e,
            "n_seeds": len(grp),
            "return_mean": float(np.mean(rm)),
            "return_std_across_seeds": float(np.std(rm, ddof=1)) if len(rm) > 1 else 0.0,
            "avg_dose_mean": float(np.mean(ad)),
            "avg_dose_std_across_seeds": float(np.std(ad, ddof=1)) if len(ad) > 1 else 0.0,
            "violation_pct_mean": float(np.mean(viol)),
            "violation_pct_std_across_seeds": float(np.std(viol, ddof=1)) if len(viol) > 1 else 0.0,
            "data_tag": args.data_tag,
        })

    out_agg = ROOT / args.output_agg
    with open(out_agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"Saved {out_agg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
