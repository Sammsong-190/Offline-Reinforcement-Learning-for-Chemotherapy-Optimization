#!/usr/bin/env python3
"""
实验 C：逐 episode 记录 Q_C(s,a) 均值 vs 真实违规率，供 mismatch 图。
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="SafeCQL .pt")
    ap.add_argument("-o", "--output", default="results/qc_mismatch_episodes.csv")
    ap.add_argument("--n-ep", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()

    from src.evaluation import Evaluator, PyTorchAgent

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        print(f"Not found: {ckpt}")
        return 1

    agent = PyTorchAgent(str(ckpt), "safe_cql")
    ev = Evaluator()
    rows = ev.episode_rollouts(agent, n_episodes=args.n_ep, base_seed=args.base_seed)

    out = []
    for m in rows:
        out.append({
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
