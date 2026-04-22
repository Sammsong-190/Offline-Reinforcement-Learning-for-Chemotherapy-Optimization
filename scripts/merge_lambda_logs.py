#!/usr/bin/env python3
"""
将 checkpoints/ 下所有 SafeCQL 的 *_lambda.json 合并为一个 JSON：
  每档训练一条曲线（cost_limit + seed + history）。

输出结构示例:
{
  "meta": { "generated_utc": "...", "n_runs": N },
  "runs": [
    {
      "id": "ε=0.0_seed123",
      "source_file": "checkpoints/safe_cql_limit0.0_seed123_lambda.json",
      "cost_limit": 0.0,
      "seed": 123,
      "history": [ { "step": 1000, "lambda": ..., ... }, ... ]
    },
    ...
  ]
}

用法:
  python scripts/merge_lambda_logs.py
  python scripts/merge_lambda_logs.py -o results/lambda_all_runs.json --dir checkpoints
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FNAME_RE = re.compile(
    r"safe_cql_limit([0-9.]+)_seed(\d+)_lambda\.json$"
)


def parse_name(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints", help="扫描目录（相对项目根）")
    ap.add_argument(
        "-o",
        "--output",
        default="checkpoints/lambda_all_runs.json",
        help="合并后的单文件路径",
    )
    args = ap.parse_args()

    ck = Path(args.dir)
    if not ck.is_absolute():
        ck = ROOT / ck
    if not ck.is_dir():
        print(f"Not a directory: {ck}", file=sys.stderr)
        return 1

    runs = []
    for p in sorted(ck.glob("safe_cql_limit*_seed*_lambda.json")):
        eps, seed = parse_name(p.name)
        if eps is None:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skip (invalid JSON): {p} ({e})", file=sys.stderr)
            continue
        hist = data.get("history", [])
        cl = data.get("cost_limit", eps)
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)
        run_id = f"ε={cl}_seed{seed}"
        runs.append({
            "id": run_id,
            "source_file": rel,
            "cost_limit": cl,
            "seed": seed,
            "n_points": len(hist),
            "history": hist,
        })

    if not runs:
        print(f"No safe_cql_*_lambda.json under {ck}", file=sys.stderr)
        return 1

    out = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_runs": len(runs),
            "description": "Merged Lagrange λ / Q_C / losses logs from SafeCQL training",
        },
        "runs": sorted(runs, key=lambda r: (r["cost_limit"], r["seed"])),
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path.resolve()} ({len(runs)} runs, {sum(r['n_points'] for r in runs)} total log points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
