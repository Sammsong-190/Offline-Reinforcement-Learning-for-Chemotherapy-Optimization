#!/usr/bin/env python3
"""
将 checkpoints/ 下训练产物扫描并整合为单一 manifest（JSON + CSV）。
含: SafeCQL .pt、对应 _lambda.json、CQL .d3、bc_policy 等。
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SAFE_RE = re.compile(r"safe_cql_limit([0-9.]+)_seed(\d+)\.pt$")


def _scan(ckpt_dir: Path) -> list:
    rows = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_lambda.json") and "safe_cql" in name:
            # 已由对应 .pt 条目的 lambda_log 引用，不重复列一行
            continue
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)
        base = {
            "file": rel,
            "name": name,
            "size_bytes": st.st_size,
            "modified_utc": mtime,
        }
        m = SAFE_RE.search(name)
        if m:
            base["kind"] = "safe_cql"
            base["cost_limit"] = float(m.group(1))
            base["seed"] = int(m.group(2))
            lam = p.with_name(p.stem + "_lambda.json")
            try:
                base["lambda_log"] = str(lam.relative_to(ROOT)) if lam.exists() else None
            except ValueError:
                base["lambda_log"] = str(lam) if lam.exists() else None
        elif name.endswith(".d3"):
            base["kind"] = "cql_d3rlpy"
        elif "bc_policy" in name and name.endswith(".pt"):
            base["kind"] = "bc"
        else:
            base["kind"] = "other"
        rows.append(base)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints", help="相对项目根或绝对路径")
    ap.add_argument("--json", default="checkpoints/manifest.json")
    ap.add_argument("--csv", default="checkpoints/manifest.csv")
    args = ap.parse_args()

    ck = Path(args.dir)
    if not ck.is_absolute():
        ck = ROOT / ck
    if not ck.is_dir():
        print(f"Not a directory: {ck}", file=sys.stderr)
        return 1

    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ck.resolve()),
        "n_files": 0,
        "runs": [],
    }
    for row in _scan(ck):
        data["runs"].append(row)
    data["n_files"] = len(data["runs"])

    jpath = Path(args.json)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    cpath = Path(args.csv)
    if not cpath.is_absolute():
        cpath = ROOT / cpath
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {jpath} ({data['n_files']} entries)")

    # 扁平 CSV：仅适合表格化的列
    if data["runs"]:
        fieldnames = [
            "kind", "name", "cost_limit", "seed", "size_bytes", "modified_utc",
            "lambda_log", "file",
        ]
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in data["runs"]:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote {cpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
