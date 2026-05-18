#!/usr/bin/env python3
"""
在 held-out TCGA digital twins 上评估 BC / CQL /（可选）SafeCQL 单 checkpoint。
划分与 build_tcga_twin_dataset.py / aggregate_eval_multi_seed --tcga-clinical 一致。

用法:
  export CLINICAL=/path/to/clinical.tsv
  .venv/bin/python scripts/eval_tcga_baselines.py
  .venv/bin/python scripts/eval_tcga_baselines.py --clinical "$CLINICAL" -o results/tcga_twin_baseline_runs.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from env.tcga_twins import load_tcga_twins_from_clinical, train_eval_split_twins
except ImportError as e:
    print(
        "Cannot import env.tcga_twins (expected alongside env/tcga_mapper.py). "
        f"Detail: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

from src.evaluation import D3RLPyAgent, Evaluator, PyTorchAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clinical",
        type=Path,
        default=None,
        help="TCGA clinical.tsv；默认环境变量 CLINICAL",
    )
    ap.add_argument("--train-frac", type=float, default=0.72)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--base-seed", type=int, default=42, help="twin rollouts RNG base")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "results/tcga_twin_baseline_runs.csv")
    ap.add_argument(
        "--safe-cql-ckpt",
        type=Path,
        default=ROOT / "checkpoints/safe_cql_limit0.1_seed15.pt",
        help="可选对照 SafeCQL；不存在则跳过",
    )
    ap.add_argument("--bc-path", type=Path, default=ROOT / "checkpoints/bc_tcga_twin.pt")
    ap.add_argument("--cql-path", type=Path, default=ROOT / "checkpoints/cql_tcga_twin.d3")
    args = ap.parse_args()

    clinical = args.clinical or Path(os.environ.get("CLINICAL", ""))
    if not clinical or not clinical.is_file():
        print(
            f"Missing clinical.tsv: {clinical or '(set --clinical or CLINICAL)'}",
            file=sys.stderr,
        )
        return 1

    if not hasattr(Evaluator, "evaluate_agent_on_twins"):
        print(
            "Evaluator missing evaluate_agent_on_twins; upgrade src/evaluation.py",
            file=sys.stderr,
        )
        return 1

    twins = load_tcga_twins_from_clinical(clinical)
    _, eval_twins = train_eval_split_twins(twins, args.train_frac, args.split_seed)
    if not eval_twins:
        print("Empty eval twin list.", file=sys.stderr)
        return 1

    ev = Evaluator()
    rows = []

    def add(name: str, agent):
        m = ev.evaluate_agent_on_twins(agent, eval_twins, base_seed=args.base_seed)
        rows.append(
            {
                "policy": name,
                "return_mean": m["return_mean"],
                "return_std": m["return_std"],
                "avg_dose": m["avg_dose"],
                "constraint_violation_rate_pct": m["constraint_violation_rate_pct"],
                "n_eval_patients": m["n_eval_patients"],
                "eval_mode": "tcga_heldout",
            }
        )

    specs = [
        ("BC", args.bc_path, "bc"),
        ("CQL", args.cql_path, "cql"),
    ]
    for name, path, kind in specs:
        p = path if path.is_absolute() else ROOT / path
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        if kind == "cql":
            try:
                add(name, D3RLPyAgent(str(p)))
            except Exception as ex:
                print(f"[skip] CQL load/eval failed {p}: {ex}", file=sys.stderr)
        else:
            add(name, PyTorchAgent(str(p), kind))

    sck = args.safe_cql_ckpt if args.safe_cql_ckpt.is_absolute() else ROOT / args.safe_cql_ckpt
    if sck.exists():
        add(f"SafeCQL_{sck.stem}", PyTorchAgent(str(sck), "safe_cql"))
    else:
        print(f"[skip] missing optional SafeCQL {sck}")

    if not rows:
        print("No checkpoints evaluated.", file=sys.stderr)
        return 1

    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out} ({len(rows)} policies)")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
