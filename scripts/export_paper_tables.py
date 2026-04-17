#!/usr/bin/env python3
"""
导出论文用 Table 1（虚拟队列特征）与 Table 2（主实验结果，来自 evaluate 导出的 CSV）。

Table 1: ODE 参数（合并后）、i_safe, n_safe, c_tox, t_fatal, SDE σ
Table 2: 宽表 — 行=算法，列=各 cohort 的 Survival, Steps, Dose, Return, dominant outcome

Usage:
  python scripts/export_paper_tables.py
  python scripts/export_paper_tables.py --eval-csv results/vtrial_all_algos.csv --out-dir tables
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _merge_cohort_table():
    from env.chemo_env import DEFAULT_PARAMS, C_TOX, I_SAFE, N_SAFE, T_FATAL
    from env.patient_cohorts import COHORT_IDS, PatientGenerator

    pg = PatientGenerator()
    rows = []
    for cid in COHORT_IDS:
        ctx = pg.from_cohort(cid, jitter=0.0)
        p = ctx["params"]
        rows.append(
            {
                "cohort_id": cid,
                "label": ctx.get("label", ""),
                "s": f"{p['s']:.4f}",
                "r1": f"{p['r1']:.4f}",
                "r2": f"{p['r2']:.4f}",
                "b1": f"{p['b1']:.4f}",
                "a1": f"{p['a1']:.4f}",
                "a2": f"{p['a2']:.4f}",
                "c_tox": f"{ctx['c_tox']:.4f}",
                "i_safe": f"{ctx['i_safe']:.4f}",
                "n_safe": f"{ctx['n_safe']:.4f}",
                "t_fatal": f"{ctx['t_fatal']:.4f}",
                "sde_sigma": f"{ctx['sde_sigma']:.4f}",
            }
        )
    rows.append(
        {
            "cohort_id": "reference",
            "label": "DEFAULT (no cohort)",
            "s": f"{DEFAULT_PARAMS['s']:.4f}",
            "r1": f"{DEFAULT_PARAMS['r1']:.4f}",
            "r2": f"{DEFAULT_PARAMS['r2']:.4f}",
            "b1": f"{DEFAULT_PARAMS['b1']:.4f}",
            "a1": f"{DEFAULT_PARAMS['a1']:.4f}",
            "a2": f"{DEFAULT_PARAMS['a2']:.4f}",
            "c_tox": f"{C_TOX:.4f}",
            "i_safe": f"{I_SAFE:.4f}",
            "n_safe": f"{N_SAFE:.4f}",
            "t_fatal": f"{T_FATAL:.4f}",
            "sde_sigma": "0.0000",
        }
    )
    return rows


def _dominant_reason(row: dict) -> str:
    best_k, best_v = "", -1.0
    for k, v in row.items():
        if not k.startswith("frac_"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > best_v:
            best_v = fv
            best_k = k.replace("frac_", "")
    return best_k if best_k else ""


def _load_eval_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wide_results(rows: list[dict]) -> tuple[list[str], list[dict]]:
    cohorts = sorted({r["cohort"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    keys = [
        ("survival_pct", "Survival_pct"),
        ("survival_steps_mean", "Steps_mean"),
        ("survival_steps_std", "Steps_std"),
        ("avg_dose", "Dose"),
        ("return_mean", "Return_mean"),
        ("return_std", "Return_std"),
        ("constraint_violation_rate_pct", "Viol_pct"),
    ]
    fieldnames = ["policy"]
    for c in cohorts:
        for _, short in keys:
            fieldnames.append(f"{c}_{short}")
        fieldnames.append(f"{c}_Outcome")

    out_rows = []
    for pol in policies:
        out: dict = {"policy": pol}
        for c in cohorts:
            sub = next((r for r in rows if r["policy"] == pol and r["cohort"] == c), None)
            if not sub:
                for _, short in keys:
                    out[f"{c}_{short}"] = ""
                out[f"{c}_Outcome"] = ""
                continue
            for csv_key, short in keys:
                out[f"{c}_{short}"] = sub.get(csv_key, "")
            out[f"{c}_Outcome"] = _dominant_reason(sub)
        out_rows.append(out)
    return fieldnames, out_rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _latex_table1(rows: list[dict]) -> str:
    cols = [
        "cohort_id",
        "s",
        "r1",
        "b1",
        "a1",
        "a2",
        "c_tox",
        "i_safe",
        "n_safe",
        "t_fatal",
        "sde_sigma",
    ]
    lines = [
        r"\begin{tabular}{lcccccccccc}",
        r"\hline",
        r"Cohort & $s$ & $r_1$ & $b_1$ & $a_1$ & $a_2$ & $c_{\mathrm{tox}}$ & $I_{\mathrm{safe}}$ & $N_{\mathrm{safe}}$ & $T_{\mathrm{fatal}}$ & $\sigma_{\mathrm{SDE}}$ \\",
        r"\hline",
    ]
    for r in rows:
        if r["cohort_id"] == "reference":
            continue
        line = " & ".join(
            [
                r["cohort_id"].replace("_", r"\_"),
                r["s"],
                r["r1"],
                r["b1"],
                r["a1"],
                r["a2"],
                r["c_tox"],
                r["i_safe"],
                r["n_safe"],
                r["t_fatal"],
                r["sde_sigma"],
            ]
        )
        lines.append(line + r" \\")
    lines.append(r"\hline\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tables")
    ap.add_argument("--eval-csv", type=Path, default=ROOT / "results" / "vtrial_all_algos.csv")
    args = ap.parse_args()
    out_dir = args.out_dir

    t1 = _merge_cohort_table()
    fn1 = list(t1[0].keys())
    _write_csv(out_dir / "table1_cohort_characteristics.csv", fn1, t1)
    tex1 = out_dir / "table1_cohort_characteristics.tex"
    tex1.write_text(_latex_table1(t1), encoding="utf-8")
    print(f"Wrote {out_dir / 'table1_cohort_characteristics.csv'}")
    print(f"Wrote {tex1}")

    eval_path = args.eval_csv
    if not eval_path.exists():
        print(f"Skip Table 2: {eval_path} not found (run evaluate.py first).", file=sys.stderr)
        return 0

    rows = _load_eval_rows(eval_path)
    if not rows:
        print("Eval CSV empty", file=sys.stderr)
        return 1

    fn2, wide = _wide_results(rows)
    _write_csv(out_dir / "table2_main_results_wide.csv", fn2, wide)

    long_fn = [
        "cohort",
        "policy",
        "survival_pct",
        "survival_steps_mean",
        "survival_steps_std",
        "avg_dose",
        "return_mean",
        "return_std",
        "constraint_violation_rate_pct",
        "dominant_reason",
    ]
    long_rows = []
    for r in rows:
        long_rows.append(
            {
                "cohort": r.get("cohort", ""),
                "policy": r.get("policy", ""),
                "survival_pct": r.get("survival_pct", ""),
                "survival_steps_mean": r.get("survival_steps_mean", ""),
                "survival_steps_std": r.get("survival_steps_std", ""),
                "avg_dose": r.get("avg_dose", ""),
                "return_mean": r.get("return_mean", ""),
                "return_std": r.get("return_std", ""),
                "constraint_violation_rate_pct": r.get("constraint_violation_rate_pct", ""),
                "dominant_reason": _dominant_reason(r),
            }
        )
    _write_csv(out_dir / "table2_main_results_long.csv", long_fn, long_rows)

    print(f"Wrote {out_dir / 'table2_main_results_wide.csv'}")
    print(f"Wrote {out_dir / 'table2_main_results_long.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
