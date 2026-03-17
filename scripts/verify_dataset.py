#!/usr/bin/env python3
"""验证 offline_dataset.npz 包含 c 和 timeout 字段"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="offline_dataset.npz", help="Path to .npz file")
    args = p.parse_args()
    path = Path(args.path) if Path(args.path).is_absolute() else ROOT / args.path
    if not path.exists():
        print(f"文件不存在: {path}")
        print("运行: python scripts/generate_data.py -o offline_dataset.npz")
        return 1
    d = __import__("numpy").load(path)
    print("Keys:", list(d.files))
    ok = True
    if "c" not in d.files and "costs" not in d.files:
        print("[FAIL] 缺少 c/costs 列，Safe CQL 无法训练")
        ok = False
    else:
        c = d["c"] if "c" in d else d["costs"]
        print(f"[OK] Cost 违规率: {c.mean()*100:.2f}% (理想 5%-15%)")
    if "timeout" not in d.files and "timeouts" not in d.files:
        print("[FAIL] 缺少 timeout 列")
        ok = False
    else:
        t = d["timeout"] if "timeout" in d else d["timeouts"]
        print(f"[OK] timeout 已保存 (步数: {t.sum()})")
    if "done" not in d.files and "terminals" not in d.files:
        print("[FAIL] 缺少 done/terminals")
        ok = False
    else:
        term = d["done"] if "done" in d else d["terminals"]
        print(f"[OK] done/terminals 已保存 (步数: {term.sum()})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
