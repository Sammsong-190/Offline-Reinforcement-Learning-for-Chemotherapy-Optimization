#!/usr/bin/env python3
"""
将多份离线训练数据（generate_data 产出的 .npz）按 transition 维拼接为一份。
键与 data/generate.py 的 save_dataset 一致，供 train.py 直接使用。

例:
  python scripts/merge_offline_datasets.py \\
    -o data/raw/offline_dataset_merged.npz \\
    data/raw/offline_dataset_v3.npz data/raw/offline_dataset_high_reward.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACTION_SPACE_DEFAULT = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)


def _load_one(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    files = set(d.files)
    n = len(d["s"]) if "s" in files else len(d["observations"])
    if "observations" in files:
        # D4RL 格式 -> 与原生对齐的临时 dict
        return {
            "_path": str(path),
            "format": "d4rl",
            "n": n,
            "s": np.asarray(d["observations"], dtype=np.float32),
            "a": np.asarray(d["actions"]).ravel().astype(np.int64),
            "r": np.asarray(d["rewards"], dtype=np.float32),
            "c": np.asarray(d["costs"], dtype=np.float32) if "costs" in files else np.zeros(n, dtype=np.float32),
            "s_next": np.asarray(d["next_observations"], dtype=np.float32),
            "done": np.asarray(d["terminals"], dtype=bool),
            "timeout": np.asarray(d["timeouts"], dtype=bool) if "timeouts" in files else np.zeros(n, dtype=bool),
            "s_raw": None,
            "s_next_raw": None,
            "cohort": None,
            "action_space": np.asarray(d.get("action_space", ACTION_SPACE_DEFAULT)),
        }
    return {
        "_path": str(path),
        "format": "native",
        "n": n,
        "s": np.asarray(d["s"], dtype=np.float32),
        "s_raw": np.asarray(d["s_raw"], dtype=np.float32) if "s_raw" in files else None,
        "a": np.asarray(d["a"]).ravel().astype(np.int64),
        "r": np.asarray(d["r"], dtype=np.float32),
        "c": np.asarray(d["c"], dtype=np.float32) if "c" in files else np.zeros(n, dtype=np.float32),
        "s_next": np.asarray(d["s_next"], dtype=np.float32),
        "s_next_raw": np.asarray(d["s_next_raw"], dtype=np.float32) if "s_next_raw" in files else None,
        "done": np.asarray(d["done"], dtype=bool),
        "timeout": np.asarray(d["timeout"], dtype=bool) if "timeout" in files else np.zeros(n, dtype=bool),
        "cohort": np.asarray(d["cohort"], dtype="U64") if "cohort" in files else None,
        "action_space": np.asarray(d["action_space"], dtype=np.float32) if "action_space" in files else ACTION_SPACE_DEFAULT.copy(),
    }


def _fill_missing_s_raw(blocks):
    for b in blocks:
        if b["s_raw"] is None:
            b["s_raw"] = np.array(b["s"], copy=True)
        if b["s_next_raw"] is None:
            b["s_next_raw"] = np.array(b["s_next"], copy=True)


def _fill_missing_cohort(blocks):
    for b in blocks:
        n = b["n"]
        if b["cohort"] is None:
            b["cohort"] = np.array(["default"] * n, dtype="U32")


def merge(blocks: list) -> dict:
    _fill_missing_s_raw(blocks)
    _fill_missing_cohort(blocks)
    a0 = blocks[0]["action_space"]
    for b in blocks[1:]:
        if not np.allclose(b["action_space"], a0):
            print("警告: action_space 与第一份不一致，已用第一份覆盖校验", file=sys.stderr)
    out = {
        "s": np.concatenate([b["s"] for b in blocks], axis=0),
        "s_raw": np.concatenate([b["s_raw"] for b in blocks], axis=0),
        "a": np.concatenate([b["a"] for b in blocks], axis=0),
        "r": np.concatenate([b["r"] for b in blocks], axis=0),
        "c": np.concatenate([b["c"] for b in blocks], axis=0),
        "s_next": np.concatenate([b["s_next"] for b in blocks], axis=0),
        "s_next_raw": np.concatenate([b["s_next_raw"] for b in blocks], axis=0),
        "done": np.concatenate([b["done"] for b in blocks], axis=0),
        "timeout": np.concatenate([b["timeout"] for b in blocks], axis=0),
        "cohort": np.concatenate(
            [np.asarray(b["cohort"], dtype="U32") for b in blocks], axis=0
        ),
        "action_space": np.array(a0, dtype=np.float32),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="+",
        help="多份 .npz（项目内相对路径或绝对路径）",
    )
    ap.add_argument(
        "-o", "--output",
        required=True,
        help="合并后的 .npz，如 data/raw/offline_dataset_merged.npz",
    )
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        pth = Path(p)
        if not pth.is_absolute():
            pth = ROOT / pth
        if not pth.exists():
            print(f"找不到文件: {pth}", file=sys.stderr)
            return 1
        paths.append(pth)

    blocks = [_load_one(p) for p in paths]
    for b in blocks:
        print(f"  {b['_path']}: {b['n']} transitions  ({b['format']})")

    merged = merge(blocks)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        s=merged["s"],
        s_raw=merged["s_raw"],
        a=merged["a"],
        r=merged["r"],
        c=merged["c"],
        s_next=merged["s_next"],
        s_next_raw=merged["s_next_raw"],
        done=merged["done"],
        timeout=merged["timeout"],
        cohort=merged["cohort"],
        action_space=merged["action_space"],
    )
    n = len(merged["s"])
    print(f"合并完成: {n} transitions -> {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
