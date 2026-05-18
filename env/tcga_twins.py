"""
从 TCGA clinical.tsv 构建 twin 列表，并与 build_tcga_twin_dataset 使用相同的 train/eval 划分逻辑。
用于 held-out digital twin 上评估策略，而非默认 X0 / DEFAULT_PARAMS。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from env.tcga_mapper import twin_from_row


def load_tcga_twins_from_clinical(clinical_path: Path) -> List[Dict[str, Any]]:
    """返回与数据构建脚本一致的 dict 列表：case_id, x0, patient_ctx, meta。"""
    path = Path(clinical_path)
    if not path.is_file():
        raise FileNotFoundError(f"clinical tsv not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df = df.drop_duplicates("cases.case_id", keep="first")
    twins: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        result = twin_from_row(rd)
        if result is None:
            continue
        x0, patient_ctx, meta = result
        twins.append(
            {
                "case_id": rd.get("cases.case_id"),
                "x0": x0,
                "patient_ctx": patient_ctx,
                "meta": meta,
            }
        )
    return twins


def train_eval_split_twins(
    twins: List[Dict[str, Any]], train_frac: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """与 scripts/build_tcga_twin_dataset.py 相同的 patient-level 划分。"""
    if not twins:
        return [], []
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(twins))
    n_train = int(len(twins) * float(train_frac))
    train = [twins[i] for i in idx[:n_train]]
    eval_twins = [twins[i] for i in idx[n_train:]]
    return train, eval_twins
