"""
虚拟患者队列 (Virtual Cohorts)：亚群特异动力学与安全阈值，用于数据生成与评估。

Cohort 1 — 年轻/强壮：免疫恢复 s↑，毒性耐受 C_TOX↑
Cohort 2 — 老年/虚弱：s↓，器官阈值更严（N_SAFE↑ 表示更早判违规）
Cohort 3 — 难治型肿瘤：r1↑，免疫介导杀伤 c2↓
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from env.chemo_env import DEFAULT_PARAMS, C_TOX, I_SAFE, N_SAFE, T_FATAL

COHORT_IDS = ("young_strong", "elderly_frail", "refractory_tumor")

# 相对 DEFAULT_PARAMS 的乘数或绝对覆盖（merge 后裁剪）
_COHORT_SPECS: Dict[str, Dict[str, Any]] = {
    "young_strong": {
        "label": "Cohort1 Young/Strong",
        "params_mult": {
            "s": 1.35,
            "r1": 1.12,
            # K≈1/b1：0.95 → 容纳量略高于 1，仍可超过 t_fatal；比 0.5/0.8 温和，避免 episode 过短、数据占比过低
            "b1": 0.95,
            "a1": 0.88,
            "a2": 0.90,
            "d1": 0.92,
        },
        "c_tox": C_TOX * 1.25,
        "i_safe": I_SAFE * 0.95,
        "n_safe": N_SAFE * 0.95,
        "t_fatal": 0.98,
        "sde_sigma": 0.012,
    },
    "elderly_frail": {
        "label": "Cohort2 Elderly/Frail",
        "params_mult": {
            "s": 0.72,
            "a1": 1.12,
            "a2": 1.10,
            "d1": 1.08,
        },
        "c_tox": C_TOX * 0.88,
        "i_safe": I_SAFE * 1.1,
        "n_safe": N_SAFE * 1.1,
        "t_fatal": 1.2,
        "sde_sigma": 0.02,
    },
    "refractory_tumor": {
        "label": "Cohort3 Refractory Tumor",
        "params_mult": {
            "r1": 1.38,
            "c2": 0.62,
            "c3": 0.85,
            "b1": 0.95,
        },
        "c_tox": C_TOX,
        "i_safe": I_SAFE,
        "n_safe": N_SAFE,
        "t_fatal": T_FATAL,
        "sde_sigma": 0.015,
    },
}


def _merge_params(base: Optional[Dict], mult: Dict[str, float]) -> Dict[str, float]:
    p = copy.deepcopy(base or DEFAULT_PARAMS)
    for k, m in mult.items():
        if k in p:
            p[k] = float(np.clip(p[k] * m, 1e-8, 50.0))
    return p


class PatientGenerator:
    """按亚群生成患者参数字典 + 安全阈值 + 建议 SDE 噪声强度。"""

    def __init__(self, base_params: Optional[Dict] = None, rng: Optional[np.random.Generator] = None):
        self.base_params = base_params or DEFAULT_PARAMS
        self.rng = rng or np.random.default_rng()

    @staticmethod
    def cohort_ids() -> List[str]:
        return list(COHORT_IDS)

    def from_cohort(self, cohort_id: str, jitter: float = 0.0) -> Dict[str, Any]:
        """返回 patient_ctx，可直接传给 collect_trajectory(..., patient_ctx=...)."""
        if cohort_id not in _COHORT_SPECS:
            raise ValueError(
                f"Unknown cohort {cohort_id!r}; choose from {COHORT_IDS}")
        spec = _COHORT_SPECS[cohort_id]
        params = _merge_params(self.base_params, spec["params_mult"])
        if jitter > 0:
            params = _jitter_params(params, self.rng, scale=jitter)
        return {
            "cohort": cohort_id,
            "label": spec["label"],
            "params": params,
            "c_tox": float(spec["c_tox"]),
            "i_safe": float(spec["i_safe"]),
            "n_safe": float(spec["n_safe"]),
            "t_fatal": float(spec.get("t_fatal", T_FATAL)),
            "sde_sigma": float(spec["sde_sigma"]),
        }

    def sample(
        self,
        weights: Optional[List[float]] = None,
        jitter: float = 0.0,
    ) -> Dict[str, Any]:
        """按权重随机抽一个亚群。"""
        w = weights if weights is not None else [
            1.0 / len(COHORT_IDS)] * len(COHORT_IDS)
        w = np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        cid = self.rng.choice(list(COHORT_IDS), p=w)
        return self.from_cohort(cid, jitter=jitter)


def _jitter_params(params: Dict, rng: np.random.Generator, scale: float = 0.05) -> Dict[str, float]:
    """亚群内在小扰动（可选）。"""
    p = copy.deepcopy(params)
    keys = [k for k in p if isinstance(p[k], (int, float, np.floating))]
    for k in keys:
        noise = rng.normal(1.0, scale)
        p[k] = float(np.clip(p[k] * noise, 1e-8, 50.0))
    return p
