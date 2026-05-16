"""
Map TCGA clinical rows to individualized ODE initial conditions and patient_ctx.

Age / stage / grade do not enter the agent state; they affect x0 and transition dynamics only.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from env.chemo_env import DEFAULT_PARAMS


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    low = s.lower()
    if low in ("nan", "none"):
        return True
    if s in ("--", "'--"):
        return True
    if "not applicable" in low or "not reported" in low:
        return True
    return False


def parse_age(row: Dict[str, Any]) -> Optional[float]:
    idx = row.get("demographic.age_at_index")
    if not _is_missing(idx):
        try:
            return float(idx)
        except (TypeError, ValueError):
            pass
    dob = row.get("demographic.days_to_birth")
    if _is_missing(dob):
        return None
    try:
        return abs(float(dob)) / 365.25
    except (TypeError, ValueError):
        return None


_ROMAN_STAGE_ORDER = [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]


def _stage_string_to_int(stage_raw: str) -> Optional[int]:
    s = stage_raw.strip()
    if _is_missing(s):
        return None
    tail = re.sub(r"(?is)^.*?stage\s*", "", s).strip()
    if not tail:
        return None
    u = tail.upper()
    if re.match(r"^[0-4]$", u):
        return int(u)
    for roman, n in _ROMAN_STAGE_ORDER:
        if u.startswith(roman):
            return n
    return None


def parse_stage(row: Dict[str, Any]) -> Optional[int]:
    for key in (
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_clinical_stage",
    ):
        raw = row.get(key)
        if _is_missing(raw):
            continue
        st = _stage_string_to_int(str(raw))
        if st is not None:
            return st
    return None


_GRADE_RE = re.compile(r"G\s*([1-4])", re.I)


def parse_grade(row: Dict[str, Any]) -> int:
    raw = row.get("diagnoses.tumor_grade")
    if not _is_missing(raw):
        s = str(raw).strip().upper()
        m = _GRADE_RE.search(s)
        if m:
            return int(m.group(1))
        if s in ("1", "2", "3", "4"):
            return int(s)
        if "GRADE " in s or s.startswith("GRADE"):
            digits = re.findall(r"[1-4]", s)
            if digits:
                return int(digits[0])
    return 2


def t0_from_stage(stage: int) -> float:
    return {1: 0.10, 2: 0.30, 3: 0.60, 4: 0.85}[stage]


def i0_from_age(age: float) -> float:
    return max(0.62, 1.10 - 0.006 * max(age - 40.0, 0.0))


def n0_from_age(age: float) -> float:
    return max(0.72, 1.05 - 0.004 * max(age - 45.0, 0.0))


def c_tox_from_age(age: float) -> float:
    """Offline cohort: slightly lower cap so C-related termination aligns with heterogeneous twins."""
    return max(4.05, 7.35 * (1.0 - 0.008 * max(age - 50.0, 0.0)))


def t_fatal_from_stage(stage: int) -> float:
    return {1: 1.80, 2: 1.60, 3: 1.35, 4: 1.15}[stage]


def twin_from_row(
    row: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
    age = parse_age(row)
    stage = parse_stage(row)
    grade = parse_grade(row)

    if age is None or stage is None:
        return None

    x0 = np.array(
        [
            n0_from_age(float(age)),
            t0_from_stage(int(stage)),
            i0_from_age(float(age)),
            0.0,
        ],
        dtype=np.float32,
    )

    params = copy.deepcopy(DEFAULT_PARAMS)

    stage_factor = {1: 0.88, 2: 1.00, 3: 1.12, 4: 1.22}[stage]
    grade_factor = {1: 0.90, 2: 1.00, 3: 1.12, 4: 1.20}.get(grade, 1.0)
    params["r1"] *= stage_factor * grade_factor

    patient_ctx = {
        "params": params,
        "c_tox": c_tox_from_age(float(age)),
        # Stricter safety bands offline → enough c=1 for Safe CQL calibration (~5–25%).
        "i_safe": 0.34,
        "n_safe": 0.34,
        "t_fatal": t_fatal_from_stage(int(stage)),
        "sde_sigma": 0.01,
        "cohort": f"tcga_stage_{stage}",
    }

    meta = {"age": float(age), "stage": int(stage), "grade": int(grade)}
    return x0, patient_ctx, meta


def tcga_behavior_policy(
    stage: int, rng: np.random.Generator, epsilon: float = 0.18
) -> Callable[[np.ndarray], float]:
    """Later stages: more exploration + higher-dose bias to surface toxic / sub-threshold paths."""

    eps = epsilon + (0.12 if stage >= 3 else 0.0)
    late = stage >= 3

    def policy(s: np.ndarray) -> float:
        N, T, I, C = s[:4]

        if rng.random() < eps:
            if late and rng.random() < 0.52:
                return float(rng.choice([1.0, 2.0]))
            return float(rng.choice([0.0, 0.5, 1.0, 2.0]))

        # Tolerate more burden before holding dose (Stage III/IV).
        c_stop, n_stop, i_stop = (6.25, 0.12, 0.12) if late else (5.5, 0.18, 0.18)
        if C > c_stop or N < n_stop or I < i_stop:
            return 0.0

        if stage <= 1:
            return 0.5 if T > 0.2 else 0.0

        if stage == 2:
            return 1.0 if T > 0.35 else 0.5

        if stage == 3:
            if T > 0.48:
                return 2.0 if (I > 0.26 or rng.random() < 0.35) else 1.0
            return 1.0

        if T > 0.58:
            return 2.0 if (I > 0.22 or rng.random() < 0.4) else 1.0
        return 1.0 if T > 0.35 else 0.5

    return policy
