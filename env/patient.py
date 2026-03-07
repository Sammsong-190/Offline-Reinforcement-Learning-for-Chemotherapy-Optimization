"""
Patient parameter randomization (clinical grouping)
"""
import copy
import numpy as np
from .chemo_env import DEFAULT_PARAMS

PHYS_KEYS = ['r1', 'r2', 'b1', 'b2']
DRUG_KEYS = ['a1', 'a2', 'a3', 'c4']
IMMUNE_KEYS = ['c1', 'c2', 'rho', 'alpha', 'c3', 's', 'd1', 'd2']


def randomize_params(params=None, scale=0.1, clip_range=(0.5, 1.5)):
    """Grouped randomization with clip to avoid ODE explosion.
    scale: std of multiplicative noise (0.1-0.2 typical). clip_range: avoid extreme patients."""
    base = params or DEFAULT_PARAMS
    p = copy.deepcopy(base)
    lo, hi = clip_range
    for k in PHYS_KEYS:
        if k in p:
            theta0 = base[k]
            p[k] = np.clip(theta0 * np.random.normal(1.0, scale), lo * theta0, hi * theta0)
            p[k] = max(p[k], 1e-6)
    for k in DRUG_KEYS:
        if k in p:
            theta0 = base[k]
            p[k] = np.clip(theta0 * np.random.normal(1.0, scale * 0.5), lo * theta0, hi * theta0)
            p[k] = max(p[k], 1e-6)
    for k in IMMUNE_KEYS:
        if k in p:
            theta0 = base[k]
            p[k] = np.clip(theta0 * np.random.normal(1.0, scale), lo * theta0, hi * theta0)
            p[k] = max(p[k], 1e-6)
    return p
