"""
Chemotherapy ODE environment
Based on Padmanabhan et al. 2017, 4-dim system (N, T, I, C)
"""

import numpy as np

# State scale for normalization (log-domain for N,T,I)
X_SCALE = np.array([1.0, 1.0, 1.0, 3.0], dtype=np.float32)  # log1p(N,T,I), C

DEFAULT_PARAMS = {
    'r1': 1.5, 'r2': 1.0, 'b1': 1.0, 'b2': 1.0,
    'c1': 1.0, 'c2': 0.05, 'c3': 0.05, 'c4': 0.5,
    'a1': 0.35, 'a2': 0.45, 'a3': 0.25,  # stronger toxicity: high dose -> immune collapse
    's': 0.33, 'rho': 0.3, 'alpha': 0.3,
    'd1': 0.2, 'd2': 0.5,
}

DT = 0.3
MAX_STEPS = 300  # tumor dynamics are slow; 250 insufficient (Tctrl=75)
X0 = [1.0, 0.7, 1.0, 0.0]
ACTION_SPACE = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
ACTION_TO_IDX = {float(a): i for i, a in enumerate(ACTION_SPACE)}
I_THRESHOLD = 0.4
T_CLEAR = 0.02  # tumor "cleared" for done/metric/bonus
C_TOX = 8.0    # toxicity limit: C > C_tox -> terminate (safety constraint)
STATE_MAX = 30.0  # ODE explosion guard: any state > 30 -> terminate


def is_done(x):
    """Tumor cleared | organ failure | immune collapse | toxicity limit | ODE explosion"""
    T, N, I, C = x[1], x[0], x[2], x[3]
    return (T < T_CLEAR) or (N < 0.1) or (I < 0.1) or (C > C_TOX) or (np.max(x) > STATE_MAX)


def cancer_ode(t, x, u, params):
    """dx/dt = f(x, u)"""
    N, T, I, C = x
    p = params
    dN = p['r2']*N*(1 - p['b2']*N) - p['c4']*N*T - p['a3']*N*C
    dT = p['r1']*T*(1 - p['b1']*T) - p['c2']*I*T - p['c3']*T*N - p['a2']*T*C
    dI = p['s'] + p['rho']*I*T/(p['alpha'] + T) - p['c1']*I*T - p['d1']*I - p['a1']*I*C
    dC = -p['d2']*C + u
    return [dN, dT, dI, dC]


def step_ode(x, u, dt, params=None, n_sub=5):
    """Euler integration with clip. State > STATE_MAX indicates ODE explosion."""
    params = params or DEFAULT_PARAMS
    h = dt / n_sub
    x = np.asarray(x, dtype=np.float64)
    for _ in range(n_sub):
        dx = cancer_ode(0.0, x, u, params)
        x = x + h * np.asarray(dx)
        x = np.clip(x, 0.0, 50.0)
    return x.astype(np.float32)


def normalize_state(x):
    """Log-scale for N,T,I; linear for C. Improves numerical stability."""
    x = np.array(x, dtype=np.float32)
    x = x.copy()
    x[:3] = np.log1p(np.maximum(x[:3], 0.0))  # N, T, I
    return x / X_SCALE


def denormalize_state(x_norm):
    """Inverse of normalize_state"""
    x = np.array(x_norm, dtype=np.float32) * X_SCALE
    x[:3] = np.expm1(np.maximum(x[:3], 0.0))
    return x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def reward_fn(s, dt, s_prev=None):
    """
    Paper-grade reward: r = (-2T - 0.3C - 0.5σ(Ith-I) + 0.3N)Δt + 10*1_{T<0.02}
    -2T: strong tumor suppression
    -0.3C: moderate toxicity (avoids "no treatment" optimal)
    +0.3N: protect normal cells
    """
    N, T, I, C = s[:4]
    r = -2.0 * T - 0.3 * C - 0.5 * _sigmoid(I_THRESHOLD - I) + 0.3 * N
    r = r * dt
    if T < T_CLEAR:
        r += 10.0  # tumor cleared bonus
    return r


def reward_fn_v2(s, dt, s_prev=None):
    """
    Improved reward v2: progress shaping, stronger clearance, milestones.
    Encourages tumor clearance (TumorClear > 0%).
    """
    N, T, I, C = s[:4]
    cfg = {
        "w_tumor": 3.0, "w_progress": 2.0, "w_normal": 0.5,
        "w_immune": 2.0, "w_toxicity": 0.5, "R_clear": 50.0,
        "C_ref": 2.0, "collapse": -20.0, "m5": 3.0, "m10": 5.0,
    }
    prev_T = s_prev[1] if s_prev is not None else T

    tumor_penalty = -cfg["w_tumor"] * T * dt
    progress = cfg["w_progress"] * (prev_T - T) * dt if s_prev is not None else 0.0
    normal_reward = cfg["w_normal"] * N * dt
    immune_penalty = -cfg["w_immune"] * max(0, 0.3 - I) * dt if I < 0.3 else 0.0
    toxicity = -cfg["w_toxicity"] * np.tanh(C / cfg["C_ref"]) * dt
    clearance = cfg["R_clear"] if T < T_CLEAR else 0.0
    milestone = (cfg["m10"] if T < 0.1 else 0.0) + (cfg["m5"] if T < 0.05 else 0.0)
    collapse = cfg["collapse"] if (N < 0.1 or I < 0.05) else 0.0

    return tumor_penalty + progress + normal_reward + immune_penalty + toxicity + clearance + milestone + collapse
