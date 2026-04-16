"""
Chemotherapy ODE environment (v3-only)
Based on Padmanabhan et al. 2017, 4-dim system (N, T, I, C)

This file:
- keeps only reward v3 as default
- implements transition_reward(prev, s, dt) with crossing-only bonuses
- clips per-step reward (REWARD_CLIP)
- exposes DEFAULT_REWARD_VERSION='v3'
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
MAX_STEPS = 300  # tumor dynamics are slow
X0 = [1.0, 0.7, 1.0, 0.0]
ACTION_SPACE = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
ACTION_TO_IDX = {float(a): i for i, a in enumerate(ACTION_SPACE)}
I_THRESHOLD = 0.4
T_CLEAR = 0.02  # tumor "cleared" for done/metric/bonus
C_TOX = 8.0    # toxicity limit: C > C_TOX -> terminate (safety constraint)
STATE_MAX = 30.0  # ODE explosion guard: any state > STATE_MAX -> terminate

# SDE: σ=0 退化为原 ODE；>0 时在每子步加入扩散项（未观测变异 / 代谢噪声）
SDE_DEFAULT_SIGMA = 0.0

# Reward clipping to avoid per-step extremes that destabilize offline RL
REWARD_CLIP = (-100.0, 100.0)

# Default reward version used in experiments / paper
DEFAULT_REWARD_VERSION = 'v3'

# Safe RL (CMDP): 二值 cost c ∈ {0, 1}
# c=1: 违规 (Unsafe) | c=0: 安全 (Safe)
# 阈值与 reward_fn 软惩罚区分: 此为硬约束
# 目标违规率 5%-15%: 过严(0.3/0.4)导致 80%+ 违规，算法无法学习
I_SAFE = 0.2   # 免疫崩溃硬约束 (0.3→0.2 放松，与 is_done 0.1 保持梯度)
N_SAFE = 0.2   # 器官衰竭硬约束 (0.4→0.2 放松)


def transition_cost(s_curr, i_safe=None, n_safe=None):
    """二值指示函数: c=1 if 违规 else 0. 可传入队列特异阈值。"""
    thr_i = I_SAFE if i_safe is None else float(i_safe)
    thr_n = N_SAFE if n_safe is None else float(n_safe)
    N, I = float(s_curr[0]), float(s_curr[2])
    return 1.0 if (I < thr_i or N < thr_n) else 0.0


get_cost = transition_cost  # alias for paper/API


def termination_info(x, patient_ctx=None):
    """
    单次状态判断终止原因（用于 Rollout / Kaplan-Meier）。
    patient_ctx is None: 与旧版 is_done(x) 一致（器官 N<0.1、免疫 I<0.1，不按 I_SAFE 终止）。
    patient_ctx 为 dict: 毒性致死 = I < i_safe 或 C > c_tox；治愈 = T < T_CLEAR。

    返回 (done: bool, reason: str)
    reason ∈ {running, cured, toxicity_death, organ_failure, immune_collapse, state_explosion, timeout}
    """
    T, N, I, C = float(x[1]), float(x[0]), float(x[2]), float(x[3])
    mx = float(np.max(x))

    if patient_ctx is None:
        ct = float(C_TOX)
        smax = float(STATE_MAX)
        if T < T_CLEAR:
            return True, "cured"
        if C > ct:
            return True, "toxicity_death"
        if N < 0.1:
            return True, "organ_failure"
        if I < 0.1:
            return True, "immune_collapse"
        if mx > smax:
            return True, "state_explosion"
        return False, "running"

    c_tox = float(patient_ctx.get("c_tox", C_TOX))
    i_safe = float(patient_ctx.get("i_safe", I_SAFE))
    n_organ = float(patient_ctx.get("n_safe", N_SAFE))
    smax = float(patient_ctx.get("state_max", STATE_MAX))
    if T < T_CLEAR:
        return True, "cured"
    if C > c_tox:
        return True, "toxicity_death"
    if I < i_safe:
        return True, "toxicity_death"
    if N < n_organ:
        return True, "organ_failure"
    if I < 0.1:
        return True, "immune_collapse"
    if mx > smax:
        return True, "state_explosion"
    return False, "running"


def is_done(x, c_tox=None, state_max=None, patient_ctx=None):
    """兼容旧 API；若传入 patient_ctx 则与 termination_info 一致。"""
    if patient_ctx is not None:
        return termination_info(x, patient_ctx)[0]
    T, N, I, C = x[1], x[0], x[2], x[3]
    ct = C_TOX if c_tox is None else float(c_tox)
    smax = STATE_MAX if state_max is None else float(state_max)
    return (T < T_CLEAR) or (N < 0.1) or (I < 0.1) or (C > ct) or (np.max(x) > smax)


def cancer_ode(t, x, u, params):
    """dx/dt = f(x, u)"""
    N, T, I, C = x
    p = params
    dN = p['r2']*N*(1 - p['b2']*N) - p['c4']*N*T - p['a3']*N*C
    dT = p['r1']*T*(1 - p['b1']*T) - p['c2']*I*T - p['c3']*T*N - p['a2']*T*C
    dI = p['s'] + p['rho']*I*T/(p['alpha'] + T) - p['c1']*I*T - p['d1']*I - p['a1']*I*C
    dC = -p['d2']*C + u
    return [dN, dT, dI, dC]


def step_ode(x, u, dt, params=None, n_sub=5, sde_sigma=None, rng=None):
    """Euler integration；可选 SDE 项: x += h*dx + sigma*sqrt(h)*N(0,I)（每子步）。"""
    params = params or DEFAULT_PARAMS
    sig = SDE_DEFAULT_SIGMA if sde_sigma is None else float(sde_sigma)
    h = dt / n_sub
    x = np.asarray(x, dtype=np.float64)
    for _ in range(n_sub):
        dx = cancer_ode(0.0, x, u, params)
        x = x + h * np.asarray(dx)
        if sig > 0.0:
            if rng is not None and isinstance(rng, np.random.Generator):
                z = rng.standard_normal(4)
            else:
                z = np.random.randn(4)
            x = x + sig * np.sqrt(h) * z
        x = np.clip(x, 0.0, 50.0)
    return x.astype(np.float32)


def normalize_state(x):
    """Log-scale for N,T,I; linear for C. Improves numerical stability."""
    x = np.array(x, dtype=np.float32)
    x = x.copy()
    x[:3] = np.log1p(np.maximum(x[:3], 0.0))  # N, T, I
    return x / X_SCALE


# Approximate mean/std for Z-score (log1p(N,T,I), C/3). Tune from data for production.
X_MEAN = np.array([0.5, 0.35, 0.5, 0.4], dtype=np.float32)
X_STD = np.array([0.35, 0.3, 0.35, 0.35], dtype=np.float32)


def normalize_state_zscore(x, mean=None, std=None):
    """Z-score normalization. Use mean/std from training data for best stability."""
    x = np.array(x, dtype=np.float32)
    x = x.copy()
    x[:3] = np.log1p(np.maximum(x[:3], 0.0))
    x[3] = x[3] / X_SCALE[3]
    m = mean if mean is not None else X_MEAN
    s = std if std is not None else X_STD
    return (x - m) / np.maximum(s, 1e-6)


def denormalize_state(x_norm):
    """Inverse of normalize_state"""
    x = np.array(x_norm, dtype=np.float32) * X_SCALE
    x[:3] = np.expm1(np.maximum(x[:3], 0.0))
    return x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def transition_reward(s_prev, s_curr, dt, debug=False):
    """Compute reward for transition (s_prev -> s_curr) with time step dt.
    Implements v3 logic (only v3 present in this repo).
    Crossing-only bonuses and per-step clipping are enforced.
    Returns scalar reward (float) or (reward, info) if debug=True."""
    # Expect s_prev or s_curr as raw (denormalized) states: [N, T, I, C]
    N, T, I, C = float(s_curr[0]), float(s_curr[1]), float(s_curr[2]), float(s_curr[3])
    prev_T = float(s_prev[1]) if s_prev is not None else T
    prev_N = float(s_prev[0]) if s_prev is not None else N
    prev_I = float(s_prev[2]) if s_prev is not None else I

    # v3: scaled, milestones only on first crossing of sub-thresholds
    r_tumor = -3.0 * T * dt
    r_tox = -1.5 * np.tanh(C / 2.0) * dt
    r_progress = 2.0 * (prev_T - T) * dt if s_prev is not None else 0.0
    R_clear = 50.0 if (prev_T >= T_CLEAR and T < T_CLEAR) else 0.0
    r_milestone = 0.0
    if prev_T >= 0.5 and T < 0.5:
        r_milestone += 2.0 * dt
    if prev_T >= 0.3 and T < 0.3:
        r_milestone += 3.0 * dt
    if prev_T >= 0.1 and T < 0.1:
        r_milestone += 5.0 * dt
    r_collapse = -20.0 if (N < 0.1 or I < 0.05) and (prev_N >= 0.1 and prev_I >= 0.05) else 0.0
    r = r_tumor + r_tox + r_progress + R_clear + r_milestone + r_collapse

    # Clip reward to avoid extremely large per-step values
    r_clipped = float(np.clip(r, REWARD_CLIP[0], REWARD_CLIP[1]))

    if debug:
        info = {
            'N': N, 'T': T, 'I': I, 'C': C,
            'prev_T': prev_T, 'prev_N': prev_N, 'prev_I': prev_I,
            'reward_raw': float(r), 'reward_clipped': r_clipped,
            'reward_version': 'v3'
        }
        return r_clipped, info
    return r_clipped


# Backward compatibility: reward_fn(s, dt, s_prev) -> transition_reward(s_prev, s, dt)
def reward_fn(s, dt, s_prev=None):
    """Compat wrapper: reward_fn(s, dt, s_prev) delegates to transition_reward."""
    return transition_reward(s_prev, s, dt)


reward_fn_v2 = reward_fn
reward_fn_v3 = reward_fn
