"""
Chemotherapy ODE environment
Based on Padmanabhan et al. 2017, 4-dim system (N, T, I, C)

- transition_reward(prev, s, dt) with crossing-only bonuses
- clips per-step reward (REWARD_CLIP)

Reward profile (实验：奖励敏感性): 在生成数据或评估前设置环境变量
  CHEMO_REWARD_PROFILE=high_incentive
  会放大「彻底治愈」奖励并加重带瘤生存的肿瘤惩罚；默认不传或 default 则为原始标度。
"""
import os
import numpy as np


def reward_profile() -> str:
    """default | high_incentive — 由环境变量控制，便于数据生成脚本在 import 前设置。"""
    return os.environ.get("CHEMO_REWARD_PROFILE", "default").strip().lower()


def _reward_clip_range():
    if reward_profile() == "high_incentive":
        return (-200.0, 600.0)
    return (-100.0, 100.0)

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
# 肿瘤负荷过高 → 癌症进展致死（避免「零剂量苟满 300 步」的 reward hacking）
T_FATAL = 1.5  # 初始 T≈0.7；超过此阈值视为肿瘤相关死亡（可与队列 t_fatal 覆盖）
# Default baseline thresholds. Dynamically overridden by patient_cohorts.py during multi-cohort simulations.
C_TOX = 8.0    # toxicity limit: C > C_TOX -> terminate (safety constraint)
STATE_MAX = 30.0  # ODE explosion guard: any state > STATE_MAX -> terminate

# SDE: σ=0 退化为原 ODE；>0 时在每子步加入扩散项（未观测变异 / 代谢噪声）
SDE_DEFAULT_SIGMA = 0.0

# Reward clipping to avoid per-step extremes that destabilize offline RL
REWARD_CLIP = (-100.0, 100.0)

# Safe RL (CMDP): 二值 cost c ∈ {0, 1}
# c=1: 违规 (Unsafe) | c=0: 安全 (Safe)
# Default baseline thresholds; multi-cohort runs pass cohort-specific i_safe / n_safe via patient_ctx.
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
    patient_ctx is None: 与 is_done(x) 无 ctx 分支一致。
    patient_ctx 为 dict: 可含 t_fatal（肿瘤致死阈值）；毒性 = I < i_safe 或 C > c_tox。

    返回 (done: bool, reason: str)
    reason ∈ {running, cured, cancer_death, toxicity_death, organ_failure, immune_collapse, state_explosion}
    """
    T, N, I, C = float(x[1]), float(x[0]), float(x[2]), float(x[3])
    mx = float(np.max(x))

    if patient_ctx is None:
        ct = float(C_TOX)
        smax = float(STATE_MAX)
        t_fatal = float(T_FATAL)
        if T < T_CLEAR:
            return True, "cured"
        if T > t_fatal:
            return True, "cancer_death"
        if C > ct:
            return True, "toxicity_death"
        if N < 0.1:
            return True, "organ_failure"
        if I < 0.1:
            return True, "immune_collapse"
        if mx > smax:
            return True, "state_explosion"
        return False, "running"

    t_fatal = float(patient_ctx.get("t_fatal", T_FATAL))
    c_tox = float(patient_ctx.get("c_tox", C_TOX))
    i_safe = float(patient_ctx.get("i_safe", I_SAFE))
    n_organ = float(patient_ctx.get("n_safe", N_SAFE))
    smax = float(patient_ctx.get("state_max", STATE_MAX))
    if T < T_CLEAR:
        return True, "cured"
    if T > t_fatal:
        return True, "cancer_death"
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
    return (
        (T < T_CLEAR)
        or (T > T_FATAL)
        or (N < 0.1)
        or (I < 0.1)
        or (C > ct)
        or (np.max(x) > smax)
    )


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
    Crossing-only bonuses and per-step clipping are enforced.
    Returns scalar reward (float) or (reward, info) if debug=True."""
    # Expect s_prev or s_curr as raw (denormalized) states: [N, T, I, C]
    N, T, I, C = float(s_curr[0]), float(s_curr[1]), float(s_curr[2]), float(s_curr[3])
    prev_T = float(s_prev[1]) if s_prev is not None else T
    prev_N = float(s_prev[0]) if s_prev is not None else N
    prev_I = float(s_prev[2]) if s_prev is not None else I

    prof = reward_profile()
    if prof == "high_incentive":
        # 附录实验「高诱惑」：清除奖励 ~10×，带瘤负荷惩罚加重（敏感性分析）
        r_tumor = -10.0 * T * dt
        R_clear_bonus = 500.0
    else:
        r_tumor = -3.0 * T * dt
        R_clear_bonus = 50.0
    r_tox = -1.5 * np.tanh(C / 2.0) * dt
    r_progress = 2.0 * (prev_T - T) * dt if s_prev is not None else 0.0
    R_clear = R_clear_bonus if (prev_T >= T_CLEAR and T < T_CLEAR) else 0.0
    r_milestone = 0.0
    if prev_T >= 0.5 and T < 0.5:
        r_milestone += 2.0 * dt
    if prev_T >= 0.3 and T < 0.3:
        r_milestone += 3.0 * dt
    if prev_T >= 0.1 and T < 0.1:
        r_milestone += 5.0 * dt
    r_collapse = -20.0 if (N < 0.1 or I < 0.05) and (prev_N >= 0.1 and prev_I >= 0.05) else 0.0
    r = r_tumor + r_tox + r_progress + R_clear + r_milestone + r_collapse

    lo, hi = _reward_clip_range()
    r_clipped = float(np.clip(r, lo, hi))

    if debug:
        info = {
            'N': N, 'T': T, 'I': I, 'C': C,
            'prev_T': prev_T, 'prev_N': prev_N, 'prev_I': prev_I,
            'reward_raw': float(r), 'reward_clipped': r_clipped,
        }
        return r_clipped, info
    return r_clipped


def reward_fn(s, dt, s_prev=None):
    """Compat wrapper: reward_fn(s, dt, s_prev) delegates to transition_reward."""
    return transition_reward(s_prev, s, dt)
