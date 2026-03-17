"""
Gym-style wrappers: 归一化、安全约束计算
- NormalizeObservation: 自动归一化 state
- SafetyMonitorWrapper: 核心！在 info 中返回 cost，供 Safe RL 使用
"""
import numpy as np
from env.chemo_env import (
    normalize_state, transition_cost, I_SAFE, N_SAFE,
    step_ode, transition_reward, is_done, DEFAULT_PARAMS, DT, MAX_STEPS, X0, ACTION_SPACE,
)


def compute_cost(s_raw: np.ndarray, i_safe=None, n_safe=None) -> float:
    """Cost for CMDP: 1 if I<i_safe or N<n_safe else 0."""
    i_safe = i_safe if i_safe is not None else I_SAFE
    n_safe = n_safe if n_safe is not None else N_SAFE
    N, I = float(s_raw[0]), float(s_raw[2])
    return 1.0 if (I < i_safe or N < n_safe) else 0.0


def is_safe(s_raw: np.ndarray, i_safe=None, n_safe=None) -> bool:
    """Check if state satisfies safety constraints."""
    i_safe = i_safe if i_safe is not None else I_SAFE
    n_safe = n_safe if n_safe is not None else N_SAFE
    N, I = float(s_raw[0]), float(s_raw[2])
    return N >= n_safe and I >= i_safe


class ChemoEnv:
    """Minimal env interface: reset(), step(). 纯 ODE 物理，无奖励/归一化。"""

    def __init__(self, params=None, x0=None):
        self.params = params or DEFAULT_PARAMS
        self.x0 = np.array(x0 or X0, dtype=np.float32)
        self.x = None
        self.step_count = 0

    def reset(self):
        self.x = self.x0.copy()
        self.step_count = 0
        return self.x.copy()

    def step(self, action):
        """Returns (s_next, r, done, info). info 不含 cost，由 wrapper 添加。"""
        u = float(np.clip(action, 0, 2))
        idx = np.argmin(np.abs(ACTION_SPACE - u))
        u = float(ACTION_SPACE[idx])
        x_prev = self.x.copy()
        self.x = step_ode(self.x, u, DT, self.params)
        self.step_count += 1
        r = transition_reward(x_prev, self.x, DT)
        done = is_done(self.x)
        timeout = self.step_count >= MAX_STEPS and not done
        info = {"timeout": timeout}
        return self.x.copy(), r, done, info


class SafetyMonitorWrapper:
    """
    核心！计算当前 Cost，在 info 中返回 cost 字段，供 Safe RL 算法使用。
    """
    def __init__(self, env, i_safe=None, n_safe=None):
        self.env = env
        self.i_safe = i_safe if i_safe is not None else I_SAFE
        self.n_safe = n_safe if n_safe is not None else N_SAFE

    def reset(self):
        return self.env.reset()

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        info["cost"] = compute_cost(s_next, self.i_safe, self.n_safe)
        info["safe"] = is_safe(s_next, self.i_safe, self.n_safe)
        return s_next, r, done, info


class NormalizeObservationWrapper:
    """Wrapper: 对 policy 暴露归一化 state，内部保留 raw 供 reward/cost 计算。"""

    def __init__(self, env):
        self.env = env

    def reset(self):
        s = self.env.reset()
        return normalize_state(s)

    def step(self, action):
        s_next, r, done, info = self.env.step(action)
        return normalize_state(s_next), r, done, info


# Legacy aliases
NormalizeObservation = NormalizeObservationWrapper
