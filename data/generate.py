"""
Offline dataset generation for Supervised Offline RL
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""

import numpy as np
from env.chemo_env import (
    step_ode, DEFAULT_PARAMS, normalize_state, reward_fn,
    DT, MAX_STEPS, X0, ACTION_SPACE, ACTION_TO_IDX, I_THRESHOLD, T_CLEAR,
)
from env.patient import randomize_params


def discretize_action(u):
    """Round to nearest in ACTION_SPACE"""
    idx = np.argmin(np.abs(ACTION_SPACE - u))
    return ACTION_SPACE[idx]


def action_to_index(a):
    """Map action value to index (cached lookup)"""
    return ACTION_TO_IDX.get(float(a), int(np.argmin(np.abs(ACTION_SPACE - a))))


def expert_policy(s, epsilon=0.2):
    """
    Aggressive expert: reward favors high dose (Fixed 2.0 >> Fixed 1.0).
    Default 2.0 when T>0.2 and N,I>0.25; back off only when N or I critical.
    """
    N, T, I, C = s[:4]
    if T < T_CLEAR:
        base = 0.0
    elif N < 0.2 or I < 0.2:
        base = 0.0  # critical: stop
    elif N < 0.3 or I < 0.3:
        base = 0.5  # risky: low dose
    elif N < 0.4 or I < 0.4:
        base = 1.0  # cautious
    else:
        base = 2.0  # safe: aggressive (reward-optimal)

    if np.random.rand() < epsilon:
        return float(np.random.choice(ACTION_SPACE))
    return base


def conservative_policy(s):
    """Cautious: low dose when uncertain."""
    N, T, I, C = s[:4]
    if N > 0.5 and I > 0.5 and T > 0.3:
        return 0.5
    return 0.0


def noisy_expert_policy(s, noise_std=0.2):
    """Expert + Gaussian noise, then discretize. Mimics real medical variability."""
    a_expert = expert_policy(s, epsilon=0.0)
    idx = action_to_index(a_expert)
    u_cont = float(ACTION_SPACE[idx]) + np.random.normal(0, noise_std)
    return discretize_action(np.clip(u_cont, 0, 2.0))


def behavior_policy(s, expert_ratio=0.85, conservative_ratio=0.10, use_noisy_expert=0.1):
    """
    Expert-heavy mix for BC≈Expert: expert (85%), conservative (10%), random (5%).
    use_noisy_expert: fraction of expert calls that use noisy_expert instead.
    """
    p = np.random.rand()
    if p < expert_ratio:
        if np.random.rand() < use_noisy_expert:
            return float(noisy_expert_policy(s))
        return expert_policy(s, epsilon=0.05)  # low epsilon for cleaner expert signal
    elif p < expert_ratio + conservative_ratio:
        return float(discretize_action(conservative_policy(s)))
    else:
        return float(np.random.choice(ACTION_SPACE))


def _is_done(x):
    T, N, I = x[1], x[0], x[2]
    return (T < T_CLEAR) or (N < 0.1) or (I < 0.1)


def collect_trajectory(policy, params=None, x0=None, randomize_patient=False):
    """Collect one trajectory. done=natural end, timeout=hit MAX_STEPS (d3rlpy needs one)."""
    params = randomize_params(params, scale=0.15) if randomize_patient else (params or DEFAULT_PARAMS)
    x = np.array(x0 or X0, dtype=np.float32)
    transitions = []
    for step in range(MAX_STEPS):
        s = x.copy()
        try:
            a = discretize_action(policy(s))
        except TypeError:
            a = discretize_action(policy(s, epsilon=0.2))
        x_next = step_ode(x, a, DT, params)
        done = _is_done(x_next)
        timeout = (step == MAX_STEPS - 1) and not done  # hit max steps
        r = reward_fn(x_next, DT, s_prev=x)  # includes terminal_bonus when T<1e-6
        s_norm = normalize_state(s)
        s_next_norm = normalize_state(x_next)
        transitions.append({
            's': s_norm, 's_raw': s.copy(),
            'a': a, 'a_idx': action_to_index(a),
            'r': r, 's_next': s_next_norm, 's_next_raw': x_next.copy(),
            'done': done,
            'timeout': timeout,
        })
        x = x_next
        if done:
            break
    return transitions


def generate_dataset(n_trajectories=1000, policy=None, randomize_patient=True):
    """Generate offline dataset. Default: mixture behavior policy for better coverage."""
    policy = policy or behavior_policy
    all_transitions = []
    for _ in range(n_trajectories):
        traj = collect_trajectory(policy, randomize_patient=randomize_patient)
        all_transitions.extend(traj)
    return all_transitions


def save_dataset(transitions, path='offline_dataset.npz'):
    """Save with s_norm, s_raw, a_idx, r, s'_norm, s'_raw, done, timeout"""
    s = np.array([t['s'] for t in transitions])
    s_raw = np.array([t['s_raw'] for t in transitions])
    a = np.array([t['a_idx'] for t in transitions], dtype=np.int64)
    r = np.array([t['r'] for t in transitions])
    s_next = np.array([t['s_next'] for t in transitions])
    s_next_raw = np.array([t['s_next_raw'] for t in transitions])
    done = np.array([t['done'] for t in transitions])
    timeout = np.array([t.get('timeout', False) for t in transitions])
    np.savez(path, s=s, s_raw=s_raw, a=a, r=r, s_next=s_next, s_next_raw=s_next_raw,
             done=done, timeout=timeout, action_space=ACTION_SPACE)
    print(f"Saved {len(transitions)} transitions to {path}")
    return path
