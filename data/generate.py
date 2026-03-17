"""
Offline dataset generation for Supervised Offline RL
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""

import numpy as np
from env.chemo_env import (
    step_ode, DEFAULT_PARAMS, normalize_state, is_done,
    DT, MAX_STEPS, X0, ACTION_SPACE, ACTION_TO_IDX, T_CLEAR,
)
from env.patient import randomize_params


def discretize_action(u):
    """Round to nearest in ACTION_SPACE"""
    idx = np.argmin(np.abs(ACTION_SPACE - u))
    return ACTION_SPACE[idx]


def action_to_index(a):
    """Map action value to index (cached lookup)"""
    return ACTION_TO_IDX.get(float(a), int(np.argmin(np.abs(ACTION_SPACE - a))))


def expert_policy(s, epsilon=0.15):
    """Expert: aggressive, pursues clearance. C>5 or N/I critical -> stop."""
    N, T, I, C = s[:4]
    if np.random.random() < epsilon:
        return float(np.random.choice(ACTION_SPACE))
    if C > 5.0 or N < 0.15 or I < 0.15:
        return 0.0
    if T > 0.6:
        return 2.0
    if T > 0.3 and I > 0.3:
        return 2.0 if np.random.random() < 0.5 else 1.0
    if T > 0.1:
        return 1.0
    if T > 0.02:
        return 0.5
    return 0.0


def balanced_policy(s):
    """Balanced sampling: improve action distribution toward ~25% each."""
    N, T, I, C = s[:4]
    if C > 4.0 or N < 0.2:
        weights = [0.7, 0.2, 0.1, 0.0]
    elif T > 0.5:
        weights = [0.05, 0.15, 0.35, 0.45]
    elif T > 0.2:
        weights = [0.1, 0.25, 0.40, 0.25]
    else:
        weights = [0.40, 0.35, 0.20, 0.05]
    return float(np.random.choice(ACTION_SPACE, p=np.array(weights) / sum(weights)))


def aggressive_policy(s):
    """High-dose trajectories for clearance path exploration."""
    N, T, I, C = s[:4]
    if C > 6.0 or N < 0.1:
        return 0.0
    return 2.0 if T > 0.05 else 0.5


def balanced_expert_policy(s, balance_ratio=0.6, expert_epsilon=0.2):
    """balance_ratio: uniform sampling; (1-ratio): expert. expert_epsilon: exploration in expert."""
    if np.random.rand() < balance_ratio:
        return float(np.random.choice(ACTION_SPACE))
    return expert_policy(s, epsilon=expert_epsilon)


def add_state_noise(s, sigma=0.02):
    """Add Gaussian noise to state for data augmentation."""
    return np.array(s, dtype=np.float32) + np.random.normal(0, sigma, size=s.shape).astype(np.float32)


def _policy_by_type(s, traj_type="expert", expert_balance_ratio=0.6, expert_epsilon=0.2):
    """Policy for given trajectory type. expert_balance_ratio: uniform % in expert traj."""
    if traj_type == "expert":
        return balanced_expert_policy(s, balance_ratio=expert_balance_ratio, expert_epsilon=expert_epsilon)
    if traj_type == "balanced":
        return balanced_policy(s)
    if traj_type == "aggressive":
        return aggressive_policy(s)
    if traj_type == "conservative":
        N, T, I, C = s[:4]
        if N > 0.5 and I > 0.5 and T > 0.3:
            return 0.5
        return 0.0
    return float(np.random.choice(ACTION_SPACE))


def behavior_policy(s):
    """Mixture policy (60% expert, 20% balanced, 10% aggressive, 10% conservative). Used for verify baseline."""
    t = np.random.choice(["expert", "balanced", "aggressive", "conservative"], p=[0.6, 0.2, 0.1, 0.1])
    return _policy_by_type(s, t)


def _is_done(x):
    return is_done(x)


def collect_trajectory(policy, params=None, x0=None, randomize_patient=False, reward_fn=None,
                      state_noise_sigma=0.0, patient_scale=0.15):
    """Collect one trajectory. state_noise_sigma: add noise to state for policy input (data aug)."""
    from env.chemo_env import reward_fn as default_reward
    reward_fn = reward_fn or default_reward
    params = randomize_params(params, scale=patient_scale) if randomize_patient else (
        params or DEFAULT_PARAMS)
    x = np.array(x0 or X0, dtype=np.float32)
    transitions = []
    for step in range(MAX_STEPS):
        s = x.copy()
        s_for_policy = add_state_noise(s, state_noise_sigma) if state_noise_sigma > 0 else s
        try:
            a = discretize_action(policy(s_for_policy))
        except TypeError:
            a = discretize_action(policy(s_for_policy, epsilon=0.2))
        x_next = step_ode(x, a, DT, params)
        done = _is_done(x_next)
        timeout = (step == MAX_STEPS - 1) and not done  # hit max steps
        r = reward_fn(x_next, DT, s_prev=x)
        from env.chemo_env import transition_cost
        c = transition_cost(x_next)
        s_norm = normalize_state(s)
        s_next_norm = normalize_state(x_next)
        transitions.append({
            's': s_norm, 's_raw': s.copy(),
            'a': a, 'a_idx': action_to_index(a),
            'r': r, 'c': c, 's_next': s_next_norm, 's_next_raw': x_next.copy(),
            'done': done,
            'timeout': timeout,
        })
        x = x_next
        if done:
            break
    return transitions


def _policy_for_traj(traj_type, expert_balance_ratio=0.6, expert_epsilon=0.2):
    """Return policy function for given trajectory type."""

    def policy(s):
        return _policy_by_type(s, traj_type, expert_balance_ratio, expert_epsilon)

    return policy


def generate_dataset(
    n_trajectories=1000,
    expert_ratio=0.50,
    balanced_ratio=0.30,
    aggressive_ratio=0.10,
    conservative_ratio=0.10,
    use_reward_v3=True,
    randomize_patient=True,
    state_noise_sigma=0.02,
    expert_balance_ratio=0.6,
    expert_epsilon=0.2,
    patient_scale=0.15,
    seed=None,
):
    """
    Generate offline dataset. Default: 50% expert, 30% balanced, 10% aggressive, 10% conservative.
    seed: 复现性 (SCI 红线)。若提供则 set_seed(seed) 并用于 traj shuffle。
    """
    if seed is not None:
        from env.robust import set_seed
        set_seed(seed)
    rng = np.random.RandomState(seed if seed is not None else 0)

    from env.chemo_env import reward_fn_v2, reward_fn_v3
    reward_fn_impl = reward_fn_v3 if use_reward_v3 else reward_fn_v2

    traj_types = (
        ["expert"] * int(n_trajectories * expert_ratio)
        + ["balanced"] * int(n_trajectories * balanced_ratio)
        + ["aggressive"] * int(n_trajectories * aggressive_ratio)
        + ["conservative"] * int(n_trajectories * conservative_ratio)
    )
    rng.shuffle(traj_types)

    all_transitions = []
    reward_fn = reward_fn_impl

    for traj_type in traj_types:
        policy = _policy_for_traj(traj_type, expert_balance_ratio, expert_epsilon)
        traj = collect_trajectory(
            policy, randomize_patient=randomize_patient, reward_fn=reward_fn,
            state_noise_sigma=state_noise_sigma, patient_scale=patient_scale,
        )
        all_transitions.extend(traj)

    # Action distribution
    a_idx = np.array([t["a_idx"] for t in all_transitions])
    for i, val in enumerate([0.0, 0.5, 1.0, 2.0]):
        pct = (a_idx == i).mean() * 100
        print(f"  Action {val}: {pct:.1f}%")
    # Cost 分布 (SCI: 理想 5%-15%)
    c_arr = np.array([t.get("c", 0.0) for t in all_transitions])
    cost_rate = c_arr.mean() * 100
    print(f"  Cost 违规率: {cost_rate:.2f}% (理想 5%-15%)")
    return all_transitions


def save_dataset(transitions, path='offline_dataset.npz'):
    """D4RL 标准: 显式 done(terminals), timeout, costs。生成时存好，不猜测。"""
    s = np.array([t['s'] for t in transitions])
    s_raw = np.array([t['s_raw'] for t in transitions])
    a = np.array([t['a_idx'] for t in transitions], dtype=np.int64)
    r = np.array([t['r'] for t in transitions])
    c = np.array([t.get('c', 0.0) for t in transitions], dtype=np.float32)
    s_next = np.array([t['s_next'] for t in transitions])
    s_next_raw = np.array([t['s_next_raw'] for t in transitions])
    done = np.array([t['done'] for t in transitions])
    timeout = np.array([t.get('timeout', False) for t in transitions])
    np.savez(path, s=s, s_raw=s_raw, a=a, r=r, c=c, s_next=s_next, s_next_raw=s_next_raw,
             done=done, timeout=timeout, action_space=ACTION_SPACE)
    print(f"Saved {len(transitions)} transitions to {path}")
    return path


def save_dataset_d4rl(transitions, path='offline_dataset_d4rl.npz'):
    """D4RL 兼容: observations, actions, rewards, next_observations, terminals, timeouts, costs
    保留 terminals(done) 与 timeouts 分开，算法可区分真死 vs 时间到。"""
    s = np.array([t['s'] for t in transitions], dtype=np.float32)
    a = np.array([t['a_idx'] for t in transitions], dtype=np.int64)
    r = np.array([t['r'] for t in transitions], dtype=np.float32)
    s_next = np.array([t['s_next'] for t in transitions], dtype=np.float32)
    done = np.array([t['done'] for t in transitions], dtype=bool)
    timeout = np.array([t.get('timeout', False) for t in transitions], dtype=bool)
    c = np.array([t.get('c', 0.0) for t in transitions], dtype=np.float32)
    np.savez(path,
             observations=s,
             actions=a,
             rewards=r,
             next_observations=s_next,
             terminals=done,
             timeouts=timeout,
             costs=c,
             action_space=ACTION_SPACE)
    print(f"Saved D4RL format: {len(transitions)} transitions to {path}")
    return path
