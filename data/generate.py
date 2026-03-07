"""
Offline dataset generation for Supervised Offline RL
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""

import numpy as np
from env.chemo_env import (
    step_ode, DEFAULT_PARAMS, normalize_state, reward_fn, is_done,
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
    Aggressive T-based expert: more aggressive therapy for larger tumors.
    T>0.5: 2.0, T>0.3: 1.0, T>0.1: 0.5, else: 0.
    Dataset will have more aggressive therapy -> better for Offline RL.
    """
    N, T, I, C = s[:4]
    if T < T_CLEAR:
        base = 0.0
    elif N < 0.2 or I < 0.2:
        base = 0.0  # critical: stop
    elif T > 0.5:
        base = 2.0  # large tumor: aggressive
    elif T > 0.3:
        base = 1.0
    elif T > 0.1:
        base = 0.5
    else:
        base = 0.0

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


def behavior_policy(s, expert_ratio=0.5, conservative_ratio=0.3, noisy_expert_ratio=0.1):
    """
    Suboptimal mix for Offline RL: expert 50%, conservative 30%, noisy expert 10%, random 10%.
    More suboptimal data -> RL can surpass behavior.
    """
    p = np.random.rand()
    if p < expert_ratio:
        return expert_policy(s, epsilon=0.1)
    elif p < expert_ratio + conservative_ratio:
        return float(discretize_action(conservative_policy(s)))
    elif p < expert_ratio + conservative_ratio + noisy_expert_ratio:
        return float(noisy_expert_policy(s))
    else:
        return float(np.random.choice(ACTION_SPACE))


def expert_policy_v2(s, epsilon=0.15):
    """Improved expert: more aggressive, pursues clearance. C>5 or N/I critical -> stop."""
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


def behavior_policy_v2(s, traj_type="expert"):
    """Improved mix: expert 60%, balanced 20%, aggressive 10%, conservative 10%."""
    if traj_type == "expert":
        return expert_policy_v2(s, epsilon=0.15)
    if traj_type == "balanced":
        return balanced_policy(s)
    if traj_type == "aggressive":
        return aggressive_policy(s)
    if traj_type == "conservative":
        return float(discretize_action(conservative_policy(s)))
    return float(np.random.choice(ACTION_SPACE))


def _is_done(x):
    return is_done(x)


def collect_trajectory(policy, params=None, x0=None, randomize_patient=False, reward_fn=None):
    """Collect one trajectory. done=natural end, timeout=hit MAX_STEPS (d3rlpy needs one)."""
    from env.chemo_env import reward_fn as default_reward
    reward_fn = reward_fn or default_reward
    params = randomize_params(params, scale=0.15) if randomize_patient else (
        params or DEFAULT_PARAMS)
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
        r = reward_fn(x_next, DT, s_prev=x)
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


def make_policy_for_traj(traj_type):
    """Return policy function for given trajectory type (for v2 dataset)."""

    def policy(s):
        return behavior_policy_v2(s, traj_type)

    return policy


def generate_dataset_v2(
    n_trajectories=500,
    expert_ratio=0.60,
    balanced_ratio=0.20,
    aggressive_ratio=0.10,
    conservative_ratio=0.10,
    use_reward_v2=True,
    randomize_patient=True,
):
    """
    Improved dataset: 60% expert, 20% balanced, 10% aggressive, 10% conservative.
    Balanced improves action distribution; aggressive explores clearance path.
    """
    from env.chemo_env import reward_fn_v2

    traj_types = (
        ["expert"] * int(n_trajectories * expert_ratio)
        + ["balanced"] * int(n_trajectories * balanced_ratio)
        + ["aggressive"] * int(n_trajectories * aggressive_ratio)
        + ["conservative"] * int(n_trajectories * conservative_ratio)
    )
    np.random.shuffle(traj_types)

    all_transitions = []
    reward_fn = reward_fn_v2 if use_reward_v2 else None

    for traj_type in traj_types:
        policy = make_policy_for_traj(traj_type)
        traj = collect_trajectory(
            policy, randomize_patient=randomize_patient, reward_fn=reward_fn
        )
        all_transitions.extend(traj)

    # Action distribution
    a_idx = np.array([t["a_idx"] for t in all_transitions])
    for i, val in enumerate([0.0, 0.5, 1.0, 2.0]):
        pct = (a_idx == i).mean() * 100
        print(f"  Action {val}: {pct:.1f}%")
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
