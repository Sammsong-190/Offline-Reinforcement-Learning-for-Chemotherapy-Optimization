"""评估脚本: 含安全指标计算"""
import numpy as np
from env.chemo_env import step_ode, reward_fn, DEFAULT_PARAMS, MAX_STEPS, X0, is_done, T_CLEAR, I_SAFE, N_SAFE


def evaluate_policy(policy_fn, n_ep=10, params=None):
    """Rollout policy, return mean_return, std_return, tumor_clear%, survival%, avg_dose, cost_rate%."""
    params = params or DEFAULT_PARAMS
    returns, tumor_clears, survivals, doses, cost_rates = [], [], [], [], []
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        R, actions, cost_count = 0.0, [], 0
        for step in range(MAX_STEPS):
            x_prev = x.copy()
            a = policy_fn(x)
            actions.append(float(a))
            x = step_ode(x, a, 0.3, params)
            R += reward_fn(x, 0.3, s_prev=x_prev)
            if x[2] < I_SAFE or x[0] < N_SAFE:
                cost_count += 1
            if is_done(x):
                break
        returns.append(R)
        tumor_clears.append(1.0 if x[1] < T_CLEAR else 0.0)
        survivals.append(1.0 if (x[0] > 0.1 and x[2] > 0.1) else 0.0)
        doses.append(np.mean(actions) if actions else 0.0)
        cost_rates.append(cost_count / max(step + 1, 1) * 100)
    return (
        np.mean(returns), np.std(returns),
        np.mean(tumor_clears) * 100, np.mean(survivals) * 100,
        np.mean(doses), np.mean(cost_rates),
    )
