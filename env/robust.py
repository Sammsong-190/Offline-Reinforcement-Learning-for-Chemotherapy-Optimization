"""
Robustness utilities: seeds, parameter-shift evaluation
"""
import os
import random
import numpy as np


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    # Reduce nondeterminism (optional)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rollout_param_shift(policy_fn, n_patients=50, n_ep_per_patient=2, params_base=None, scale=0.15):
    """
    Robust evaluation: test policy on randomized patients.
    scale: 0.15 = in-distribution, 0.30 = OOD (out-of-distribution).
    """
    from env.chemo_env import step_ode, reward_fn_v2, DT, MAX_STEPS, X0, is_done
    from env.patient import randomize_params

    params_base = params_base or __import__("env.chemo_env", fromlist=["DEFAULT_PARAMS"]).DEFAULT_PARAMS
    all_returns = []
    for _ in range(n_patients):
        params = randomize_params(params_base, scale=scale)
        for _ in range(n_ep_per_patient):
            x = np.array(X0, dtype=np.float32)
            R = 0.0
            for _ in range(MAX_STEPS):
                x_prev = x.copy()
                a = policy_fn(x)
                x = step_ode(x, a, DT, params)
                R += reward_fn_v2(x, DT, s_prev=x_prev)
                if is_done(x):
                    break
            all_returns.append(R)
    return np.mean(all_returns), np.std(all_returns)
