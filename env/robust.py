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
    from env.chemo_env import step_ode, reward_fn_v3, DT, MAX_STEPS, X0, termination_info
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
                R += reward_fn_v3(x, DT, s_prev=x_prev)
                if termination_info(x, None)[0]:
                    break
            all_returns.append(R)
    return np.mean(all_returns), np.std(all_returns)


def rollout_virtual_cohorts(policy_fn, n_ep_per_cohort=20, cohort_ids=None, seed=42):
    """
    按虚拟患者亚群分别评估（用于「异质性」实验叙事）。
    cohort_ids: None 表示三类都跑。
    """
    from env.chemo_env import step_ode, reward_fn_v3, DT, MAX_STEPS, X0, termination_info
    from env.patient_cohorts import PatientGenerator, COHORT_IDS

    rng = np.random.default_rng(seed)
    gen = PatientGenerator(rng=rng)
    ids = cohort_ids or list(COHORT_IDS)
    results = {}
    for cid in ids:
        ctx = gen.from_cohort(cid, jitter=0.0)
        rets, surv_steps, reasons = [], [], []
        for _ in range(n_ep_per_cohort):
            x = np.array(X0, dtype=np.float32)
            R = 0.0
            survival_steps = MAX_STEPS
            reason = "timeout"
            for step in range(MAX_STEPS):
                x_prev = x.copy()
                a = policy_fn(x)
                x = step_ode(x, a, DT, ctx["params"], sde_sigma=ctx.get("sde_sigma", 0.0), rng=rng)
                R += reward_fn_v3(x, DT, s_prev=x_prev)
                done, r = termination_info(x, ctx)
                if done:
                    survival_steps = step + 1
                    reason = r
                    break
            rets.append(R)
            surv_steps.append(survival_steps)
            reasons.append(reason)
        uniq = set(reasons)
        row = {
            "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)),
            "survival_steps_mean": float(np.mean(surv_steps)),
            "survival_steps_std": float(np.std(surv_steps)),
            "survival_time_mean": float(np.mean(surv_steps)) * DT,
        }
        for r in uniq:
            row[f"frac_{r}"] = float(np.mean([x == r for x in reasons]))
        results[cid] = row
    return results
