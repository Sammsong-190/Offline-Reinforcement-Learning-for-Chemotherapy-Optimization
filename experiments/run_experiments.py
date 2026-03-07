"""
Paper-level experiments: Main Performance, Patient Robustness, Safety Analysis
Run: python -m experiments.run_experiments
"""
import os
import sys
import types
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gymnasium compatibility for d3rlpy
try:
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit
    sys.modules["gym"] = gym
    tl_mod = types.ModuleType("gym.wrappers.time_limit")
    tl_mod.TimeLimit = TimeLimit
    sys.modules["gym.wrappers.time_limit"] = tl_mod
except ImportError:
    pass

from env.robust import set_seed
from env.chemo_env import step_ode, reward_fn, DEFAULT_PARAMS, DT, MAX_STEPS, X0, T_CLEAR, is_done
from env.patient import randomize_params

set_seed(42)


def vary_param(params, key, scale=0.2):
    """Vary single param by ±scale (e.g., 0.2 = ±20%)"""
    p = params.copy()
    if key in p:
        p[key] = p[key] * (1 + np.random.uniform(-scale, scale))
        p[key] = max(p[key], 1e-6)
    return p


def rollout_metrics(policy_fn, params, n_ep=5):
    """Return (mean_return, final_tumor, avg_dose, max_tox, tumor_clear_pct, survival_pct)"""
    returns, final_t, doses, max_cs, clears, survs = [], [], [], [], [], []
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        R, actions, toxics = 0.0, [], []
        for _ in range(MAX_STEPS):
            a = policy_fn(x)
            actions.append(float(a))
            x = step_ode(x, a, DT, params)
            toxics.append(x[3])
            R += reward_fn(x, DT)
            if is_done(x):
                break
        returns.append(R)
        final_t.append(x[1])
        doses.append(np.mean(actions) if actions else 0)
        max_cs.append(np.max(toxics) if toxics else 0)
        clears.append(1.0 if x[1] < T_CLEAR else 0.0)
        survs.append(1.0 if (x[0] > 0.1 and x[2] > 0.1) else 0.0)
    return (
        np.mean(returns), np.mean(final_t), np.mean(doses),
        np.mean(max_cs), np.mean(clears) * 100, np.mean(survs) * 100,
    )


def toxicity_violation_rate(policy_fn, params, threshold=1.5, n_ep=20):
    """% of episodes where max(C) > threshold"""
    violations = 0
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        max_c = 0.0
        for _ in range(MAX_STEPS):
            a = policy_fn(x)
            x = step_ode(x, a, DT, params)
            max_c = max(max_c, x[3])
            if is_done(x):
                break
        if max_c > threshold:
            violations += 1
    return violations / n_ep * 100


def get_policies():
    """Load BC, CQL, IQL, Expert policies"""
    from env.chemo_env import ACTION_SPACE, normalize_state
    from data.generate import expert_policy

    policies = {}

    def policy_expert(s):
        return expert_policy(s, epsilon=0.0)

    policies["Expert"] = policy_expert

    if os.path.exists("bc_policy.pt"):
        import torch
        from train_offline import PolicyNet
        net = PolicyNet()
        net.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
        net.eval()

        def policy_bc(s):
            s_norm = normalize_state(s)
            x = torch.FloatTensor(s_norm).unsqueeze(0)
            with torch.no_grad():
                idx = net(x).argmax(dim=1).item()
            return float(ACTION_SPACE[idx])

        policies["BC"] = policy_bc

    for name, path, key in [
        ("CQL", "cql_model.d3", "cql"),
        ("IQL", "iql_model.d3", "iql"),
    ]:
        if os.path.exists(path):
            try:
                import d3rlpy
                algo = d3rlpy.load_learnable(path)

                def make_policy(algo):
                    def fn(s):
                        s_norm = normalize_state(s)
                        x = np.array(s_norm, dtype=np.float32).reshape(1, -1)
                        idx = algo.predict(x)[0]
                        return float(ACTION_SPACE[int(idx) if hasattr(idx, 'item') else idx])
                    return fn

                policies[name] = make_policy(algo)
            except Exception:
                pass

    return policies


def exp1_main_performance(policies, n_ep=20):
    """Experiment 1: Main Performance Table"""
    print("\n" + "=" * 70)
    print("Experiment 1: Main Performance")
    print("=" * 70)
    rows = []
    for name, policy_fn in policies.items():
        ret, final_t, dose, max_c, tc, surv = rollout_metrics(policy_fn, DEFAULT_PARAMS, n_ep=n_ep)
        rows.append({
            "Policy": name,
            "Return": ret,
            "FinalTumor": final_t,
            "AvgDose": dose,
            "MaxTox": max_c,
            "TumorClear": tc,
            "Survival": surv,
        })
        print(f"{name:10} Return={ret:8.2f}  FinalT={final_t:.4f}  AvgDose={dose:.2f}  MaxTox={max_c:.3f}  TumorClear={tc:.0f}%  Survival={surv:.0f}%")
    return rows


def exp2_patient_robustness(policies, n_patients=50, variation=0.2):
    """Experiment 2: Return vs parameter variation (tumor/immune/drug)"""
    print("\n" + "=" * 70)
    print("Experiment 2: Patient Robustness (return vs parameter shift ±20%)")
    print("=" * 70)

    # Variation modes: tumor growth (r1), immune (c1), drug decay (d2)
    modes = [
        ("tumor_growth", "r1"),
        ("immune_strength", "c1"),
        ("drug_decay", "d2"),
    ]
    results = {m[0]: {} for m in modes}

    for mode_name, param_key in modes:
        print(f"\n  {mode_name} ({param_key}):")
        for scale in [0.0, 0.1, 0.2, 0.3]:  # 0%, ±10%, ±20%, ±30%
            set_seed(42 + hash(mode_name) % 1000)
            returns = {name: [] for name in policies}
            base = dict(DEFAULT_PARAMS)
            for _ in range(n_patients):
                p = base.copy()
                p[param_key] = base[param_key] * (1 + np.random.uniform(-scale, scale))
                p[param_key] = max(p[param_key], 1e-6)
                for name, policy_fn in policies.items():
                    ret, _, _, _, _, _ = rollout_metrics(policy_fn, p, n_ep=1)
                    returns[name].append(ret)
            for name in policies:
                mean_r = np.mean(returns[name])
                std_r = np.std(returns[name])
                results[mode_name][f"{name}_scale{scale}"] = (mean_r, std_r)
                print(f"    scale={scale:.1f}  {name}: {mean_r:8.2f}±{std_r:.2f}")
    return results


def exp3_safety_analysis(policies, threshold=1.5, n_ep=50):
    """Experiment 3: Toxicity Violation Rate (C > threshold)"""
    print("\n" + "=" * 70)
    print(f"Experiment 3: Safety Analysis (Toxicity Violation: C > {threshold})")
    print("=" * 70)
    rows = []
    for name, policy_fn in policies.items():
        rate = toxicity_violation_rate(policy_fn, DEFAULT_PARAMS, threshold=threshold, n_ep=n_ep)
        rows.append({"Policy": name, "ToxicityViolation": rate})
        print(f"  {name:10} {rate:.1f}%")
    return rows


def main():
    policies = get_policies()
    if len(policies) < 2:
        print("Need at least BC and Expert. Run: bash run_all.sh")
        return

    out_dir = "experiments/results"
    os.makedirs(out_dir, exist_ok=True)

    # Exp 1
    exp1 = exp1_main_performance(policies)
    with open(f"{out_dir}/exp1_main.json", "w") as f:
        json.dump(exp1, f, indent=2)

    # Exp 2
    exp2 = exp2_patient_robustness(policies)
    exp2_flat = {}
    for k, v in exp2.items():
        exp2_flat[k] = {kk: [float(x) for x in vv] for kk, vv in v.items()}
    with open(f"{out_dir}/exp2_robustness.json", "w") as f:
        json.dump(exp2_flat, f, indent=2)

    # Exp 3
    exp3 = exp3_safety_analysis(policies)
    with open(f"{out_dir}/exp3_safety.json", "w") as f:
        json.dump(exp3, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    return exp1, exp2, exp3


if __name__ == "__main__":
    main()
