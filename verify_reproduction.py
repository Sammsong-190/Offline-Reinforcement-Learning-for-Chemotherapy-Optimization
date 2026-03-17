"""
Verify reproduction: compare BC vs Expert vs Random vs Fixed-dose baselines
"""
import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

from env.robust import set_seed, rollout_param_shift
set_seed(42)

import numpy as np
from env.chemo_env import (
    step_ode, DEFAULT_PARAMS, reward_fn, reward_fn_v2, reward_fn_v3,
    DT, MAX_STEPS, X0, ACTION_SPACE, normalize_state, T_CLEAR, is_done,
)
from data.generate import expert_policy, behavior_policy


def rollout(policy_fn, n_ep=10, params=None, use_reward_v3=True):
    """Rollout policy in ODE env, return mean ± std of returns"""
    rfn = reward_fn_v3 if use_reward_v3 else reward_fn_v2
    params = params or DEFAULT_PARAMS
    returns = []
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        R = 0.0
        for _ in range(MAX_STEPS):
            x_prev = x.copy()
            a = policy_fn(x)
            x = step_ode(x, a, DT, params)
            R += rfn(x, DT, s_prev=x_prev)
            if is_done(x):
                break
        returns.append(R)
    return np.mean(returns), np.std(returns)


T_CONTROL_THRESH = 0.05  # tumor "control" threshold for time-to-control metric


def rollout_with_metrics(policy_fn, n_ep=10, params=None, use_reward_v3=True):
    """Rollout and return (mean_return, std_return, tumor_clear, survival, avg_dose,
       avg_tumor, max_toxicity, drug_usage, treatment_efficiency, time_to_control)"""
    rfn = reward_fn_v3 if use_reward_v3 else reward_fn_v2
    params = params or DEFAULT_PARAMS
    returns, tumor_clears, survivals, doses = [], [], [], []
    final_tumors, max_toxicities, drug_usages = [], [], []
    treatment_effs, times_to_control = [], []
    tumor_start = X0[1]  # 0.7
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        R, actions = 0.0, []
        toxics = []
        t_control = np.nan  # step*DT when T first < T_CONTROL_THRESH
        for step in range(MAX_STEPS):
            x_prev = x.copy()
            a = policy_fn(x)
            actions.append(float(a))
            x = step_ode(x, a, DT, params)
            toxics.append(x[3])
            if np.isnan(t_control) and x[1] < T_CONTROL_THRESH:
                t_control = step * DT
            R += rfn(x, DT, s_prev=x_prev)
            if is_done(x):
                break
        returns.append(R)
        tumor_clears.append(1.0 if x[1] < T_CLEAR else 0.0)
        survivals.append(1.0 if (x[0] > 0.1 and x[2] > 0.1) else 0.0)
        doses.append(np.mean(actions) if actions else 0.0)
        final_tumors.append(x[1])
        max_toxicities.append(np.max(toxics) if toxics else 0.0)
        drug_total = np.sum(actions) * DT if actions else 0.0
        drug_usages.append(np.sum(actions) if actions else 0.0)
        # Treatment Efficiency: (Tumor_start - Tumor_end) / sum(u_t)
        eff = (tumor_start - x[1]) / (drug_total + 1e-8) if drug_total > 0 else 0.0
        treatment_effs.append(eff)
        times_to_control.append(t_control if not np.isnan(t_control) else MAX_STEPS * DT)
    return (
        np.mean(returns), np.std(returns),
        np.mean(tumor_clears) * 100,
        np.mean(survivals) * 100,
        np.mean(doses),
        np.mean(final_tumors),
        np.mean(max_toxicities),
        np.mean(drug_usages),
        np.mean(treatment_effs),
        np.mean(times_to_control),
    )


def policy_expert(s):
    return expert_policy(s, epsilon=0.0)


def policy_random(s):
    return float(np.random.choice(ACTION_SPACE))


def policy_fixed(dose):
    def fn(s):
        return dose
    return fn


def policy_bc(net):
    import torch
    device = next(net.parameters()).device

    def fn(s):
        s_norm = normalize_state(s)
        x = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = net(x).argmax(dim=1).cpu().item()
        return float(ACTION_SPACE[idx])
    return fn


def policy_cql(cql):
    """Wrap CQL (d3rlpy or native) to policy_fn(s) -> dose"""

    def fn(s):
        s_norm = normalize_state(s)
        x = np.array(s_norm, dtype=np.float32).reshape(1, -1)
        idx = cql.predict(x)
        if isinstance(idx, np.ndarray):
            idx = int(idx.flat[0])
        else:
            idx = int(idx)
        return float(ACTION_SPACE[idx])
    return fn


def policy_safe_cql():
    """Load Safe CQL (Lagrangian) policy."""
    from src.algos.safe_cql import SafeCQL
    return SafeCQL().get_policy("safe_cql_model.pt")


def main():
    from train_offline import PolicyNet
    import torch
    import os

    n_ep = 20  # robust: more episodes for stable metrics
    results = []
    use_metrics = True  # set False for simple rollout

    def run_policy(name, policy_fn):
        if use_metrics:
            mean, std, tc, surv, dose, avg_t, max_c, drug, eff, t_ctrl = rollout_with_metrics(policy_fn, n_ep=n_ep)
            results.append((name, mean, std, tc, surv, dose, avg_t, max_c, drug, eff, t_ctrl))
            print(f"{name:12} Return={mean:8.2f}±{std:.2f}  TumorClear={tc:.0f}%  Survival={surv:.0f}%  AvgDose={dose:.2f}  Eff={eff:.3f}  Tctrl={t_ctrl:.1f}")
        else:
            mean, std = rollout(policy_fn, n_ep=n_ep)
            results.append((name, mean, std, None, None, None, None, None, None, None, None))
            print(f"{name:12} {mean:8.2f} ± {std:.2f}")

    # 1. Expert (deterministic)
    run_policy("Expert", policy_expert)

    # 1b. Behavior (mixture policy used to generate data)
    run_policy("Behavior", behavior_policy)

    # 2. BC (bc_policy.pt)
    if os.path.exists("bc_policy.pt"):
        net = PolicyNet()
        net.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
        net.eval()
        run_policy("BC", policy_bc(net))
    else:
        print("BC:          (no bc_policy.pt, run train_offline.py first)")

    # 2b. CQL (d3rlpy)
    if os.path.exists("cql_model.d3"):
        try:
            import d3rlpy
            cql = d3rlpy.load_learnable("cql_model.d3")
            run_policy("CQL", policy_cql(cql))
        except Exception as e:
            print(f"CQL:         (load failed: {e})")
    else:
        print("CQL:         (no cql_model.d3, run train_cql.py)")

    # 2c. IQL/BCQ baseline (if exists)
    if os.path.exists("iql_model.d3"):
        try:
            import d3rlpy
            iql = d3rlpy.load_learnable("iql_model.d3")
            run_policy("IQL", policy_cql(iql))
        except Exception as e:
            print(f"IQL:         (load failed: {e})")

    # 2d. Safe CQL (Lagrangian constrained)
    if os.path.exists("safe_cql_model.pt"):
        try:
            run_policy("Safe CQL", policy_safe_cql())
        except Exception as e:
            print(f"Safe CQL:    (load failed: {e})")
    else:
        print("Safe CQL:    (no safe_cql_model.pt, run scripts/train.py --algo safe_cql)")

    # 3. Random
    run_policy("Random", policy_random)

    # 4. Fixed doses
    for dose in [0.0, 0.5, 1.0, 2.0]:
        run_policy(f"Fixed {dose}", policy_fixed(dose))

    # 5. Action coverage (if dataset exists)
    if os.path.exists("offline_dataset.npz"):
        d = np.load("offline_dataset.npz")
        a = np.array(d["a"]).flatten()
        pct = np.bincount(a.astype(int), minlength=4) / len(a) * 100
        print(f"\nAction coverage: {pct}%")
        if (pct < 5).any():
            print("  [!] Some action < 5% -> offline RL may be unstable")

    # Metrics table (if available)
    if use_metrics and any(r[3] is not None for r in results):
        print("\n" + "-" * 115)
        print(f"{'Policy':<12} {'Return':>8} {'TumorClear':>10} {'Survival':>8} {'AvgDose':>8} {'FinalT':>8} {'MaxTox':>7} {'DrugUse':>8} {'TrtEff':>8} {'Tctrl':>7}")
        print("-" * 115)
        for r in results:
            if r[3] is not None:
                print(f"{r[0]:<12} {r[1]:>8.2f} {r[3]:>9.0f}% {r[4]:>7.0f}% {r[5]:>8.2f} {r[6]:>8.3f} {r[7]:>7.3f} {r[8]:>8.1f} {r[9]:>8.3f} {r[10]:>7.1f}")
        print("-" * 115)

    # 6. Parameter-shift: In-Dist (σ=0.15) vs OOD (σ=0.30)
    if use_metrics:
        print("\n--- Robust Eval: 100 patients (In-Dist σ=0.15) ---")
        set_seed(123)
        mean, std = rollout_param_shift(policy_expert, n_patients=100, n_ep_per_patient=1, scale=0.15)
        print(f"  Expert     Return={mean:8.2f}±{std:.2f}")
        if os.path.exists("bc_policy.pt"):
            net = PolicyNet()
            net.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
            net.eval()
            mean, std = rollout_param_shift(policy_bc(net), n_patients=100, n_ep_per_patient=1, scale=0.15)
            print(f"  BC         Return={mean:8.2f}±{std:.2f}")
        if os.path.exists("cql_model.d3"):
            try:
                import d3rlpy
                cql = d3rlpy.load_learnable("cql_model.d3")
                mean, std = rollout_param_shift(policy_cql(cql), n_patients=100, n_ep_per_patient=1, scale=0.15)
                print(f"  CQL        Return={mean:8.2f}±{std:.2f}")
            except Exception:
                pass
        if os.path.exists("iql_model.d3"):
            try:
                import d3rlpy
                iql = d3rlpy.load_learnable("iql_model.d3")
                mean, std = rollout_param_shift(policy_cql(iql), n_patients=100, n_ep_per_patient=1, scale=0.15)
                print(f"  IQL        Return={mean:8.2f}±{std:.2f}")
            except Exception:
                pass
        print("\n--- OOD Eval: 50 patients (σ=0.30, out-of-distribution) ---")
        set_seed(456)
        mean, std = rollout_param_shift(policy_expert, n_patients=50, n_ep_per_patient=1, scale=0.30)
        print(f"  Expert     Return={mean:8.2f}±{std:.2f}")
        if os.path.exists("bc_policy.pt"):
            net = PolicyNet()
            net.load_state_dict(torch.load("bc_policy.pt", map_location="cpu"))
            net.eval()
            mean, std = rollout_param_shift(policy_bc(net), n_patients=50, n_ep_per_patient=1, scale=0.30)
            print(f"  BC         Return={mean:8.2f}±{std:.2f}")
        if os.path.exists("cql_model.d3"):
            try:
                import d3rlpy
                cql = d3rlpy.load_learnable("cql_model.d3")
                mean, std = rollout_param_shift(policy_cql(cql), n_patients=50, n_ep_per_patient=1, scale=0.30)
                print(f"  CQL        Return={mean:8.2f}±{std:.2f}")
            except Exception:
                pass
        if os.path.exists("iql_model.d3"):
            try:
                import d3rlpy
                iql = d3rlpy.load_learnable("iql_model.d3")
                mean, std = rollout_param_shift(policy_cql(iql), n_patients=50, n_ep_per_patient=1, scale=0.30)
                print(f"  IQL        Return={mean:8.2f}±{std:.2f}")
            except Exception:
                pass
        set_seed(42)

    # Verdict
    print("\n" + "=" * 50)
    if os.path.exists("bc_policy.pt"):
        bc_mean = next((r[1] for r in results if r[0] == "BC"), None)
        if bc_mean is not None:
            expert_mean = next(r[1] for r in results if r[0] == "Expert")
            random_mean = next(r[1] for r in results if r[0] == "Random")
            tol = 0.5
            bc_approx_expert = abs(bc_mean - expert_mean) < tol
            bc_beats_random = bc_mean > random_mean
            if bc_approx_expert:
                print("=> Reproduction SUCCESS: BC ≈ Expert (imitation achieved)")
                if bc_beats_random:
                    print("   (BC also > Random)")
                else:
                    print("   (Random may beat BC when it luckily picks optimal dose)")
            elif bc_beats_random:
                print("=> Reproduction PARTIAL: BC > Random but BC ≠ Expert")
            else:
                print("=> Reproduction FAIL: BC << Expert and BC <= Random")
            # CQL vs BC
            if any(r[0] == "CQL" for r in results):
                cql_mean = next(r[1] for r in results if r[0] == "CQL")
                if cql_mean > bc_mean:
                    print(f"   CQL > BC ({cql_mean:.2f} vs {bc_mean:.2f})")
                else:
                    print(f"   BC >= CQL ({bc_mean:.2f} vs {cql_mean:.2f})")
            # IQL vs BC
            if any(r[0] == "IQL" for r in results):
                iql_mean = next(r[1] for r in results if r[0] == "IQL")
                if iql_mean > bc_mean:
                    print(f"   IQL > BC ({iql_mean:.2f} vs {bc_mean:.2f})")
                else:
                    print(f"   BC >= IQL ({bc_mean:.2f} vs {iql_mean:.2f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
