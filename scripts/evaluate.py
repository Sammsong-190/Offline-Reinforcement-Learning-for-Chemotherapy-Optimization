#!/usr/bin/env python3
"""
统一评估入口 - SCI 论文格式
自动计算: Average Return, Constraint Violation Rate, Survival Rate
多种子运行，保存 CSV 供 notebooks 画图
"""
import argparse
import csv
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_policy(name, root):
    from env.chemo_env import ACTION_SPACE, normalize_state
    import numpy as np

    def policy_expert(s):
        from data.generate import expert_policy
        return expert_policy(s, epsilon=0.0)

    def policy_random(s):
        return float(np.random.choice(ACTION_SPACE))

    if name == "expert":
        return policy_expert
    if name == "random":
        return policy_random
    if name == "safe_cql":
        from src.algos.safe_cql import SafeCQL
        algo = SafeCQL()
        path = root / "safe_cql_model.pt"
        if not path.exists():
            return None
        return algo.get_policy(str(path))
    if name == "bc":
        import torch
        from train_offline import PolicyNet
        net = PolicyNet()
        p = root / "bc_policy.pt"
        if not p.exists():
            return None
        net.load_state_dict(torch.load(p, map_location="cpu"))
        net.eval()
        def policy(s):
            s_norm = normalize_state(s)
            idx = net(torch.FloatTensor(s_norm).unsqueeze(0)).argmax(1).item()
            return float(ACTION_SPACE[idx])
        return policy
    if name == "cql":
        try:
            import d3rlpy
            cql = d3rlpy.load_learnable(str(root / "cql_model.d3"))
            def policy(s):
                s_norm = normalize_state(s)
                idx = cql.predict(s_norm.reshape(1, -1))
                return float(ACTION_SPACE[int(idx)])
            return policy
        except Exception:
            return None
    return None


def run_evaluation(policy_fn, n_ep=20, seed=42):
    from env.robust import set_seed
    from env.chemo_env import step_ode, reward_fn_v3, DEFAULT_PARAMS, MAX_STEPS, X0, is_done, T_CLEAR, I_SAFE, N_SAFE
    import numpy as np

    set_seed(seed)
    params = DEFAULT_PARAMS
    returns, tumor_clears, survivals, doses, violation_rates = [], [], [], [], []
    for _ in range(n_ep):
        x = np.array(X0, dtype=np.float32)
        R, actions, violation_steps = 0.0, [], 0
        for step in range(MAX_STEPS):
            x_prev = x.copy()
            a = policy_fn(x)
            actions.append(float(a))
            x = step_ode(x, a, 0.3, params)
            R += reward_fn_v3(x, 0.3, s_prev=x_prev)
            if x[2] < I_SAFE or x[0] < N_SAFE:
                violation_steps += 1
            if is_done(x):
                break
        total_steps = step + 1
        returns.append(R)
        tumor_clears.append(100.0 if x[1] < T_CLEAR else 0.0)
        survivals.append(100.0 if (x[0] > 0.1 and x[2] > 0.1) else 0.0)
        doses.append(np.mean(actions) if actions else 0.0)
        violation_rates.append(violation_steps / total_steps * 100)
    return {
        "return_mean": np.mean(returns),
        "return_std": np.std(returns),
        "tumor_clear_pct": np.mean(tumor_clears),
        "survival_pct": np.mean(survivals),
        "avg_dose": np.mean(doses),
        "constraint_violation_rate_pct": np.mean(violation_rates),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["expert", "bc", "safe_cql", "cql", "random"],
                        help="Policies to evaluate")
    parser.add_argument("--n_ep", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds for multi-seed eval")
    parser.add_argument("--output", "-o", default="results/eval_results.csv", help="CSV output path")
    parser.add_argument("--no-csv", action="store_true", help="Only print, do not save CSV")
    args = parser.parse_args()

    rows = []
    for policy_name in args.policies:
        policy = get_policy(policy_name, ROOT)
        if policy is None:
            print(f"{policy_name}: (skip, model not found)")
            continue
        all_metrics = []
        for seed in args.seeds:
            m = run_evaluation(policy, n_ep=args.n_ep, seed=seed)
            all_metrics.append(m)
        # Aggregate over seeds
        return_means = [x["return_mean"] for x in all_metrics]
        violation_means = [x["constraint_violation_rate_pct"] for x in all_metrics]
        survival_means = [x["survival_pct"] for x in all_metrics]
        row = {
            "policy": policy_name,
            "return_mean": np.mean(return_means),
            "return_std": np.std(return_means) if len(return_means) > 1 else all_metrics[0]["return_std"],
            "constraint_violation_rate_pct": np.mean(violation_means),
            "survival_pct": np.mean(survival_means),
            "tumor_clear_pct": np.mean([x["tumor_clear_pct"] for x in all_metrics]),
            "avg_dose": np.mean([x["avg_dose"] for x in all_metrics]),
            "n_seeds": len(args.seeds),
        }
        rows.append(row)
        print(f"{policy_name:12} Return={row['return_mean']:.2f}±{row['return_std']:.2f}  "
              f"Violation={row['constraint_violation_rate_pct']:.1f}%  Survival={row['survival_pct']:.0f}%  "
              f"TumorClear={row['tumor_clear_pct']:.0f}%  Dose={row['avg_dose']:.2f}")

    if not args.no_csv and rows:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["policy", "return_mean", "return_std", "constraint_violation_rate_pct",
                                              "survival_pct", "tumor_clear_pct", "avg_dose", "n_seeds"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
