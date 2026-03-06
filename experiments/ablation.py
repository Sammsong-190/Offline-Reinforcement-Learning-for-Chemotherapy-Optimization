"""
Ablation study: reward weight, dataset size, behavior noise
Run: python -m experiments.ablation
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.robust import set_seed
set_seed(42)


def run_ablation_reward_weight():
    """Vary toxicity weight in reward: 0.3, 0.5, 0.8, 1.0"""
    from env.chemo_env import step_ode, reward_fn, DEFAULT_PARAMS, DT, MAX_STEPS, X0, T_CLEAR
    from data.generate import expert_policy

    # Need to modify reward_fn - we'll pass a custom reward via wrapper
    # For simplicity, we re-import and patch, or run separate data gen with different reward
    print("Ablation: reward toxicity weight (requires regenerating data per weight)")
    print("  Run with different w3 in chemo_env.reward_fn, then train & eval")
    return []


def run_ablation_dataset_size():
    """Dataset size: 10k, 25k, 50k, 100k transitions"""
    from data.generate import generate_dataset, save_dataset
    from train_offline import train_bc, PolicyNet
    from env.chemo_env import step_ode, reward_fn, is_done, DEFAULT_PARAMS, DT, MAX_STEPS, X0, T_CLEAR
    from env.chemo_env import ACTION_SPACE, normalize_state

    sizes = [50, 100, 200, 500]  # trajectories -> ~12k, 25k, 50k, 125k transitions
    results = []
    for n_traj in sizes:
        set_seed(42)
        data = generate_dataset(n_trajectories=n_traj)
        path = f"offline_dataset_n{n_traj}.npz"
        from data.generate import save_dataset
        save_dataset(data, path)
        # Train BC (save to temp path to avoid overwriting)
        net = train_bc(data_path=path, epochs=100, save_path=f"bc_ablation_n{n_traj}.pt")
        # Eval
        returns = []
        for _ in range(10):
            x = np.array(X0, dtype=np.float32)
            R = 0.0
            for _ in range(MAX_STEPS):
                s_norm = normalize_state(x)
                import torch
                idx = net(torch.FloatTensor(s_norm).unsqueeze(0)).argmax(dim=1).item()
                a = float(ACTION_SPACE[idx])
                x = step_ode(x, a, DT, DEFAULT_PARAMS)
                R += reward_fn(x, DT)
                if is_done(x):
                    break
            returns.append(R)
        mean_r = np.mean(returns)
        results.append({"n_traj": n_traj, "n_transitions": len(data), "mean_return": float(mean_r)})
        print(f"  n_traj={n_traj}  n_trans={len(data)}  return={mean_r:.2f}")
    return results


def run_ablation_behavior_noise():
    """Expert epsilon in behavior: 0.05, 0.15, 0.3, 0.5"""
    print("Ablation: behavior noise (expert epsilon)")
    print("  Modify data/generate.py behavior_policy epsilon, regenerate, train, eval")
    return []


def main():
    print("\n" + "=" * 50)
    print("Ablation Study")
    print("=" * 50)
    print("\n1. Dataset size:")
    r1 = run_ablation_dataset_size()
    print("\n2. Reward weight / Behavior noise: (manual - modify config & rerun)")
    r2 = run_ablation_reward_weight()
    r3 = run_ablation_behavior_noise()

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/ablation_dataset_size.json", "w") as f:
        json.dump(r1, f, indent=2)
    print(f"\nSaved experiments/results/ablation_dataset_size.json")


if __name__ == "__main__":
    main()
