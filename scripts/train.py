#!/usr/bin/env python3
"""
统一训练入口
Usage:
  python scripts/train.py --algo safe_cql [--config configs/experiment/train_safe.yaml]
  python scripts/train.py --algo bc
  python scripts/train.py --algo cql
"""
import argparse
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config(path):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["bc", "cql", "safe_cql"], default="safe_cql")
    parser.add_argument("--config", default="configs/experiment/train_safe.yaml")
    parser.add_argument("--agent-config", default=None, help="e.g. agent/safe_cql_strict.yaml")
    parser.add_argument("--data", default=None, help="Override data path")
    parser.add_argument("--save", default=None, help="Override save path")
    args = parser.parse_args()

    from env.robust import set_seed
    set_seed(42)

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    data_path = args.data or cfg.get("data", {}).get("path", "data/raw/offline_dataset.npz")
    data_path = str(ROOT / data_path) if not Path(data_path).is_absolute() else data_path
    save_path = args.save or cfg.get("output", {}).get("save_name", "safe_cql_model.pt")

    # Ensure data exists
    if not Path(data_path).exists():
        alt = ROOT / "offline_dataset.npz"
        if alt.exists():
            data_path = str(alt)
        else:
            print("Generating data...")
            from data.generate import generate_dataset, save_dataset
            data = generate_dataset(n_trajectories=1000, use_reward_v3=True, state_noise_sigma=0.02, expert_balance_ratio=0.6)
            save_dataset(data, data_path)

    if args.algo == "safe_cql":
        ac_path = args.agent_config or "configs/agent/safe_cql.yaml"
        agent_cfg = load_config(ac_path) if (ROOT / ac_path).exists() else {}
        p = agent_cfg.get("params", agent_cfg)
        net = agent_cfg.get("network", {})
        algo = __import__("src.algos.safe_cql", fromlist=["SafeCQL"]).SafeCQL(
            actor_lr=p.get("actor_lr", 1e-4),
            critic_lr=p.get("critic_lr", p.get("learning_rate", 3e-4)),
            lagrangian_lr=p.get("lagrangian_lr", p.get("lambda_lr", 1e-3)),
            gamma=p.get("gamma", 0.99),
            alpha_cql=p.get("alpha_cql", 5.0),
            cost_limit=p.get("cost_limit", 0.1),
            hidden=net.get("hidden", 64),
            state_dim=net.get("state_dim", 4),
            n_actions=net.get("n_actions", 4),
        )
        algo.train(data_path, n_steps=p.get("n_steps", 200000), batch_size=p.get("batch_size", 256), save_path=save_path)
    elif args.algo == "bc":
        from train_offline import train_bc
        train_bc(data_path=data_path, save_path=save_path.replace("safe_cql", "bc") if "safe_cql" in save_path else "bc_policy.pt")
    elif args.algo == "cql":
        from train_cql import train_cql
        train_cql(data_path=data_path, save_path="cql_model.d3")


if __name__ == "__main__":
    main()
