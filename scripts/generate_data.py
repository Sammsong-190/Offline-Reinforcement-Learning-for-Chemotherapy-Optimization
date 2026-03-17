#!/usr/bin/env python3
"""
统一数据生成入口
Usage: python scripts/generate_data.py [--output data/raw/offline_dataset.npz] [--n 1000]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/raw/offline_dataset.npz", help="Output path")
    parser.add_argument("--n", type=int, default=1000, help="n_trajectories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d4rl", action="store_true", help="Also save D4RL format (observations, actions, rewards, next_observations, terminals, costs)")
    args = parser.parse_args()

    from env.robust import set_seed
    from data.generate import generate_dataset, save_dataset, save_dataset_d4rl

    set_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_dataset(
        n_trajectories=args.n,
        use_reward_v3=True,
        state_noise_sigma=0.02,
        expert_balance_ratio=0.6,
    )
    save_dataset(data, str(out_path))
    if args.d4rl:
        d4rl_path = str(out_path).replace(".npz", "_d4rl.npz")
        save_dataset_d4rl(data, d4rl_path)
    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
