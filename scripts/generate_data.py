#!/usr/bin/env python3
"""
统一数据生成入口
Usage: python scripts/generate_data.py [--output data/raw/offline_dataset.npz] [--n 1000]

奖励敏感性实验：生成前设置 CHEMO_REWARD_PROFILE=high_incentive（通过 --reward-profile）
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="data/raw/offline_dataset.npz", help="Output path")
    parser.add_argument("--n", type=int, default=1000, help="n_trajectories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d4rl", action="store_true", help="Also save D4RL format")
    parser.add_argument("--preset", choices=["default", "safe"], default="default",
                        help="default: 行为混合; safe: 更偏 expert/conservative 的混合（压低激进轨迹）")
    parser.add_argument("--cohorts", action="store_true",
                        help="虚拟患者三类亚群 + SDE 步进（PatientGenerator），替代均匀参数噪声")
    parser.add_argument(
        "--reward-profile",
        choices=["default", "high_incentive"],
        default="default",
        help="high_incentive: 附录奖励敏感性 — 放大治愈奖励与带瘤惩罚，需与评估时 profile 一致",
    )
    args = parser.parse_args()

    if args.reward_profile == "high_incentive":
        os.environ["CHEMO_REWARD_PROFILE"] = "high_incentive"
    else:
        os.environ.pop("CHEMO_REWARD_PROFILE", None)

    from env.robust import set_seed
    from data.generate import generate_dataset, save_dataset, save_dataset_d4rl

    set_seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.preset == "safe":
        kw = dict(
            expert_ratio=0.75, balanced_ratio=0.10, aggressive_ratio=0.02, conservative_ratio=0.13,
            expert_balance_ratio=0.20, expert_epsilon=0.12, patient_scale=0.08,
        )
    else:
        kw = dict(expert_balance_ratio=0.6)

    data = generate_dataset(
        n_trajectories=args.n,
        state_noise_sigma=0.02,
        seed=args.seed,
        use_cohorts=args.cohorts,
        **kw,
    )
    save_dataset(data, str(out_path))
    if args.d4rl:
        d4rl_path = str(out_path).replace(".npz", "_d4rl.npz")
        save_dataset_d4rl(data, d4rl_path)
    print(f"Done. Saved to {out_path}  (CHEMO_REWARD_PROFILE={os.environ.get('CHEMO_REWARD_PROFILE', 'default')})")


if __name__ == "__main__":
    main()
