"""
CQL (Conservative Q-Learning) training - d3rlpy baseline
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
保留 d3rlpy 作为强基线，创新算法用 src/algos/safe_cql.py (纯 PyTorch)
"""
import os
import argparse
import numpy as np
from env.robust import set_seed
import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained")


def load_dataset_for_d3rlpy(data_path="offline_dataset.npz"):
    """D4RL 标准格式: 显式 terminals(done) 和 timeouts，不猜测。"""
    d = np.load(data_path)
    s = np.array(d["s"], dtype=np.float32)
    a = np.array(d["a"]).flatten().astype(np.int64)
    r = np.array(d["r"], dtype=np.float32)
    s_next = np.array(d["s_next"], dtype=np.float32)
    done = np.array(d["done"], dtype=bool)
    timeout = np.array(d["timeout"], dtype=bool) if "timeout" in d else np.zeros_like(done, dtype=bool)
    # D4RL: terminals = done | timeout. 数据生成时已显式保存，无猜测。
    if not (done.any() or timeout.any()):
        import warnings
        warnings.warn("No terminals in dataset. Marking last step as timeout (legacy data?).")
        timeout[-1] = True
    return s, a, r, s_next, done, timeout


def train_cql(data_path="offline_dataset.npz", n_epochs=100, save_path="cql_model.d3",
              alpha=5.0, lr=1e-4, batch_size=256, n_steps=None):
    """Train DiscreteCQL on offline dataset"""
    try:
        import d3rlpy
    except ImportError:
        print("d3rlpy not installed. Run: pip install d3rlpy")
        return None

    s, a, r, s_next, done, timeout = load_dataset_for_d3rlpy(data_path)
    n_transitions = len(s)

    try:
        dataset = d3rlpy.dataset.MDPDataset(
            observations=s, actions=a, rewards=r,
            terminals=done, timeouts=timeout,
        )
    except TypeError:
        dataset = d3rlpy.dataset.MDPDataset(s, a, r, done, timeout)

    cql = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=lr,
        gamma=0.99,
        alpha=alpha,
        target_update_interval=2000,
        n_critics=2,
    ).create(device="cpu")

    n_steps = n_steps or min(n_epochs * (n_transitions // 64), 200000)
    print(f"Training CQL for {n_steps} steps ({n_epochs} epochs)...")
    cql.fit(dataset, n_steps=n_steps)

    out_path = save_path.replace(".pt", ".d3") if save_path.endswith(".pt") else save_path
    cql.save(out_path)
    print(f"Saved {out_path}")
    return cql


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="offline_dataset.npz")
    parser.add_argument("--save", default="cql_model.d3")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    if not os.path.exists(args.data):
        from data.generate import generate_dataset, save_dataset
        data = generate_dataset(n_trajectories=1000, use_reward_v3=True, state_noise_sigma=0.02, expert_balance_ratio=0.6)
        save_dataset(data, args.data)
    train_cql(data_path=args.data, save_path=args.save, alpha=args.alpha, lr=args.lr,
              batch_size=args.batch_size, n_steps=args.n_steps)
