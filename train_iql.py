"""
IQL (Implicit Q-Learning) / BCQ baseline for offline RL
d3rlpy IQL is continuous-only; we use DiscreteBCQ as discrete offline RL baseline.
"""
import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained")

from env.robust import set_seed
set_seed(42)

import numpy as np
import os


def load_dataset_for_d3rlpy(data_path="offline_dataset.npz"):
    """Same as train_cql"""
    d = np.load(data_path)
    s = np.array(d["s"], dtype=np.float32)
    a = np.array(d["a"]).flatten().astype(np.int64)
    r = np.array(d["r"], dtype=np.float32)
    s_next = np.array(d["s_next"], dtype=np.float32)
    done = np.array(d["done"], dtype=bool)
    timeout = np.array(d["timeout"], dtype=bool) if "timeout" in d else np.zeros_like(done, dtype=bool)
    if not (done.any() or timeout.any()):
        timeout[-1] = True
    return s, a, r, s_next, done, timeout


def train_iql(data_path="offline_dataset.npz", n_epochs=100, save_path="iql_model.d3"):
    """Train DiscreteBCQ. action_flexibility=0.1 allows more treatment actions (was 0.3, too conservative)."""
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

    # action_flexibility: lower = more actions considered (less conservative, encourages treatment)
    # beta: imitation regularization; slightly lower may allow more Q-guided exploration
    algo = d3rlpy.algos.DiscreteBCQConfig(
        batch_size=128,
        learning_rate=1e-4,
        gamma=0.99,
        action_flexibility=0.1,
        beta=0.4,
    ).create(device="cpu")

    n_steps = min(n_epochs * (n_transitions // 128), 100000)
    print(f"Training BCQ (offline RL baseline) for {n_steps} steps...")
    algo.fit(dataset, n_steps=n_steps)

    out_path = save_path.replace(".pt", ".d3") if save_path.endswith(".pt") else save_path
    algo.save(out_path)
    print(f"Saved {out_path}")
    return algo


if __name__ == "__main__":
    if not os.path.exists("offline_dataset.npz"):
        from data.generate import generate_dataset, save_dataset
        set_seed(42)
        data = generate_dataset(n_trajectories=1000, use_reward_v3=True, state_noise_sigma=0.02, expert_balance_ratio=0.6)
        save_dataset(data)
    train_iql()
