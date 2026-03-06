"""
IQL (Implicit Q-Learning) / BCQ baseline for offline RL
d3rlpy IQL is continuous-only; we use DiscreteBCQ as discrete offline RL baseline.
"""
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


def train_iql(data_path="offline_dataset.npz", n_epochs=50, save_path="iql_model.d3"):
    """Train DiscreteBCQ (IQL-like offline RL for discrete actions)"""
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

    # DiscreteBCQ: offline RL baseline (IQL is continuous-only in d3rlpy)
    algo = d3rlpy.algos.DiscreteBCQConfig(
        batch_size=64,
        learning_rate=6.25e-5,
        gamma=0.99,
        action_flexibility=0.3,
        beta=0.5,
    ).create(device="cpu")

    n_steps = min(n_epochs * (n_transitions // 64), 50000)
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
        data = generate_dataset(n_trajectories=500)
        save_dataset(data)
    train_iql()
