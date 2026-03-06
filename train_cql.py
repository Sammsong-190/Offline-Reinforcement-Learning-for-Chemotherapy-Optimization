"""
CQL (Conservative Q-Learning) training on offline chemotherapy data
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""
from env.robust import set_seed
set_seed(42)

import numpy as np
import os


def load_dataset_for_d3rlpy(data_path="offline_dataset.npz"):
    """Convert npz to d3rlpy-compatible format. d3rlpy needs terminals or timeouts > 0."""
    d = np.load(data_path)
    s = np.array(d["s"], dtype=np.float32)
    a = np.array(d["a"]).flatten().astype(np.int64)
    r = np.array(d["r"], dtype=np.float32)
    s_next = np.array(d["s_next"], dtype=np.float32)
    done = np.array(d["done"], dtype=bool)
    timeout = np.array(d["timeout"], dtype=bool) if "timeout" in d else np.zeros_like(done, dtype=bool)
    # Ensure at least one terminal: if none, mark last of each traj as timeout
    if not (done.any() or timeout.any()):
        # Infer trajectory ends: every ~100 steps (approx) or force last step
        timeout[-1] = True
    return s, a, r, s_next, done, timeout


def train_cql(data_path="offline_dataset.npz", n_epochs=50, save_path="cql_model.d3"):
    """Train DiscreteCQL on offline dataset"""
    try:
        import d3rlpy
    except ImportError:
        print("d3rlpy not installed. Run: pip install d3rlpy")
        return None

    s, a, r, s_next, done, timeout = load_dataset_for_d3rlpy(data_path)
    n_transitions = len(s)
    action_size = 4

    # Build dataset for d3rlpy (requires terminals or timeouts > 0)
    try:
        dataset = d3rlpy.dataset.MDPDataset(
            observations=s,
            actions=a,
            rewards=r,
            terminals=done,
            timeouts=timeout,
        )
    except TypeError:
        try:
            dataset = d3rlpy.dataset.MDPDataset(s, a, r, done, timeout)
        except TypeError:
            dataset = d3rlpy.dataset.MDPDataset(s, a, r, done)

    # Create and train DiscreteCQL
    cql = d3rlpy.algos.DiscreteCQLConfig(
        batch_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        alpha=1.0,
    ).create(device="cpu")

    n_steps = min(n_epochs * (n_transitions // 64), 50000)
    print(f"Training CQL for {n_steps} steps ({n_epochs} epochs)...")
    cql.fit(dataset, n_steps=n_steps)

    # Save (d3rlpy uses .d3 or directory)
    out_path = save_path.replace(".pt", ".d3") if save_path.endswith(".pt") else save_path
    cql.save(out_path)
    print(f"Saved {out_path}")
    return cql


if __name__ == "__main__":
    if not os.path.exists("offline_dataset.npz"):
        from data.generate import generate_dataset, save_dataset

        data = generate_dataset(n_trajectories=500)
        save_dataset(data)
    train_cql()
