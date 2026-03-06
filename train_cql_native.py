"""
Train CQL using self-implemented cql_agent (no d3rlpy/gym)
"""
from env.robust import set_seed

set_seed(42)

import numpy as np
import os
from cql_agent import DiscreteCQL
from config import CQL_CONFIG


def load_dataset(data_path="offline_dataset.npz"):
    d = np.load(data_path)
    s = np.array(d["s"], dtype=np.float32)
    a = np.array(d["a"]).flatten().astype(np.int64)
    r = np.array(d["r"], dtype=np.float32)
    s_next = np.array(d["s_next"], dtype=np.float32)
    done = np.array(d["done"], dtype=np.float32)
    return s, a, r, s_next, done


def train_cql_native(
    data_path="offline_dataset.npz",
    save_path="cql_native.pt",
    total_steps=None,
    **kwargs,
):
    cfg = {**CQL_CONFIG, **kwargs}
    total_steps = total_steps or cfg.get("total_steps", 100_000)

    s, a, r, s_next, done = load_dataset(data_path)
    n = len(s)
    batch_size = cfg["batch_size"]

    agent = DiscreteCQL(
        state_dim=s.shape[1],
        action_dim=4,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
        target_update=cfg["target_update"],
        hidden=cfg["hidden_dim"],
    )

    print(f"Training CQL (native) for {total_steps} steps, batch={batch_size}...")
    for step in range(total_steps):
        idx = np.random.randint(0, n, batch_size)
        batch = (s[idx], a[idx], r[idx], s_next[idx], done[idx])
        loss = agent.update(batch)
        if (step + 1) % 5000 == 0:
            print(f"  Step {step+1}/{total_steps} loss={loss['total']:.4f}")

    agent.save(save_path)
    print(f"Saved {save_path}")
    return agent


if __name__ == "__main__":
    if not os.path.exists("offline_dataset.npz"):
        from data.generate import generate_dataset_v2, save_dataset

        data = generate_dataset_v2(n_trajectories=500, use_reward_v2=True)
        save_dataset(data)
    train_cql_native()
