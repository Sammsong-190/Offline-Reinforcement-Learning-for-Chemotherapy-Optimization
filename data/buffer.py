"""
ReplayBuffer / 数据加载器
支持 npz 格式离线数据，兼容 Safe RL 的 cost 列
"""
import numpy as np
from pathlib import Path


def load_npz(path: str):
    """Load offline dataset from npz. Supports both native and D4RL format."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    d = np.load(path)
    # D4RL format: observations, actions, rewards, next_observations, terminals, costs
    if "observations" in d:
        out = {
            "s": np.array(d["observations"], dtype=np.float32),
            "a": np.array(d["actions"]).flatten().astype(np.int64),
            "r": np.array(d["rewards"], dtype=np.float32),
            "s_next": np.array(d["next_observations"], dtype=np.float32),
            "terminals": np.array(d["terminals"], dtype=bool),
        }
        out["done"] = out["terminals"].copy()
        out["timeout"] = np.zeros_like(out["done"])
        out["c"] = np.array(d["costs"], dtype=np.float32) if "costs" in d else np.zeros_like(out["r"])
    else:
        out = {
            "s": np.array(d["s"], dtype=np.float32),
            "a": np.array(d["a"]).flatten().astype(np.int64),
            "r": np.array(d["r"], dtype=np.float32),
            "s_next": np.array(d["s_next"], dtype=np.float32),
            "done": np.array(d["done"], dtype=bool),
            "timeout": np.array(d["timeout"], dtype=bool) if "timeout" in d else np.zeros(d["done"].shape, dtype=bool),
        }
        out["c"] = np.array(d["c"], dtype=np.float32) if "c" in d else np.zeros_like(out["r"])
    out["action_space"] = d.get("action_space", np.array([0.0, 0.5, 1.0, 2.0]))
    return out


class ReplayBuffer:
    """Simple replay buffer for offline RL. Supports cost for Safe RL."""

    def __init__(self, data_path: str):
        data = load_npz(data_path)
        self.s = data["s"]
        self.a = data["a"]
        self.r = data["r"]
        self.c = data["c"]
        self.s_next = data["s_next"]
        self.done = data["done"]
        self.timeout = data["timeout"]
        self.action_space = data["action_space"]
        self.n = len(self.s)
        self.has_cost = "c" in data

    def __len__(self):
        return self.n

    def sample(self, batch_size: int):
        """Sample batch. Returns (s, a, r, c, s_next, term)."""
        idx = np.random.choice(self.n, size=batch_size, replace=True)
        term = self.done[idx] | self.timeout[idx]
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.c[idx],
            self.s_next[idx],
            term.astype(np.float32),
        )
