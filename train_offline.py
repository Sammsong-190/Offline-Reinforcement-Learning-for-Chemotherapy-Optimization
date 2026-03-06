"""
Supervised Offline RL: Behavioral Cloning
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""
from env.robust import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from env.chemo_env import step_ode, DEFAULT_PARAMS, normalize_state, reward_fn, T_CLEAR
from env.chemo_env import DT, MAX_STEPS, X0, ACTION_SPACE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ACTIONS = 4


class PolicyNet(nn.Module):
    """MLP: state (normalized) -> action logits (discrete)"""
    def __init__(self, state_dim=4, n_actions=4, hidden=64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s):
        return self.net(s)


def train_bc(data_path='offline_dataset.npz', epochs=200, lr=1e-3, batch=64, save_path='bc_policy.pt'):
    """Behavioral Cloning: supervised classification on (s_norm, a_idx)"""
    d = np.load(data_path)
    s = torch.FloatTensor(d['s'])
    a_raw = d['a']
    if a_raw.ndim > 1:
        a_raw = a_raw.squeeze()
    if np.issubdtype(a_raw.dtype, np.floating):
        action_space = d.get('action_space', np.array([0., 0.5, 1., 2.]))
        a = np.array([np.argmin(np.abs(action_space - v)) for v in a_raw], dtype=np.int64)
    else:
        a = a_raw
    a = torch.LongTensor(a)
    # Diagnostic: action distribution (should be expert-heavy for BC≈Expert)
    action_dist = np.bincount(a.numpy(), minlength=4) / len(a) * 100
    print(f"Dataset action dist: {action_dist.round(1)}%")

    ds = TensorDataset(s, a)
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    net = PolicyNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = net(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={total/len(loader):.6f}")

    torch.save(net.state_dict(), save_path)
    return net


def evaluate_policy(net, n_ep=10):
    """Rollout with learned policy in ODE env"""
    net.eval()
    returns = []
    with torch.no_grad():
        for _ in range(n_ep):
            x = np.array(X0, dtype=np.float32)
            R = 0.0
            for _ in range(MAX_STEPS):
                x_prev = x.copy()
                s_norm = normalize_state(x)
                s = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
                idx = net(s).argmax(dim=1).cpu().item()
                a = float(ACTION_SPACE[idx])
                x = step_ode(x, a, DT, DEFAULT_PARAMS)
                R += reward_fn(x, DT, s_prev=x_prev)
                if x[1] < T_CLEAR or x[0] < 0.1 or x[2] < 0.1:
                    break
            returns.append(R)
    print(f"Mean return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    return returns


if __name__ == '__main__':
    import os
    if not os.path.exists('offline_dataset.npz'):
        from data.generate import generate_dataset, save_dataset
        data = generate_dataset(n_trajectories=100)
        save_dataset(data)
    net = train_bc()
    evaluate_policy(net)
