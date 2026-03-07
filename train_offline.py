"""
Supervised Offline RL: Behavioral Cloning
Paper: Supervised Optimal Chemotherapy Regimen Based on Offline Reinforcement Learning
"""
from env.chemo_env import DT, MAX_STEPS, X0, ACTION_SPACE
from env.chemo_env import step_ode, DEFAULT_PARAMS, normalize_state, reward_fn, T_CLEAR, is_done
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import numpy as np
from env.robust import set_seed
set_seed(42)


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


def train_bc(
    data_path='offline_dataset.npz',
    epochs=200,
    lr=1e-3,
    batch=64,
    save_path='bc_policy.pt',
    val_ratio=0.1,
    patience=30,
    label_smoothing=0.05,
    weight_decay=1e-4,
):
    """BC with validation-based early stopping to prevent overfitting."""
    d = np.load(data_path)
    s = torch.FloatTensor(d['s'])
    a_raw = d['a']
    if a_raw.ndim > 1:
        a_raw = a_raw.squeeze()
    if np.issubdtype(a_raw.dtype, np.floating):
        action_space = d.get('action_space', np.array([0., 0.5, 1., 2.]))
        a = np.array([np.argmin(np.abs(action_space - v))
                     for v in a_raw], dtype=np.int64)
    else:
        a = a_raw
    a = torch.LongTensor(a)

    action_dist = (torch.bincount(a, minlength=4).float() / len(a) * 100).tolist()
    print(f"Dataset action dist: {[round(x, 1) for x in action_dist]}%")

    # Train/val split
    n = len(s)
    idx = torch.randperm(n)
    val_size = int(n * val_ratio)
    train_idx, val_idx = idx[val_size:], idx[:val_size]
    s_train, a_train = s[train_idx], a[train_idx]
    s_val, a_val = s[val_idx], a[val_idx]

    ds_train = TensorDataset(s_train, a_train)
    loader_train = DataLoader(ds_train, batch_size=batch, shuffle=True)
    loader_val = DataLoader(TensorDataset(s_val, a_val), batch_size=batch)

    net = PolicyNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_loss = float('inf')
    no_improve = 0

    for ep in range(epochs):
        net.train()
        total = 0.0
        for x, y in loader_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = net(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item()
        train_loss = total / len(loader_train)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(net(x), y).item()
        val_loss /= len(loader_val)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(net.state_dict(), save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep+1} (val_loss no improve)")
            break
        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best_val={best_val_loss:.4f}")

    net.load_state_dict(torch.load(save_path, map_location=DEVICE))
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
                if is_done(x):
                    break
            returns.append(R)
    print(f"Mean return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
    return returns


if __name__ == '__main__':
    import os
    if not os.path.exists('offline_dataset.npz'):
        from data.generate import generate_dataset_v2, save_dataset
        data = generate_dataset_v2(n_trajectories=500, use_reward_v2=True)
        save_dataset(data)
    net = train_bc()
    evaluate_policy(net)
