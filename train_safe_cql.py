"""
Safe CQL: Lagrangian Constrained Offline RL for Chemotherapy
CMDP: max E[sum r] s.t. E[sum c] <= epsilon
Cost: c=1 if I<0.3 or N<0.4, else 0
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env.robust import set_seed

set_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_ACTIONS = 4
STATE_DIM = 4


class QNet(nn.Module):
    """Discrete Q: state -> [Q(s,a0), Q(s,a1), ...]"""

    def __init__(self, state_dim=4, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s):
        return self.net(s)


class PolicyNet(nn.Module):
    """Discrete policy: state -> action probs"""

    def __init__(self, state_dim=4, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s):
        return F.softmax(self.net(s), dim=-1)


def load_dataset(data_path="offline_dataset.npz"):
    """Load s, a, r, c, s_next, done, timeout."""
    d = np.load(data_path)
    s = np.array(d["s"], dtype=np.float32)
    a = np.array(d["a"]).flatten().astype(np.int64)
    r = np.array(d["r"], dtype=np.float32)
    s_next = np.array(d["s_next"], dtype=np.float32)
    done = np.array(d["done"], dtype=bool)
    timeout = np.array(d["timeout"], dtype=bool) if "timeout" in d else np.zeros_like(done, dtype=bool)
    c = np.array(d["c"], dtype=np.float32) if "c" in d else np.zeros_like(r)
    return s, a, r, c, s_next, done, timeout


def train_safe_cql(
    data_path="offline_dataset.npz",
    save_path="safe_cql_model.pt",
    n_steps=200000,
    batch_size=256,
    lr=1e-4,
    gamma=0.99,
    alpha_cql=5.0,
    cost_limit=0.1,
    target_update_interval=2000,
    n_critics=2,
):
    """
    Lagrangian Safe CQL.
    cost_limit (epsilon): max allowed expected cumulative cost.
    """
    s, a, r, c, s_next, done, timeout = load_dataset(data_path)
    n = len(s)
    print(f"Loaded {n} transitions, cost rate: {c.mean():.4f}")

    # Convert to tensors
    s_t = torch.FloatTensor(s).to(DEVICE)
    a_t = torch.LongTensor(a).to(DEVICE)
    r_t = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
    c_t = torch.FloatTensor(c).unsqueeze(1).to(DEVICE)
    s_next_t = torch.FloatTensor(s_next).to(DEVICE)
    done_t = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)
    timeout_t = torch.FloatTensor(timeout).unsqueeze(1).to(DEVICE)
    term = done_t + timeout_t

    # Networks
    q_r_nets = [QNet(STATE_DIM, N_ACTIONS).to(DEVICE) for _ in range(n_critics)]
    q_r_targets = [QNet(STATE_DIM, N_ACTIONS).to(DEVICE) for _ in range(n_critics)]
    for i in range(n_critics):
        q_r_targets[i].load_state_dict(q_r_nets[i].state_dict())

    q_c_nets = [QNet(STATE_DIM, N_ACTIONS).to(DEVICE) for _ in range(n_critics)]
    q_c_targets = [QNet(STATE_DIM, N_ACTIONS).to(DEVICE) for _ in range(n_critics)]
    for i in range(n_critics):
        q_c_targets[i].load_state_dict(q_c_nets[i].state_dict())

    actor = PolicyNet(STATE_DIM, N_ACTIONS).to(DEVICE)
    log_lambda = nn.Parameter(torch.tensor(0.0, device=DEVICE))

    opt_qr = torch.optim.Adam([p for net in q_r_nets for p in net.parameters()], lr=lr)
    opt_qc = torch.optim.Adam([p for net in q_c_nets for p in net.parameters()], lr=lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_lambda = torch.optim.Adam([log_lambda], lr=1e-3)

    def sample_batch():
        idx = np.random.choice(n, size=batch_size, replace=True)
        return (
            s_t[idx], a_t[idx], r_t[idx], c_t[idx],
            s_next_t[idx], term[idx],
        )

    def get_next_actions(s_next):
        with torch.no_grad():
            probs = actor(s_next)
            return torch.argmax(probs, dim=-1)

    print("Training Safe CQL (Lagrangian)...")
    for step in range(n_steps):
        ss, aa, rr, cc, ss_next, term_b = sample_batch()
        aa_onehot = F.one_hot(aa, N_ACTIONS).float()

        # --- 1. Reward Q (CQL) ---
        q_r = sum(net(ss).gather(1, aa.unsqueeze(1)).squeeze(1) for net in q_r_nets) / n_critics
        with torch.no_grad():
            next_a = get_next_actions(ss_next)
            q_next = torch.min(
                torch.stack([t(ss_next).gather(1, next_a.unsqueeze(1)).squeeze(1) for t in q_r_targets]),
                dim=0
            )[0]
            td_target = rr.squeeze() + gamma * (1 - term_b.squeeze()) * q_next

        td_loss = F.mse_loss(q_r, td_target)
        # CQL conservative: log_sum_exp(Q) - E[Q(s,a_b)]
        log_sum_exp_q = torch.logsumexp(torch.stack([net(ss) for net in q_r_nets], dim=0).mean(0), dim=-1)
        q_data = (torch.stack([net(ss) for net in q_r_nets], dim=0).mean(0) * aa_onehot).sum(-1)
        cql_penalty = (log_sum_exp_q - q_data).mean()
        q_r_loss = td_loss + alpha_cql * cql_penalty

        opt_qr.zero_grad()
        q_r_loss.backward()
        opt_qr.step()

        # --- 2. Cost Q (MSE only) ---
        q_c = sum(net(ss).gather(1, aa.unsqueeze(1)).squeeze(1) for net in q_c_nets) / n_critics
        with torch.no_grad():
            next_a = get_next_actions(ss_next)
            q_c_next = torch.min(
                torch.stack([t(ss_next).gather(1, next_a.unsqueeze(1)).squeeze(1) for t in q_c_targets]),
                dim=0
            )[0]
            c_td_target = cc.squeeze() + gamma * (1 - term_b.squeeze()) * q_c_next

        q_c_loss = F.mse_loss(q_c, c_td_target)
        opt_qc.zero_grad()
        q_c_loss.backward()
        opt_qc.step()

        # --- 3. Lagrange multiplier ---
        with torch.no_grad():
            pi_probs = actor(ss)
            q_c_pi = torch.stack([net(ss) for net in q_c_nets], dim=0).mean(0)
            current_risk = (pi_probs * q_c_pi).sum(-1).mean()
        lambda_loss = log_lambda * (current_risk - cost_limit)
        opt_lambda.zero_grad()
        lambda_loss.backward()
        opt_lambda.step()

        # --- 4. Actor: max (Q_R - lambda * Q_C) ---
        pi_probs = actor(ss)
        q_r_pi = (torch.stack([net(ss) for net in q_r_nets], dim=0).mean(0) * pi_probs).sum(-1)
        q_c_pi = (torch.stack([net(ss) for net in q_c_nets], dim=0).mean(0) * pi_probs).sum(-1)
        with torch.no_grad():
            lam = torch.exp(log_lambda).clamp(0.01, 100.0)
        actor_loss = -(q_r_pi - lam * q_c_pi).mean()
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        # Target update
        if (step + 1) % target_update_interval == 0:
            for i in range(n_critics):
                q_r_targets[i].load_state_dict(q_r_nets[i].state_dict())
                q_c_targets[i].load_state_dict(q_c_nets[i].state_dict())

        if (step + 1) % 10000 == 0:
            lam_val = torch.exp(log_lambda).item()
            print(f"  step {step+1}: lambda={lam_val:.3f}, q_r_loss={q_r_loss.item():.4f}, "
                  f"q_c_loss={q_c_loss.item():.4f}")

    torch.save({
        'actor': actor.state_dict(),
        'q_r': q_r_nets[0].state_dict(),
        'q_c': q_c_nets[0].state_dict(),
    }, save_path)
    print(f"Saved {save_path}")
    return actor


def load_safe_cql_policy(path="safe_cql_model.pt"):
    """Load actor and return policy function for rollout."""
    ckpt = torch.load(path, map_location='cpu')
    actor = PolicyNet(STATE_DIM, N_ACTIONS)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()

    def policy(s):
        s_np = np.array(s, dtype=np.float32)
        if s_np.ndim == 1:
            s_np = s_np.reshape(1, -1)
        with torch.no_grad():
            probs = actor(torch.FloatTensor(s_np))
            a_idx = torch.argmax(probs, dim=-1).item()
        return float([0.0, 0.5, 1.0, 2.0][a_idx])

    return policy


if __name__ == "__main__":
    need_data = not os.path.exists("offline_dataset.npz")
    if not need_data:
        with np.load("offline_dataset.npz") as z:
            need_data = "c" not in z.files
    if need_data:
        print("Regenerating dataset with costs...")
        from data.generate import generate_dataset, save_dataset
        data = generate_dataset(n_trajectories=1000, use_reward_v3=True, state_noise_sigma=0.02, expert_balance_ratio=0.6)
        save_dataset(data)
    train_safe_cql()
