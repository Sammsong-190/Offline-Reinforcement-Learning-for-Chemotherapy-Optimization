"""
Self-implemented Discrete CQL - no d3rlpy/gym dependency
Paper: Kumar et al. Conservative Q-Learning for Offline Reinforcement Learning
"""
import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """Double DQN backbone"""

    def __init__(self, state_dim=4, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DiscreteCQL:
    def __init__(
        self,
        state_dim=4,
        action_dim=4,
        lr=3e-4,
        gamma=0.99,
        alpha=1.0,
        target_update=200,
        hidden=128,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.action_dim = action_dim
        self.update_count = 0
        self.target_update = target_update

        self.q_net = QNetwork(state_dim, action_dim, hidden).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def update(self, batch):
        s, a, r, s_next, done = [torch.FloatTensor(x).to(self.device) for x in batch]
        a = a.long()

        # TD target (Double DQN)
        with torch.no_grad():
            online_next = self.q_net(s_next)
            best_action = online_next.argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(s_next).gather(1, best_action).squeeze(1)
            td_target = r + self.gamma * (1 - done.float()) * target_next_q

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        td_loss = nn.functional.mse_loss(current_q, td_target)

        # CQL conservative term
        q_all = self.q_net(s)
        logsumexp = torch.logsumexp(q_all, dim=1)
        q_data = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        cql_loss = (logsumexp - q_data).mean()

        loss = td_loss + self.alpha * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Target network soft update
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.copy_(0.005 * op.data + 0.995 * tp.data)

        return {"td_loss": td_loss.item(), "cql_loss": cql_loss.item(), "total": loss.item()}

    def predict(self, state):
        state = np.array(state, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        with torch.no_grad():
            s = torch.FloatTensor(state).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(), "target_net": self.target_net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt.get("target_net", ckpt["q_net"]))
