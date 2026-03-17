"""Q 网络 (Double Q for offline RL)"""
import torch.nn as nn


class Critic(nn.Module):
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
