"""
Improved model architectures: ResidualBlock, ImprovedPolicyNet, ImprovedQNet
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ImprovedPolicyNet(nn.Module):
    """
    Improved policy: deeper, LayerNorm, GELU, Dropout.
    """

    def __init__(self, state_dim=4, action_dim=4, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        return self.head(x)

    def get_action(self, state, deterministic=True):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits = self.forward(s)
            if deterministic:
                return logits.argmax(dim=1).item()
            probs = torch.softmax(logits, dim=1)
            return torch.multinomial(probs, 1).item()


class ImprovedQNet(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)"""

    def __init__(self, state_dim=4, action_dim=4, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x):
        feat = self.encoder(x)
        V = self.value_stream(feat)
        A = self.adv_stream(feat)
        return V + A - A.mean(dim=1, keepdim=True)
