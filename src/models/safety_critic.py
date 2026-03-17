"""成本网络 (Cost Critic): 预测未来累积违规成本"""
import torch.nn as nn
from .critic import Critic


class SafetyCritic(Critic):
    """
    Cost Q-function: Q_C(s,a) = E[sum gamma^t c_t].
    Same architecture as Critic, different training objective.
    """

    def __init__(self, state_dim=4, n_actions=4, hidden=64):
        super().__init__(state_dim, n_actions, hidden)
