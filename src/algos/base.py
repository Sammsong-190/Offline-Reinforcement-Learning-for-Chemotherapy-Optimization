"""Base class for offline RL algorithms"""
from abc import ABC, abstractmethod


class BaseAlgo(ABC):
    """Abstract base for offline RL algorithms."""

    @abstractmethod
    def train(self, data_path: str, **kwargs):
        pass

    @abstractmethod
    def get_policy(self):
        """Return policy function s -> a for rollout."""
        pass
