from dataclasses import dataclass

from masters.networks.mlp import MLP, MLPConfig


@dataclass
class CriticConfig(MLPConfig):
    _target_: str = "masters.networks.critic.Critic"


class Critic(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
