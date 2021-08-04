from dataclasses import dataclass

import numpy as np

from masters.networks.mlp import MLP, MLPConfig


@dataclass
class ActorConfig(MLPConfig):
    _target_: str = "masters.networks.actor.Actor"


class Actor(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space_size = self.output_shape[-1]
        self.population_size = int(np.prod(self.output_shape[:-1]))
