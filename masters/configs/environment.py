from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from masters.constants import CONFIGS_DIR

ENVIRONMENT_CONFIG_DIR = CONFIGS_DIR / "environment"


@dataclass
class EnvironmentConfig:
    in_neurons: int = MISSING
    in_shape: List[int] = MISSING
    out_neurons: int = MISSING


@dataclass
class CartPoleConfig(EnvironmentConfig):
    in_neurons: int = 8
    in_shape: List[int] = field(default_factory=lambda: [2, 4])
    out_neurons: int = 2


def register_environment_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="environment", name="base_cartpole", node=CartPoleConfig)
