from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING

from masters.constants import CONFIGS_DIR

from .environment import ENVIRONMENT_CONFIG_DIR, CartPoleConfig
from .model import MODEL_CONFIG_DIR, InhibitoryConfig

CONFIG_DIR = str(CONFIGS_DIR)


defaults = [
    {"environment": "cartpole"},
    {"model": "inhibitory"}
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    train_episodes: int = 100
    test_episodes: int = 100

    environment: Any = MISSING
    model: Any = MISSING


__all__ = [
    "Config", "CONFIG_DIR",  "InhibitoryConfig", "MODEL_CONFIG_DIR",  "CartPoleConfig", "ENVIRONMENT_CONFIG_DIR"
]
