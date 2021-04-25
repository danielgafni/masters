from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from masters.constants import CONFIGS_DIR

MODEL_CONFIG_DIR = CONFIGS_DIR / "model"


@dataclass
class ModelConfig:
    pass


@dataclass
class InhibitoryConfig(ModelConfig):
    hidden_neurons: int = 100


def register_model_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", name="base_inhibitory", node=InhibitoryConfig)
