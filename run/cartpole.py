import logging
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
from bindsnet.encoding.encoders import GaussianReceptiveFieldsEncoder
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from masters.a2c.agent import A2CAgent, A2CAgentConfig
from masters.a2c.trainer import A2CTrainer, A2CTrainerConfig
from masters.networks.actor import ActorConfig
from masters.networks.critic import CriticConfig

cartpole_info = {
    0: dict(start=-4.8, end=4.8, scale=0.5, n=20),
    1: dict(start=-10, end=10, scale=0.1, n=100),
    2: dict(start=-0.418, end=0.418, scale=0.02, n=40),
    3: dict(start=-10, end=10, scale=0.2, n=100),
}


@dataclass
class CartPoleActorConfig(ActorConfig):
    input_shape: List[int] = field(default_factory=lambda: [260])
    # n_hidden: int = 100
    n_out: int = 200
    action_space_size: int = 2
    a_minus: float = -1e-2
    a_plus: float = 1e-2
    thresh: float = -57.0
    weight_decay: Optional[float] = 1e-2
    time: int = 100
    dev: bool = True

    @staticmethod
    def register_config():
        cs = ConfigStore.instance()
        cs.store(node=CartPoleActorConfig, name="actor")


@dataclass
class CartPoleCriticConfig(CriticConfig):
    input_shape: List[int] = field(default_factory=lambda: [260])
    # n_hidden: int = 100
    n_out: int = 100
    a_minus: float = -1e2
    a_plus: float = 1e-2
    thresh: float = -57.0
    weight_decay: Optional[float] = 1e-2
    time: int = 100
    dev: bool = True

    @staticmethod
    def register_config():
        cs = ConfigStore.instance()
        cs.store(node=CartPoleCriticConfig, name="critic")


@dataclass
class CartPoleA2CAgentConfig(A2CAgentConfig):
    actor: CartPoleActorConfig = CartPoleActorConfig()
    critic: CartPoleCriticConfig = CartPoleCriticConfig()

    @staticmethod
    def register_config():
        cs = ConfigStore.instance()
        CartPoleActorConfig.register_config()
        CartPoleCriticConfig.register_config()
        cs.store(node=CartPoleA2CAgentConfig, name="agent")


@dataclass
class CartPoleA2CTrainerConfig(A2CTrainerConfig):
    num_episodes: int = 1000
    max_steps: int = 1000
    gamma: float = 0.99
    spikes_to_value: float = 0.15
    value_offset: float = -3.0
    start_actor_train: int = 200
    num_test_episodes: int = 200
    experiment_name: str = "a2c/CartPole"

    @staticmethod
    def register_config():
        cs = ConfigStore.instance()
        cs.store(node=CartPoleA2CTrainerConfig, name="trainer")


@dataclass
class RunConfig:
    agent: CartPoleA2CAgentConfig = CartPoleA2CAgentConfig()
    trainer: CartPoleA2CTrainerConfig = CartPoleA2CTrainerConfig()
    intensity: float = 200
    logging_level: str = "INFO"

    @staticmethod
    def register_config():
        cs = ConfigStore.instance()
        cs.store(node=RunConfig, name="run")


def run(cfg: RunConfig):
    encoder = GaussianReceptiveFieldsEncoder(encoding_info=cartpole_info, intensity=cfg.intensity)
    agent: A2CAgent = instantiate(cfg.agent, encoder=encoder)
    trainer: A2CTrainer = instantiate(cfg.trainer)

    agent.to(trainer.device)

    trainer.fit(agent, "CartPole-v0")
    trainer.test(agent, "CartPole-v0")


@hydra.main(config_path=None, config_name="run")
def main(cfg: RunConfig):
    logging.basicConfig(level=cfg.logging_level)
    run(cfg=cfg)


if __name__ == "__main__":
    RunConfig.register_config()
    main()
