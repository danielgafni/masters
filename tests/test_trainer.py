import pytest

from masters.a2c.agent import A2CAgent
from masters.a2c.trainer import A2CTrainer


@pytest.fixture
def trainer():
    return A2CTrainer(max_steps=5, num_episodes=1)


@pytest.mark.parametrize("env_name,", ["CartPole-v0"])
def test_play(trainer: A2CTrainer, agent: A2CAgent, env_name: str):
    episode = trainer.play_episode(agent=agent, env_name=env_name)

    assert episode is not None
    assert len(episode.transitions) > 0
