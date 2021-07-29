import gym
import pytest
import torch
from bindsnet.encoding import Encoder
from bindsnet.encoding.encoders import PoissonEncoder
from bindsnet.network import Network

from masters.a2c.agent import A2CAgent
from masters.networks.mlp import make_mlp
from tests.test_make_wrapped_env import env  # noqa


@pytest.fixture
def actor():
    return make_mlp(input_shape=[4], output_shape=[2])


@pytest.fixture
def critic():
    return make_mlp(input_shape=[4], output_shape=[1])


@pytest.fixture
def prev_critic():
    return make_mlp(input_shape=[4], output_shape=[10, 1])


@pytest.fixture
def encoder():
    return PoissonEncoder(time=100)


@pytest.fixture
def agent(actor: Network, critic: Network, prev_critic: Network, encoder: Encoder):
    return A2CAgent(actor=actor, critic=critic, prev_critic=prev_critic, encoder=encoder)


def test_agent(agent: A2CAgent):
    assert agent is not None


def test_get_action(agent: A2CAgent, env: gym.Env):  # noqa
    observation = env.reset()
    observation = torch.from_numpy(observation).float()

    action = agent.run_actor(observation=observation)

    assert action is not None
