import numpy as np
import pytest
import torch
from bindsnet.encoding.encoders import PoissonEncoder, PositiveEncoder


@pytest.fixture
def observation():
    return np.random.rand(10, 4, 1).astype("float32")


@pytest.fixture
def positive_poisson_encoder():
    encoder = PoissonEncoder(time=100, rate=1)
    positive_encoder = PositiveEncoder(encoder=encoder)

    return positive_encoder


def test_positive_encoder(observation: np.ndarray, positive_poisson_encoder: PositiveEncoder):
    encoded_observation = positive_poisson_encoder(torch.from_numpy(observation))

    assert encoded_observation is not None
