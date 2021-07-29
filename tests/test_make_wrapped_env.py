import itertools
from typing import Optional

import numpy as np
import pytest

from masters.env_wrapper import make_wrapped_env


@pytest.fixture
def env():
    return make_wrapped_env(env_name="CartPole-v0", random_seed=0)


@pytest.mark.parametrize(
    "env_name,random_seed_1,random_seed_2",
    itertools.product(["CartPole-v0"], [None, 0, 1], [None, 0, 1]),
)
def test_random_seed(env_name: str, random_seed_1: Optional[int], random_seed_2: Optional[int]):
    env_1 = make_wrapped_env(env_name=env_name, random_seed=random_seed_1)
    env_2 = make_wrapped_env(env_name=env_name, random_seed=random_seed_2)

    obs_1: np.ndarray = env_1.reset()
    obs_2: np.ndarray = env_2.reset()

    if random_seed_1 is not None and random_seed_2 is not None:
        if random_seed_1 == random_seed_2:
            # equal seeds
            assert (obs_1 == obs_2).all()
        else:
            # different seeds
            assert (obs_1 != obs_2).all()
    else:
        # one of the seeds is None
        assert (obs_1 != obs_2).all()
