import pytest

from masters.networks.mlp import MLP


@pytest.fixture
def agent():
    pass


@pytest.fixture
def critic():
    pass


@pytest.fixture
def prev_critic():
    pass


def test_make_mlp():
    mlp = MLP(input_shape=[4], n_hidden=0, n_out=2)

    assert mlp is not None
