import torch


def select_softmax(spikes: torch.FloatTensor):
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.

    :return: Action sampled from softmax over activity of similarly-sized output layer.
    """
    spikes = spikes.sum(dim=0).flatten()  # type: ignore

    probabilities = torch.softmax(spikes, dim=0)

    return torch.multinomial(probabilities, num_samples=1).item()
