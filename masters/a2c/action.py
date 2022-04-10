import torch


def select_softmax(logits: torch.FloatTensor):
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.

    :return: Action sampled from logits softmax
    """

    probabilities = torch.softmax(logits, dim=0)

    return torch.multinomial(probabilities, num_samples=1).item()
