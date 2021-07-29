from typing import List, Optional, Sequence, Union

import numpy as np
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from masters.networks import INPUT_LAYER_NAME, OUTPUT_LAYER_NAME


def make_mlp(
    input_shape: List[int],
    output_shape: List[int],
    norm: float = 0.5,
    nu: Optional[Union[float, Sequence[float]]] = (1e-2, 1e-3),
    time: int = 100,
    dev: bool = False,
):
    # Build network.
    network = Network(dt=1.0)

    layers = []
    # Layers of neurons.

    inpt = Input(n=np.prod(input_shape), shape=input_shape, traces=True)
    layers.append(inpt)
    network.add_layer(inpt, INPUT_LAYER_NAME)

    out = LIFNodes(n=np.prod(output_shape), shape=output_shape, traces=True)
    layers.append(out)
    network.add_layer(out, OUTPUT_LAYER_NAME)

    network.add_connection(
        Connection(
            source=inpt,
            target=layers[1],
            wmin=0,
            wmax=1,
            update_rule=MSTDPET,
            nu=nu,
            norm=norm * inpt.n,
        ),
        source=INPUT_LAYER_NAME,
        target=OUTPUT_LAYER_NAME,
    )

    if dev:
        for name, layer in network.layers.items():
            state_vars = ("s",) if type(layer) == Input else ("s", "v")
            monitor = Monitor(layer, state_vars=state_vars, time=time)
            network.add_monitor(monitor, name)

    else:
        output_monitor = Monitor(network.layers[OUTPUT_LAYER_NAME], state_vars=("s",), time=time)
        network.add_monitor(output_monitor, OUTPUT_LAYER_NAME)

    return network
