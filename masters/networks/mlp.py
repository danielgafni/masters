from dataclasses import dataclass
from typing import List

import numpy as np
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from masters.networks import INPUT_LAYER_NAME, OUTPUT_LAYER_NAME


@dataclass
class MLPConfig:
    input_shape: List[int]
    output_shape: List[int]
    norm: float = 0.5
    a_plus: float = 1e-2
    a_minus: float = -1e-3
    thresh: float = -52.0
    time: int = 100
    dev: bool = False

    _target_: str = "masters.networks.mlp.MLP"


class MLP:
    def __init__(
        self,
        input_shape: List[int],
        output_shape: List[int],
        norm: float = 0.5,
        a_plus: float = 1e-2,
        a_minus: float = -1e-3,
        thresh: float = -52.0,
        time: int = 100,
        dev: bool = False,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.norm = norm
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.thresh = thresh
        self.time = time
        self.dev = dev

        self.input_size = int(np.prod(self.input_shape))

        nu = (-a_minus, a_plus)

        # Build network.
        self.network = Network(dt=1.0)

        layers = []
        # Layers of neurons.

        inpt = Input(n=np.prod(input_shape), shape=input_shape, traces=True)
        layers.append(inpt)
        self.network.add_layer(inpt, INPUT_LAYER_NAME)

        out = LIFNodes(n=np.prod(output_shape), shape=output_shape, traces=True, thresh=thresh)
        layers.append(out)
        self.network.add_layer(out, OUTPUT_LAYER_NAME)

        norm_ = norm * inpt.n if norm is not None else None

        self.network.add_connection(
            Connection(
                source=inpt,
                target=layers[1],
                wmin=0,
                wmax=1,
                update_rule=MSTDPET,
                nu=nu,
                norm=norm_,
            ),
            source=INPUT_LAYER_NAME,
            target=OUTPUT_LAYER_NAME,
        )

        if dev:
            for name, layer in self.network.layers.items():
                state_vars = ("s",) if type(layer) == Input else ("s", "v")
                monitor = Monitor(layer, state_vars=state_vars, time=time)
                self.network.add_monitor(monitor, name)

        else:
            output_monitor = Monitor(self.network.layers[OUTPUT_LAYER_NAME], state_vars=("s",), time=time)
            self.network.add_monitor(output_monitor, OUTPUT_LAYER_NAME)
