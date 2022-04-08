from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from bindsnet.learning import MSTDPET, PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from masters.networks import HIDDEN_LAYER_NAME, INPUT_LAYER_NAME, OUTPUT_LAYER_NAME


@dataclass
class MLPConfig:
    input_shape: List[int]
    n_hidden: int
    n_out: int
    norm_hidden_out: Optional[float] = 0.5
    norm_recurrent: Optional[float] = 5e-3
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
        n_hidden: int,
        n_out: int,
        norm_hidden_out: Optional[float] = 0.5,
        norm_recurrent: Optional[float] = 5e-3,
        a_plus: float = 1e-2,
        a_minus: float = -1e-3,
        thresh: float = -52.0,
        time: int = 100,
        dev: bool = False,
    ):
        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.norm_hidden_out = norm_hidden_out
        self.norm_recurrent = norm_recurrent
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.thresh = thresh
        self.time = time
        self.dev = dev

        self.input_size = int(np.prod(self.input_shape))

        nu = (-a_minus, a_plus)

        # Build network.
        self.network = Network(dt=1.0)

        # Layers of neurons.
        inpt = Input(n=np.prod(input_shape), shape=input_shape, traces=True)
        hidden = LIFNodes(n=n_hidden, traces=True, thresh=thresh)
        out = LIFNodes(n=n_out, traces=True, thresh=thresh, refrac=0)

        # Connections
        inpt_middle = Connection(source=inpt, target=hidden, wmin=0, wmax=1)

        # Connections from hidden layer to output layer
        middle_out = Connection(
            source=hidden,
            target=out,
            wmin=0,  # minimum weight value
            wmax=1,  # maximum weight value
            update_rule=MSTDPET,  # learning rule
            nu=nu,  # learning rate
            norm=norm_hidden_out * hidden.n if norm_hidden_out is not None else None,  # normalization
        )

        # Recurrent connection, retaining data within the hidden layer
        recurrent = Connection(
            source=hidden,
            target=hidden,
            wmin=0,  # minimum weight value
            wmax=1,  # maximum weight value
            update_rule=PostPre,  # learning rule
            nu=nu,  # learning rate
            norm=norm_recurrent * hidden.n if norm_recurrent is not None else None,  # normalization
        )

        # Add all layers and connections to the network.
        self.network.add_layer(inpt, name=INPUT_LAYER_NAME)
        self.network.add_layer(hidden, name=HIDDEN_LAYER_NAME)
        self.network.add_layer(out, name=OUTPUT_LAYER_NAME)
        self.network.add_connection(inpt_middle, source=INPUT_LAYER_NAME, target=HIDDEN_LAYER_NAME)
        self.network.add_connection(middle_out, source=HIDDEN_LAYER_NAME, target=OUTPUT_LAYER_NAME)
        self.network.add_connection(recurrent, source=HIDDEN_LAYER_NAME, target=HIDDEN_LAYER_NAME)

        if dev:
            for name, layer in self.network.layers.items():
                state_vars = ("s",) if type(layer) == Input else ("s", "v")
                monitor = Monitor(layer, state_vars=state_vars, time=time)
                self.network.add_monitor(monitor, name)

        else:
            output_monitor = Monitor(self.network.layers[OUTPUT_LAYER_NAME], state_vars=("s",), time=time)
            self.network.add_monitor(output_monitor, OUTPUT_LAYER_NAME)
