from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection

class TwoLayerNetworkShaped(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a
    fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
        inpt_shape: Optional[Iterable[int]] = None, #addition
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.
        :param n_inpt: Number of input neurons. Matches the 1D size of the input dvs_data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization
            constant.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt
        self.inpt_shape = inpt_shape

        self.add_layer(Input(n=self.n_inpt, shape=self.inpt_shape,traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNodes(
                n=self.n_neurons,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                w=w,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )