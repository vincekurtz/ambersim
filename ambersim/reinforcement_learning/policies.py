from typing import Sequence

import flax.linen as nn
import jax.numpy as jp
from brax.training import distribution, networks, types

"""Tools for defining neural network policies with various architectures."""


class MLP(nn.Module):
    """Your classic multilayer perceptron with a variable number of hidden layers.

    Args:
        layer_sizes: Sizes of all hidden layers, followed by the output layer.
        activation: Activation function to use.
        kernel_init: Initialization function for the weights.
        activate_final: Whether to apply the activation function to the final
            layer.
        bias: Whether to include a bias vector in each layer.
    """

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = nn.relu
    kernel_init: networks.Initializer = nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_size,
                kernel_init=self.kernel_init,
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x
