from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """A simple pickle-able multi-layer perceptron.

    Args:
        layer_sizes: Sizes of all hidden layers and the output layer.
        activate_final: Whether to apply an activation function to the output.
        bias: Whether to use a bias in the linear layers.
    """

    layer_sizes: Sequence[int]
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # TODO(vincekurtz): consider using jax control flow here. Note that
        # standard jax control flows (e.g. jax.lax.scan) do not play nicely with
        # flax, see for example https://github.com/google/flax/discussions/1283.
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                layer_size,
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = nn.relu(x)
        return x


class SeriesComposition(nn.Module):
    """A set of modules applied one after the other, i.e.

              -------     -------             -------
        x --> | m_1 | --> | m_2 | --> ... --> | m_N | --> y
              -------     -------             -------

    Args:
        module_type: Type of module to create
        num_modules: Number modules in the chain
        module_kwargs: Keyword arguments to pass to each module.
    """

    module_type: nn.Module
    num_modules: int
    module_kwargs: dict

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # TODO(vincekurtz): use jax control flow
        for _ in range(self.num_modules):
            x = self.module_type(**self.module_kwargs)(x)
        return x


class ParallelComposition(nn.Module):
    """A set of modules applied in parallel and then summed, i.e.

                     -------
             ------> | m_1 | ------
             |       -------      |
             |                    |
             |       -------      |
        x -->|-----> | m_2 | ----(+)---> y
             |       -------      |
             |                    |
             |       -------      |
             ------> | m_3 | ------
                     -------

    Args:
        module_type: Type of module to create
        num_modules: Number modules in the chain
        module_kwargs: Keyword arguments to pass to each module.
    """

    module_type: nn.Module
    num_modules: int
    module_kwargs: dict

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # TODO(vincekurtz): use jax control flow
        outputs = []
        for _ in range(self.num_modules):
            y = self.module_type(**self.module_kwargs)(x)
            outputs.append(y)
        y = jnp.sum(jnp.stack(outputs), axis=0)
        return y


class HierarchyComposition(nn.Module):
    """A set of modules evaluated in a hierarchy.

    Each module takes both the global input and the output of the previous
    module as input, similar to how many control architectures are structured:

                    -------
               ---> | m_1 |
               |    -------
               |       |
               |    -------
         x --> |--> | m_2 |
               |    -------
               |       |
               |    -------
               ---> | m_3 | ---> y
                    -------

    Args:
        module_type: Type of module to create
        num_modules: Number modules in the chain
        module_kwargs: Keyword arguments to pass to each module.
    """

    module_type: nn.Module
    num_modules: int
    module_kwargs: dict

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # First module takes only the global input, x --> y
        y = self.module_type(**self.module_kwargs)(x)

        # Subsequent modules also take the previous output, [x, y] --> y
        # TODO(vincekurtz): use jax control flow
        for _ in range(self.num_modules - 1):
            y = self.module_type(**self.module_kwargs)(jnp.concatenate([x, y], axis=-1))
        return y
