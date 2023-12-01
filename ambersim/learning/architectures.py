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

    def setup(self):
        """Initialize the network."""
        self.modules = [self.module_type(**self.module_kwargs) for _ in range(self.num_modules)]

    def __call__(self, x: jnp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # First module takes only the global input, x --> y
        y = self.modules[0](x)

        # Subsequent modules also take the previous output, [x, y] --> y
        # TODO(vincekurtz): use jax control flow
        for i in range(1, self.num_modules):
            y = self.modules[i](jnp.concatenate([x, y], axis=-1))
        return y


class NestedLinearPolicy(nn.Module):
    """A hierarchy of linear feedback controllers, each with different (learnable) inputs.

    This is similar to many practical control architectures, where each layer has
    access to different measurments (e.g. functions of raw observations) and passes
    its output to the next layer.

                    -------      -------
               ---> | M_1 | ---> | K_1 |
               |    -------      -------
               |                    |
               |    -------      -------
         x --> |--> | M_2 | ---> | K_2 |
               |    -------      -------
               |                    |
               |    -------      -------
               ---> | M_3 | ---> | K_3 | ---> y
                    -------      -------

    Here the measurment functions M_i are arbitrary neural networks, and the
    controllers K_i are linear layers.


    """

    measurement_networks: Sequence[nn.Module]
    measurement_network_kwargs: Sequence[dict]
    linear_policy_kwargs: Sequence[dict]

    def setup(self):
        """Initialize the network."""
        self.nets = [
            measurement_networks(**kwargs)
            for measurement_networks, kwargs in zip(self.measurement_networks, self.measurement_network_kwargs)
        ]
        self.linear_policies = [nn.Dense(**kwargs) for kwargs in self.linear_policy_kwargs]

    def __call__(self, x: jnp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # Compute the measurements
        # TODO(vincekurtz): this can be parallelized
        measurements = [net(x) for net in self.nets]

        # Compute the linear feedback outputs. The first controller takes only
        # it's own measurement, while subsequent controllers take the previous
        # output as well.
        y = self.linear_policies[0](measurements[0])
        for i in range(1, len(self.linear_policies)):
            y = self.linear_policies[i](jnp.concatenate([measurements[i], y], axis=-1))

        return y
