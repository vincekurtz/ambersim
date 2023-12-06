from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax


def print_module_summary(module: nn.Module, input_shape: Sequence[int]):
    """Print a readable summary of a flax neural network module.

    Args:
        module: The flax module to summarize.
        input_shape: The shape of the input to the module.
    """
    # Create a dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(input_shape)
    print(module.tabulate(rng, dummy_input, depth=1))


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

    Args:
        measurement_networks: Type of measurement network to create
        measurement_network_kwargs: Keyword arguments to pass to each measurement network.
        linear_policy_kwargs: Keyword arguments to pass to each linear policy.

    Note: the length of measurement_networks, measurement_network_kwargs, and
    linear_policy_kwargs must be the same. The output size is specified by the
    'features' keyword argument of the last linear policy.
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


class LinearSystemPolicy(nn.Module):
    """A feedback controller that is itself a linear dynamical system.

        z_{t+1} = A z_t + B y_t
        u_t = C z_t + D y_t

    where y_t is the observation, u_t is the action, and z_t is the controller state.
    We assume that the z is stored in the env, so this module takes as input [z_t, y_t]
    and sends as output [z_{t+1}, u_t].

    It also outputs log standard deviations for u_t and z_{t+1}, which are used in PPO.

    Args:
        nz: Dimension of the controller state.
        ny: Dimension of the observation.
        nu: Dimension of the action.
    """

    nz: int
    ny: int
    nu: int

    def setup(self):
        """Initialize the network."""
        self.A = self.param("A", nn.initializers.lecun_normal(), (self.nz, self.nz))
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.nz, self.ny))
        self.C = self.param("C", nn.initializers.lecun_normal(), (self.nu, self.nz))
        self.D = self.param("D", nn.initializers.lecun_normal(), (self.nu, self.ny))

        self.log_std_z = self.param("log_std_z", nn.initializers.zeros, (self.nz,))
        self.log_std_u = self.param("log_std_u", nn.initializers.zeros, (self.nu,))

    def __call__(self, zy: jnp.ndarray):
        """Forward pass through the network."""
        # Select z and y from the input (last dimension)
        z = zy[..., : self.nz]
        y = zy[..., self.nz :]

        # Linear map: note that the last dim holds our data so we transpose
        z_next = jnp.matmul(z, self.A.T) + jnp.matmul(y, self.B.T)
        u = jnp.matmul(z, self.C.T) + jnp.matmul(y, self.D.T)

        # Tile log_std to match the dimensions of the input (zy)
        log_std_z = jnp.tile(self.log_std_z, zy.shape[:-1] + (1,))
        log_std_u = jnp.tile(self.log_std_u, zy.shape[:-1] + (1,))

        return jnp.concatenate([z_next, u, log_std_z, log_std_u], axis=-1)


class BilinearSystemPolicy(nn.Module):
    """A feedback controller that is itself a bilinear dynamical system.

        z_{t+1} = A z_t + ∑ᵢ B[i] z_t y_t[i]
        u_t = C z_t + D y_t

    where y_t is the observation, u_t is the action, and z_t is the controller state.
    We assume that the z is stored in the env, so this module takes as input [z_t, y_t]
    and sends as output [z_{t+1}, u_t].

    It also outputs log standard deviations for u_t and z_{t+1}, which are used in PPO.

    Args:
        nz: Dimension of the controller state.
        ny: Dimension of the observation.
        nu: Dimension of the action.
    """

    nz: int
    ny: int
    nu: int

    def setup(self):
        """Initialize the network."""
        self.A = self.param("A", nn.initializers.lecun_normal(), (self.nz, self.nz))
        self.B = self.param("B", nn.initializers.lecun_normal(), (self.nz, self.nz, self.ny))
        self.C = self.param("C", nn.initializers.lecun_normal(), (self.nu, self.nz))
        self.D = self.param("D", nn.initializers.lecun_normal(), (self.nu, self.ny))

        self.log_std_z = self.param("log_std_z", nn.initializers.zeros, (self.nz,))
        self.log_std_u = self.param("log_std_u", nn.initializers.zeros, (self.nu,))

    def __call__(self, zy: jnp.ndarray):
        """Forward pass through the network."""
        # Select z and y from the input (last dimension)
        z = zy[..., : self.nz]
        y = zy[..., self.nz :]

        # System dynamics
        Bzy = jnp.einsum("ijk,...j,...k->...i", self.B, z, y)
        z_next = jnp.matmul(z, self.A.T) + Bzy
        u = jnp.matmul(z, self.C.T) + jnp.matmul(y, self.D.T)

        # Tile log_std to match the dimensions of the input (zy)
        log_std_z = jnp.tile(self.log_std_z, zy.shape[:-1] + (1,))
        log_std_u = jnp.tile(self.log_std_u, zy.shape[:-1] + (1,))

        return jnp.concatenate([z_next, u, log_std_z, log_std_u], axis=-1)


class LiftedInputLinearSystemPolicy(nn.Module):
    """A feedback controller that is itself a linear dynamical system with liffed input.

        z_{t+1} = A z_t + phi(y_t)
        u_t = C z_t + D y_t

    where y_t is the observation, u_t is the action, and z_t is the controller state.
    The input is lifted by the function phi, which is a neural network. We assume that
    the z is stored in the env, so this module takes as input [z_t, y_t] and sends as
    output [z_{t+1}, u_t].

    It also outputs log standard deviations for u_t and z_{t+1}, which are used in PPO.

    Args:
        nz: Dimension of the controller state.
        ny: Dimension of the observation.
        nu: Dimension of the action.
        phi_kwargs: Keyword arguments for constructing the lifting function (an MLP)
    """

    nz: int
    ny: int
    nu: int
    phi_kwargs: dict

    def setup(self):
        """Initialize the network."""
        # Linear map
        self.A = self.param("A", nn.initializers.lecun_normal(), (self.nz, self.nz))
        self.C = self.param("C", nn.initializers.lecun_normal(), (self.nu, self.nz))
        self.D = self.param("D", nn.initializers.lecun_normal(), (self.nu, self.ny))

        # Lifting function
        self.phi = MLP(**self.phi_kwargs)

        # Log standard deviations
        self.log_std_z = self.param("log_std_z", nn.initializers.zeros, (self.nz,))
        self.log_std_u = self.param("log_std_u", nn.initializers.zeros, (self.nu,))

    def __call__(self, zy: jnp.ndarray):
        """Forward pass through the network."""
        # Select z and y from the input (last dimension)
        z = zy[..., : self.nz]
        y = zy[..., self.nz :]

        # Controller dynamics
        z_next = jnp.matmul(z, self.A.T) + self.phi(y)
        u = jnp.matmul(z, self.C.T) + jnp.matmul(y, self.D.T)

        # Tile log_std to match the dimensions of the input (zy)
        log_std_z = jnp.tile(self.log_std_z, zy.shape[:-1] + (1,))
        log_std_u = jnp.tile(self.log_std_u, zy.shape[:-1] + (1,))

        return jnp.concatenate([z_next, u, log_std_z, log_std_u], axis=-1)
