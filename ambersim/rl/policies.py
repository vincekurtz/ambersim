from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax.numpy as jp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

"""Tools for defining neural network policies with various architectures."""


@dataclass
class PPONetworkConfig:
    """Pickleable configuration for a simple PPO network.

    Args:
        policy_layer_sizes: Sizes of hidden layers in the policy, excluding the
            output layer.
        value_layer_sizes: Sizes of hidden layers in the value function, excluding
            the output layer.
    """

    policy_hidden_layer_sizes: Sequence[int] = (64, 64)
    value_hidden_layer_sizes: Sequence[int] = (64, 64)


def make_ppo_networks_from_config(
    observation_size: int,
    action_size: int,
    config: PPONetworkConfig,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
) -> PPONetworks:
    """Creates policy and value networks for brax from a saveable config.

    Args:
        observation_size: Size of the input (observation).
        action_size: Size of the policy output (action).
        config: Configuration for the network.

    Returns:
        A tuple of (policy, value) networks.
    """
    # Create an action distribution given the action size
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    # Create the policy network, a FeedForwardNetwork that contains an "init"
    # and an "apply" function.
    policy_output_size = parametric_action_distribution.param_size
    policy_layer_sizes = config.policy_hidden_layer_sizes + (policy_output_size,)
    policy_mlp = MLP(layer_sizes=policy_layer_sizes)

    dummy_obs = jp.zeros((1, observation_size))

    def policy_init(key):
        """Initialize the policy network from a random key."""
        return policy_mlp.init(key, dummy_obs)

    def policy_apply(processor_params, policy_params, obs):
        """Apply the policy given the parameters and an observation."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_mlp.apply(policy_params, obs)

    policy_network = networks.FeedForwardNetwork(init=policy_init, apply=policy_apply)

    # Create the value network. This is just like the policy network, but with
    # a 1D output.
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=config.value_hidden_layer_sizes,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


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
        # TODO(vincekurtz): use jax control flow
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


class SequentialComposition(nn.Module):
    """A series of modules applied one after the other, i.e.

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
    def __call__(self, x: jp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # TODO(vincekurtz): use jax control flow
        for _ in range(self.num_modules):
            x = self.module_type(**self.module_kwargs)(x)
        return x


class ParallelComposition(nn.Module):
    """A series of modules applied in parallel and then summed, i.e.

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
    def __call__(self, x: jp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # TODO(vincekurtz): use jax control flow
        outputs = []
        for _ in range(self.num_modules):
            y = self.module_type(**self.module_kwargs)(x)
            outputs.append(y)
        y = jp.sum(jp.stack(outputs), axis=0)
        return y


class HierarchyComposition(nn.Module):
    """A series of modules evaluated in a hierarchy.

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
    def __call__(self, x: jp.ndarray):
        """Forward pass through the network.

        Args:
            x: Input to the network.
        """
        # First module takes only the global input, x --> y
        y = self.module_type(**self.module_kwargs)(x)

        # Subsequent modules also take the previous output, [x, y] --> y
        # TODO(vincekurtz): use jax control flow
        for _ in range(self.num_modules - 1):
            y = self.module_type(**self.module_kwargs)(jp.concatenate([x, y], axis=-1))
        return y
