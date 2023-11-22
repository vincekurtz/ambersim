from dataclasses import dataclass
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks

"""Tools for defining neural network policies with various architectures."""


@dataclass
class BraxPPONetworkWrapper:
    """A lightweight wrapper around a brax PPONetwork that allows it to be pickled."""

    policy_network: nn.Module
    value_network: nn.Module
    action_distribution: distribution.ParametricDistribution

    def network_factory(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    ) -> PPONetworks:
        """Create a PPONetwork, compatible with brax's ppo.train() function.

        Args:
            observation_size: Size of the input (observation).
            action_size: Size of the policy output (action).
            preprocess_observations_fn: Function to preprocess (e.g. normalize) observations.

        Returns:
            A PPONetworks object.
        """
        # Create an action distribution. The policy network should output the
        # parameters of this distribution.
        action_dist = self.action_distribution(event_size=action_size)

        # Set up a dummy observation and random key for size verifications
        dummy_observation = jp.zeros((1, observation_size))
        rng = jax.random.PRNGKey(0)

        # Check that the output size of the policy network matches the size of
        # the action distribution.
        dummy_params = self.policy_network.init(rng, dummy_observation)
        policy_output = self.policy_network.apply(dummy_params, dummy_observation)
        assert (
            policy_output.shape[-1] == action_dist.param_size
        ), f"policy network output size {policy_output.shape[-1]} does not match action distribution size {action_dist.param_size}"

        # Create the policy network, a FeedForwardNetwork that contains an "init"
        # and an "apply" function.
        def policy_init(key):
            """Initialize the policy network from a random key."""
            return self.policy_network.init(key, dummy_observation)

        def policy_apply(processor_params, policy_params, obs):
            """Apply the policy given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, obs)

        # Create the value network. This is just like the policy network, but with a 1D output.
        dummy_value_params = self.value_network.init(rng, dummy_observation)
        value_output = self.value_network.apply(dummy_value_params, dummy_observation)
        assert (
            value_output.shape[-1] == 1
        ), f"value network output size {value_output.shape} does not match expected size 1"

        def value_init(key):
            """Initialize the value network from a random key."""
            return self.value_network.init(key, dummy_observation)

        def value_apply(processor_params, value_params, obs):
            """Apply the value function given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return jp.squeeze(self.value_network.apply(value_params, obs), axis=-1)

        return PPONetworks(
            policy_network=networks.FeedForwardNetwork(init=policy_init, apply=policy_apply),
            value_network=networks.FeedForwardNetwork(init=value_init, apply=value_apply),
            parametric_action_distribution=action_dist,
        )


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
                use_bias=self.bias,
                name=f"dense_{i}",
            )(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = nn.relu(x)
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
