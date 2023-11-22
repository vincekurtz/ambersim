#!/usr/bin/env python

import pickle
from typing import Sequence

import flax.linen as nn
import jax

from ambersim.rl.policies import PPONetworkConfig, make_ppo_networks_from_config


class MLP(nn.Module):
    """Simple MLP model."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        """Forward pass through the network."""
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        return nn.Dense(self.features[-1])(x)


def save_to_file():
    """Create a model with a given size and save it to a file."""
    my_mlp = MLP(features=[4, 8, 1])

    # Initialize parameters for this model
    rng = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(rng, (1, 4))
    params = my_mlp.init(rng, dummy_input)
    print(my_mlp.apply(params, dummy_input))

    # Save parameters to a file
    with open("/tmp/my_mlp_params.pkl", "wb") as f:
        pickle.dump(params, f)

    # Save the model to a file
    with open("/tmp/my_mlp_model.pkl", "wb") as f:
        pickle.dump(my_mlp, f)

    print(my_mlp)
    print(my_mlp.apply(params, dummy_input).shape)


def load_from_file():
    """Load the model from a file and apply it to some input."""
    # Load the model from a file
    with open("/tmp/my_mlp_model.pkl", "rb") as f:
        saved_mlp = pickle.load(f)

    # Load parameters from a file
    with open("/tmp/my_mlp_params.pkl", "rb") as f:
        saved_params = pickle.load(f)

    # Apply the model to some input
    rng = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(rng, (1, 4))
    print(saved_mlp.apply(saved_params, dummy_input))


def save_ppo_networks():
    """Create a model with a given size and save it to a file."""
    config = PPONetworkConfig(
        policy_hidden_layer_sizes=(8, 8),
        value_hidden_layer_sizes=(8, 8),
    )
    my_ppo_networks = make_ppo_networks_from_config(4, 2, config)
    dist = my_ppo_networks.parametric_action_distribution

    with open("/tmp/my_dist.pkl", "wb") as f:
        pickle.dump(dist, f)

    with open("/tmp/my_dist.pkl", "rb") as f:
        loaded_dist = pickle.load(f)

    print(loaded_dist)


if __name__ == "__main__":
    # save_ppo_networks()
    save_to_file()
    # print("")
    # load_from_file()
