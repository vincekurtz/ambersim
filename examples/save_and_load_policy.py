#!/usr/bin/env python

import pickle
from typing import Sequence

import flax.linen as nn
import jax


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


if __name__ == "__main__":
    save_to_file()
    print("")
    load_from_file()
