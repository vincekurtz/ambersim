import jax
import jax.numpy as jnp
from brax.training import distribution

"""
Tools for defining brax probability distributions.
"""


class IdentityBijector:
    """Identity postprocessing function for use with brax's ParametricDistribution."""

    def forward(self, x):
        """Identity function."""
        return x

    def inverse(self, x):
        """Inverse of identity is identity."""
        return x

    def forward_log_det_jacobian(self, x):
        """Log det jacobian of identity is zero."""
        return 0.0


class NormalDistribution(distribution.ParametricDistribution):
    """A simple normal distribution.

    Based on distribution.NormalTanhDistribution, but without the extra tanh that
    bounds the mean between [-1,1].
    """

    def __init__(self, event_size, min_std=1e-3):
        """Initialize the distirbution.

        Args:
            event_size: the dimension of the distribution.
            min_std: the minimum standard deviation.
        """
        super().__init__(
            param_size=2 * event_size,  # mean and diagonal covariance
            postprocessor=IdentityBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std

    def create_dist(self, parameters):
        """Create a distribution from parameters."""
        mean, raw_std = jnp.split(parameters, 2, axis=-1)
        std = jax.nn.softplus(raw_std) + self._min_std
        return distribution.NormalDistribution(loc=mean, scale=std)
