from typing import Any, Dict

import brax
import jax
import jax.numpy as jnp
from mujoco import mjx

from ambersim.rl.base import MjxEnv, State

"""
Wrappers for Mjx environments.
"""


class RecurrentWrapper(MjxEnv):
    """Environment for training recurrent policies with Brax.

    This environment expects actions to be a tuple of (z_next, u), where u is the
    standard action and z_next is the next hidden state. The observations are then
    (z, y), where z is the current hidden state and y is the standard observation.

    The hidden state z is stored in the info dictionary.

    Note: this is a kind of odd way to think about recurrent policies, as z_next
    is actually sampled from a distribution just like u. However, it seems to work
    and allows us to use Brax's existing RL algorithms like PPO.
    """

    @staticmethod
    def env_factory(env: MjxEnv, nz: int, z_cost_weight: float = 1e-4) -> MjxEnv:
        """Create a function that returns a recurrent wrapper around the given environment.

        Args:
            env: the underlying environment we want to wrap
            nz: the size of the hidden state
            z_cost_weight: the weight on penalizing |z|^2

        Returns:
            make_env: a function that returns a wrapped MjxEnv
        """

        def make_env(*args) -> MjxEnv:
            return RecurrentWrapper(env(*args), nz=nz, z_cost_weight=z_cost_weight)

        return make_env

    def __init__(self, env: MjxEnv, nz: int, z_cost_weight: float = 1e-4) -> None:
        """Initialize the environment.

        Args:
            env: the underlying environment we want to wrap
            nz: the size of the hidden state
            z_cost_weight: the weight on penalizing |z|^2
        """
        self.env = env
        self.nz = nz
        self.z_cost_weight = z_cost_weight

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment.

        The hidden state z is reset to zero.

        Args:
            rng: the random number generator

        Returns:
            state: the initial state
        """
        # Reset the underlying environment
        env_state = self.env.reset(rng)

        # Reset the hidden state
        info = env_state.info
        info["z"] = jnp.zeros(self.nz)

        # Create the new observation [z, y]
        obs = jnp.concatenate([info["z"], env_state.obs])
        state = env_state.replace(info=info, obs=obs)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment.

        Args:
            state: the current state
            action: the action, which is [z_next, u]

        Returns:
            state: the next state
        """
        # Advance the underlying environment
        env_state = state.replace(obs=state.obs[self.nz :])
        env_state = self.env.step(env_state, action[self.nz :])

        # Advance the hidden state and create the new observation [z, y]
        z = action[: self.nz]
        obs = jnp.concatenate([z, env_state.obs])

        # Modify the reward to include a cost on the hidden state
        reward = env_state.reward - self.z_cost_weight * jnp.square(z).sum()

        # Update the state
        state = env_state.replace(obs=obs, reward=reward)
        state.info["z"] = z
        return state

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Observes the environment based on the system State. May modify state in place.

        Args:
            data: The physics state.
            info: Auxiliary info from the State.

        Returns:
            obs: the observation.
        """
        y = self.env.compute_obs(data, info)
        z = info["z"]
        return jnp.concatenate([z, y])

    @property
    def dt(self) -> jax.Array:
        """The timestep of the environment."""
        return self.env.dt

    @property
    def observation_size(self) -> int:
        """The size of the observation."""
        return self.env.observation_size + self.nz

    @property
    def action_size(self) -> int:
        """The size of the action."""
        return self.env.action_size + self.nz

    @property
    def model(self) -> mjx.Model:
        """The MuJoCo model."""
        return self.env.model

    @property
    def _physics_steps_per_control_step(self) -> int:
        """The number of physics steps per control step."""
        return self.env._physics_steps_per_control_step
