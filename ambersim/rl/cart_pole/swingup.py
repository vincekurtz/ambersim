from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco as mj
from flax import struct
from mujoco import mjx

from ambersim.rl.base import MjxEnv, State
from ambersim.utils.io_utils import load_mj_model_from_file


@struct.dataclass
class CartPoleSwingupConfig:
    """Config dataclass for cart-pole swingup."""

    # model path
    model_path: Union[Path, str] = "models/cart_pole/cart_pole.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 1

    # Reward function coefficients
    upright_angle_cost: float = 1.0
    center_cart_cost: float = 0.01
    velocity_cost: float = 0.01
    control_cost: float = 0.001

    # Ranges for sampling initial conditions
    pos_hi: float = 0
    pos_lo: float = -0
    # theta_hi: float = jnp.pi
    # theta_lo: float = -jnp.pi
    theta_hi: float = 0.001
    theta_lo: float = -0.001
    qvel_hi: float = 0.001
    qvel_lo: float = -0.001


class CartPoleSwingupEnv(MjxEnv):
    """Environment for training a cart-pole swingup task.

    States: x = (pos, theta, vel, dtheta), shape=(4,)
    Observations: y = (pos, cos(theta), sin(theta), vel, dtheta), shape=(5,)
    Actions: a = tau, the force on the cart, shape=(1,)
    """

    def __init__(self, config: Optional[CartPoleSwingupConfig] = None) -> None:
        """Initialize the swingup env. See parent docstring."""
        if config is None:
            config = CartPoleSwingupConfig()
        self.config = config
        mj_model = load_mj_model_from_file(config.model_path)

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the observation from the state. See parent docstring."""
        p = data.qpos[0]
        s = jnp.sin(data.qpos[1])
        c = jnp.cos(data.qpos[1])
        pd = data.qvel[0]
        td = data.qvel[1]
        return jnp.array(
            [
                p,
                c,
                s,
                pd,
                td,
                p**2,
                c**2,
                s**2,
                pd**2,
                td**2,
                p * c,
                p * s,
                p * pd,
                p * td,
                c * s,
                c * pd,
                c * td,
                s * pd,
                s * td,
                pd * td,
            ]
        )

    def compute_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the reward from the state. See parent docstring."""
        pos = data.qpos[0]
        theta = data.qpos[1]

        # Compute a normalized angle error (upright is zero)
        theta_err_normalized = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

        # Compute the reward
        upright_reward = -self.config.upright_angle_cost * jnp.square(theta_err_normalized).sum()
        center_cart_reward = -self.config.center_cart_cost * jnp.square(pos).sum()
        velocity_reward = -self.config.velocity_cost * jnp.square(data.qvel).sum()
        control_reward = -self.config.control_cost * jnp.square(data.ctrl).sum()

        return upright_reward + center_cart_reward + velocity_reward + control_reward

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Reset positions and velocities
        qpos_lo = jnp.array([self.config.pos_lo, self.config.theta_lo])
        qpos_hi = jnp.array([self.config.pos_hi, self.config.theta_hi])
        qpos = jax.random.uniform(rng1, (self.sys.nq,), minval=qpos_lo, maxval=qpos_hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=self.config.qvel_lo, maxval=self.config.qvel_hi)
        data = self.pipeline_init(qpos, qvel)

        # Other state fields
        obs = self.compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {"reward": reward}
        state_info = {"rng": rng, "step": 0}
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Steps the environment forward one timestep."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self.compute_obs(data, state.info)

        # Compute the reward
        reward = self.compute_reward(data, state.info)
        done = 0.0

        # Update metrics
        state.info["step"] += 1
        state.metrics["reward"] = reward
        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        return state
