from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from etils import epath
from mujoco import mjx

from ambersim.rl.base import MjxEnv, State
from ambersim.utils.io_utils import load_mj_model_from_file

@struct.dataclass
class HumanoidForwardConfig:
    """Config dataclass for the humanoid forward locomotion task."""

    # Location of the model file
    model_path = Path(epath.resource_path('mujoco')) / ('mjx/benchmark/model/humanoid')

    # Number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 5

    # Reward function coefficients
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    healthy_reward: float = 5.0

    # Other settings
    terminate_when_unhealthy: bool = True
    healthy_z_range: Tuple[float, float] = (1.0, 2.0)
    exclude_current_positions_from_observation: bool = True

    # Range for sampling initial conditions 
    reset_noise_scale: float = 1e-2


class HumanoidForwardEnv(MjxEnv):
    """Environment for training a humanoid to run forward.

    Adopted from the MJX tutorial: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
    """

    def __init__(self, config: Optional[HumanoidForwardConfig] = None) -> None:
        """Initialize the humanoid forward env. See parent docstring."""
        if config is None:
            config = HumanoidForwardConfig()
        self.config = config

        # Load the model
        model_path = config.model_path / 'humanoid.xml'
        mj_model = load_mj_model_from_file(config.model_path / 'humanoid.xml')
        
        # Set solver parameters
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Compute the observation from teh system State. See parent docstring."""
        position = data.qpos
        if self.config.exclude_current_positions_from_observation:
            position = position[2:]

        return jnp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to a random initial state. See parent docstring."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)

        obs = self.compute_obs(data, {})
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward': reward,
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
        }
        state_info = {'rng': rng, 'step': 0}
        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """Take a step in the environment. See parent docstring."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self.config.forward_reward_weight * velocity[0]

        min_z, max_z = self.config.healthy_z_range
        is_healthy = jnp.where(data.qpos[2] < min_z, x=0.0, y=1.0)
        is_healthy = jnp.where(
            data.qpos[2] > max_z, x=0.0, y=is_healthy
        )
        if self.config.terminate_when_unhealthy:
            healthy_reward = self.config.healthy_reward
        else:
            healthy_reward = self.config.healthy_reward * is_healthy

        ctrl_cost = self.config.ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self.compute_obs(data, state.info)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self.config.terminate_when_unhealthy else 0.0
        state.metrics.update(
            reward=reward,
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )
        state.info['step'] += 1

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )
