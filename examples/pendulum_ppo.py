from typing import Tuple

import jax
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax import envs
from brax.base import Motion, Transform
from brax.envs.base import Env, State
from brax.io import model
from brax.training.agents.ppo import train as ppo
from etils import epath
from jax import numpy as jp
from mujoco import mjx

from ambersim.utils.io_utils import load_mj_model_from_file

"""
This example demonstrates using brax PPO + MJX to train a pendulum swingup task.
Loosely based on the tutorial at

https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
"""


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ):
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return (
                mjx.step(self.sys, data),
                None,
            )

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation vector."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        """Returns the size of the action vector."""
        return self.sys.nu

    @property
    def backend(self) -> str:
        """Returns the backend used by this environment."""
        return "mjx"

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd


class Pendulum(MjxEnv):
    """Training environment for a simple pendulum."""

    def __init__(
        self,
        control_cost_weight=0.1,
        theta_cost_weight=1.0,
        theta_dot_cost_weight=1.0,
        reset_noise_scale=0.1,
        **kwargs,
    ):
        """Initializes the pendulum environment.

        Args:
          control_cost_weight: cost coefficient for control inputs
          theta_cost_weight: cost coefficient for reaching the target angle
          theta_dot_cost_weight: cost coefficient for velocities
          reset_noise_scale: scale of gaussian noise added to qpos on reset
        """
        path = "models/pendulum/pendulum.xml"
        mj_model = load_mj_model_from_file(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG

        physics_steps_per_control_step = 4
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )

        super().__init__(mj_model=mj_model, **kwargs)

        self._control_cost_weight = control_cost_weight
        self._theta_cost_weight = theta_cost_weight
        self._theta_dot_cost_weight = theta_dot_cost_weight
        self._reset_noise_scale = reset_noise_scale

        self._theta_nom = jp.array([jp.pi])

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_theta": zero,
            "reward_theta_dot": zero,
            "reward_ctrl": zero,
            "theta": zero,
            "theta_dot": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Steps the environment forward one timestep."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        theta = data.qpos[0]
        theta_dot = data.qvel[0]

        reward_theta = -self._theta_cost_weight * jp.square(theta - self._theta_nom).sum()
        reward_theta_dot = -self._theta_dot_cost_weight * jp.square(theta_dot).sum()
        reward_ctrl = -self._control_cost_weight * jp.square(action).sum()
        reward = reward_theta + reward_theta_dot + reward_ctrl

        obs = self._get_obs(data, action)
        done = 1.0

        state.metrics.update(
            reward_theta=reward_theta,
            reward_theta_dot=reward_theta_dot,
            reward_ctrl=reward_ctrl,
            theta=theta,
            theta_dot=theta_dot,
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Return the observation, which is the current state."""
        return jp.concatenate([data.qpos, data.qvel])


def visualize_open_loop():
    """Save a video of an open-loop trajectory, using just mujoco."""
    # Set up a model and a renderer
    mj_model = load_mj_model_from_file("models/pendulum/pendulum.xml")
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)

    # Set the initial state
    mj_data.qpos[0] = 1.2

    # Simulate a trajectory
    duration = 3.0  # seconds
    fps = 60
    frames = []
    while mj_data.time < duration:
        mujoco.mj_step(mj_model, mj_data)
        if len(frames) < mj_data.time * fps:
            renderer.update_scene(mj_data)
            frame = renderer.render()
            frames.append(frame)

    media.write_video("pendulum_open_loop.mp4", frames, fps=fps)


if __name__ == "__main__":
    visualize_open_loop()
