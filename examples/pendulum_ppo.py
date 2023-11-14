import functools
import time
from datetime import datetime

# mujoco viewer must be imported before jax on 22.04
# isort: off
import mujoco
import mujoco.viewer

# isort: on


import pickle

import jax
import matplotlib.pyplot as plt
from brax import envs
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax import numpy as jp
from mujoco import mjx
from torch.utils.tensorboard import SummaryWriter

from ambersim.rl.base import MjxEnv, State
from ambersim.rl.policies import PPONetworkConfig, make_ppo_networks_from_config
from ambersim.utils.io_utils import load_mj_model_from_file

"""
This example demonstrates using brax PPO + MJX to train a pendulum swingup task.
Loosely based on the tutorial at

https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
"""


class Pendulum(MjxEnv):
    """Training environment for a simple pendulum."""

    def __init__(
        self,
        control_cost_weight=0.001,
        theta_cost_weight=1.0,
        theta_dot_cost_weight=0.1,
        reset_noise_scale=4.0,
        **kwargs,
    ):
        """Initializes the pendulum environment.

        Args:
          control_cost_weight: cost coefficient for control inputs
          theta_cost_weight: cost coefficient for reaching the target angle
          theta_dot_cost_weight: cost coefficient for velocities
          reset_noise_scale: scale of gaussian noise added to qpos on reset
        """
        path = "models/pendulum/scene.xml"
        mj_model = load_mj_model_from_file(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG

        physics_steps_per_control_step = 1
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
            "reward": zero,
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

        # TODO: normalize reward_theta
        theta_err = theta - self._theta_nom
        s = jp.sin(theta_err)
        c = jp.cos(theta_err)
        theta_err_normalized = jp.arctan2(s, c)

        reward_theta = -self._theta_cost_weight * jp.square(theta_err_normalized).sum()
        reward_theta_dot = -self._theta_dot_cost_weight * jp.square(theta_dot).sum()
        reward_ctrl = -self._control_cost_weight * jp.square(action).sum()
        reward = reward_theta + reward_theta_dot + reward_ctrl

        obs = self._get_obs(data, action)
        done = 0.0

        state.metrics.update(
            reward=reward,
            reward_theta=reward_theta,
            reward_theta_dot=reward_theta_dot,
            reward_ctrl=reward_ctrl,
            theta=theta,
            theta_dot=theta_dot,
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Return the observation, [cos(theta), sin(theta), theta_dot]."""
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        return jp.array([jp.cos(theta), jp.sin(theta), theta_dot])


def train():
    """Train a policy to swing up the pendulum, then save the trained policy."""
    print("Creating pendulum environment...")
    envs.register_environment("pendulum", Pendulum)
    env = envs.get_environment("pendulum")

    # Create the PPO agent
    print("Creating PPO agent...")

    config = PPONetworkConfig(
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
    )
    network_factory = functools.partial(make_ppo_networks_from_config, config=config)

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100_000,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=1024,
        batch_size=512,
        clipping_epsilon=0.2,
        network_factory=network_factory,
        seed=0,
    )

    # Set up tensorboard logging
    log_dir = "/tmp/mjx_brax_logs/pendulum"
    print(f"Setting up tensorboard logging in {log_dir} ...")
    writer = SummaryWriter(log_dir)

    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Helper function for recording training progress."""
        reward = metrics["eval/episode_reward"]
        std = metrics["eval/episode_reward_std"]
        print(f"    Step: {num_steps},  Reward: {reward:.3f},  Std: {std:.3f}")

        # Record the current wall clock time
        times.append(datetime.now())

        # Write all the metrics to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)
            writer.add_scalar(key, val, num_steps)

    print("Training...")
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

    print(f"  time to jit: {times[1] - times[0]}")
    print(f"  time to train: {times[-1] - times[1]}")

    # Save the trained policy
    print("Saving trained policy...")
    params_path = "/tmp/pendulum_params"
    config_path = "/tmp/pendulum_config"
    model.save_params(params_path, params)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)


def test(start_angle=0.0):
    """Load a trained policy and run a little sim with it."""
    # Create an environment for evaluation
    print("Creating test environment...")
    envs.register_environment("pendulum", Pendulum)
    env = envs.get_environment("pendulum")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    rng = jax.random.PRNGKey(0)
    ctrl = jp.zeros(mj_model.nu)

    # Load the saved policy
    print("Loading policy...")
    params_path = "/tmp/pendulum_params"
    config_path = "/tmp/pendulum_config"
    params = model.load_params(params_path)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    # Create the policy network
    print("Creating policy network...")
    ppo_network = make_ppo_networks_from_config(
        env.observation_size, env.action_size, config, preprocess_observations_fn=running_statistics.normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    policy = make_inference_fn(params)
    jit_policy = jax.jit(policy)

    # Set the initial state
    mj_data.qpos[0] = start_angle
    mj_data.qvel[0] = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            act_rng, rng = jax.random.split(rng)

            # Get the last observation
            obs = env._get_obs(mjx.device_put(mj_data), ctrl)

            # Compute the control action for the next step
            ctrl, _ = jit_policy(obs, act_rng)
            mj_data.ctrl = ctrl

            # Step the simulation
            for _ in range(env._physics_steps_per_control_step):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly real time
            elapsed = time.time() - step_start
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)


if __name__ == "__main__":
    train()
    test(3.14)
