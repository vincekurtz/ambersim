import functools
import pickle
import sys
import time
from datetime import datetime
from typing import Any, Dict

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import make_inference_fn
from mujoco import mjx

from ambersim.learning.architectures import MLP
from ambersim.rl.base import MjxEnv, State
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.utils.io_utils import load_mj_model_from_file

"""
Perform pendulum swingup training with a Koopman linear system policy.
"""


class KoopmanPendulumSwingupEnv(MjxEnv):
    """Environment for training a torque-constrained pendulum swingup task.

    Includes extra utils so the (lifted) state of the controller is included.
    TODO: make this a wrapper around an existing env.
    """

    def __init__(self, nz=0) -> None:
        """Initialize the environment.

        Args:
            nz: the dimension of the lifted state.
        """
        # Problem config parameters
        model_path = "models/pendulum/scene.xml"
        physics_steps_per_control_step = 1
        self.theta_cost_weight = 1.0
        self.theta_dot_cost_weight = 0.1
        self.control_cost_weight = 0.001
        self.qpos_hi = jnp.pi
        self.qpos_lo = -jnp.pi
        self.qvel_hi = 2
        self.qvel_lo = -2

        # Initialize the environment
        mj_model = load_mj_model_from_file(model_path)
        super().__init__(
            mj_model,
            physics_steps_per_control_step,
        )

        # Store lifting dimension
        self.nz = nz

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Observes the environment based on the system State. See parent docstring."""
        theta = data.qpos[0]
        obs = jnp.stack((jnp.cos(theta), jnp.sin(theta), data.qvel[0]))
        return obs

    def compute_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the reward for the current environment state.

        Returns:
            reward (shape=(1,)): the reward, maximized at qpos[0] = np.pi.
        """
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        tau = data.ctrl[0]

        # Compute a normalized theta error
        theta_err = theta - jnp.pi
        theta_err_normalized = jnp.arctan2(jnp.sin(theta_err), jnp.cos(theta_err))

        # Compute the reward
        reward_theta = -self.theta_cost_weight * jnp.square(theta_err_normalized).sum()
        reward_theta_dot = -self.theta_dot_cost_weight * jnp.square(theta_dot).sum()
        reward_tau = -self.control_cost_weight * jnp.square(tau).sum()

        return reward_theta + reward_theta_dot + reward_tau

    def reset(self, rng: jax.Array) -> State:
        """Resets the env. See parent docstring."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # reset the positions and velocities
        qpos = jax.random.uniform(rng1, (self.sys.nq,), minval=self.qpos_lo, maxval=self.qpos_hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=self.qvel_lo, maxval=self.qvel_hi)
        data = self.pipeline_init(qpos, qvel)

        # other state fields
        obs = self.compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {"reward": reward}
        state_info = {"rng": rng, "step": 0}
        state = State(data, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Takes a step in the environment. See parent docstring."""
        rng, rng_obs = jax.random.split(state.info["rng"])

        # physics + observation + reward
        data = self.pipeline_step(state.pipeline_state, action)  # physics
        obs = self.compute_obs(data, state.info)  # observation
        reward = self.compute_reward(data, state.info)
        done = 0.0  # pendulum just runs for a fixed number of steps

        # updating state
        state.info["step"] = state.info["step"] + 1
        state.info["rng"] = rng
        state.metrics["reward"] = reward
        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        return state


def train_swingup():
    """Train a pendulum swingup agent with custom network architectures."""
    # Initialize the environment
    envs.register_environment("pendulum_swingup", KoopmanPendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Policy network takes as input observations and the current lifted state.
    # It outputs a mean and standard deviation for the action and the next lifted state.
    # N.B. a one layer MLP is just a linear map, so this is a linear policy.
    policy_network = MLP(layer_sizes=(2 * env.action_size,))

    # Value network takes as input observations and the current lifted state,
    # and outputs a scalar value.
    value_network = MLP(layer_sizes=(128, 1))
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
    )

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
        network_factory=network_wrapper.make_ppo_networks,
        seed=0,
    )

    # Define a callback to log progress
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

    # Do the training
    print("Training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save both the parameters and the networks to disk
    print("Saving...")
    params_path = "/tmp/pendulum_params.pkl"
    networks_path = "/tmp/pendulum_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


def test_trained_swingup_policy():
    """Load a trained policy and run an interactive simulation."""
    envs.register_environment("pendulum_swingup", KoopmanPendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    print("Loading trained policy...")
    params_path = "/tmp/pendulum_params.pkl"
    networks_path = "/tmp/pendulum_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )

    make_policy = make_inference_fn(ppo_networks)
    policy = make_policy(params, deterministic=True)
    jit_policy = jax.jit(policy)

    print("Simulating...")
    rng = jax.random.PRNGKey(0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            act_rng, rng = jax.random.split(rng)

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = act
            obs = env.compute_obs(mjx.device_put(mj_data), {})

            # Step the simulation
            for _ in range(env._physics_steps_per_control_step):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # TODO: step the MJX env too so the lifted state is updated

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            dt = float(env.dt)
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    usage_message = "Usage: python ex_koopman_policy.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train_swingup()
    elif sys.argv[1] == "test":
        test_trained_swingup_policy()
    else:
        print(usage_message)
        sys.exit(1)
