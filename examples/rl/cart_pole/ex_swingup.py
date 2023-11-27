import functools
import pickle
import time
from datetime import datetime

import jax
import mujoco
import mujoco.viewer
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import make_inference_fn
from jax import numpy as jp
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP, HierarchyComposition, ParallelComposition, SeriesComposition
from ambersim.rl.base import MjxEnv, State
from ambersim.rl.cart_pole.swingup import CartPoleSwingupEnv
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.utils.io_utils import load_mj_model_from_file

"""
This example demonstrates using brax PPO + MJX to train a policy for a simple
cart-pole.
"""


# class CartPole(MjxEnv):
#    """Training environment for a simple cart-pole."""
#
#    def __init__(
#        self,
#        upright_angle_cost: float = 1.0,
#        center_cart_cost: float = 0.01,
#        velocity_cost: float = 0.01,
#        control_cost: float = 0.001,
#        termination_threshold: float = 0.2,
#        **kwargs,
#    ):
#        """Initializes the cart-pole environment."""
#        path = "models/cart_pole/cart_pole.xml"
#        mj_model = load_mj_model_from_file(path)
#        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
#
#        physics_steps_per_control_step = 1
#        kwargs["physics_steps_per_control_step"] = kwargs.get(
#            "physics_steps_per_control_step", physics_steps_per_control_step
#        )
#
#        super().__init__(mj_model=mj_model, **kwargs)
#
#        self._upright_angle_cost = upright_angle_cost
#        self._center_cart_cost = center_cart_cost
#        self._velocity_cost = velocity_cost
#        self._control_cost = control_cost
#
#        # Stop the episode if the pole falls over too far
#        self._termination_threshold = termination_threshold
#
#    def reset(self, rng: jp.ndarray) -> State:
#        """Resets the environment to a new initial state."""
#        rng, rng1, rng2 = jax.random.split(rng, 3)
#
#        low = -(self._termination_threshold - 0.05)
#        hi = self._termination_threshold - 0.05
#        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
#        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
#
#        data = self.pipeline_init(qpos, qvel)
#        obs = self._get_obs(data, jp.zeros(self.sys.nu))
#        reward, done, zero = jp.zeros(3)
#        metrics = {
#            "reward": zero,
#            "upright_reward": zero,
#            "center_cart_reward": zero,
#            "velocity_reward": zero,
#            "control_reward": zero,
#        }
#        return State(data, obs, reward, done, metrics)
#
#    def step(self, state: State, action: jp.ndarray) -> State:
#        """Steps the environment forward one timestep."""
#        data0 = state.pipeline_state
#        data = self.pipeline_step(data0, action)
#        obs = self._get_obs(data, action)
#
#        # Compute the reward
#        upright_reward = 1.0 - self._upright_angle_cost * obs[1] ** 2
#        center_cart_reward = -self._center_cart_cost * obs[0] ** 2
#        velocity_reward = -self._velocity_cost * (obs[2] ** 2 + obs[3] ** 2)
#        control_reward = -self._control_cost * action[0] ** 2
#        reward = upright_reward + center_cart_reward + velocity_reward + control_reward
#
#        # Check if the episode is done
#        done = jp.where(jp.abs(obs[1]) > self._termination_threshold, 1.0, 0.0)
#
#        state.metrics.update(
#            reward=reward,
#            upright_reward=upright_reward,
#            center_cart_reward=center_cart_reward,
#            velocity_reward=velocity_reward,
#            control_reward=control_reward,
#        )
#
#        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
#
#    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
#        """Return the observation, [cos(theta), sin(theta), theta_dot]."""
#        return jp.concatenate([data.qpos, data.qvel])


def train():
    """Train a policy to swing up the cart-pole, then save the trained policy."""
    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", CartPoleSwingupEnv)
    env = envs.get_environment("cart_pole")

    # Use a custom network architecture
    print("Creating policy network...")
    policy_network = HierarchyComposition(module_type=MLP, num_modules=3, module_kwargs={"layer_sizes": [512, 2]})
    value_network = MLP(layer_sizes=[256, 256, 1])
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
    )

    print(policy_network)

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=1_000_000,
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

    # Set up tensorboard logging
    log_dir = "/tmp/mjx_brax_logs/cart_pole"
    print(f"Setting up tensorboard at {log_dir}...")
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
    params_path = "/tmp/cart_pole_params.pkl"
    networks_path = "/tmp/cart_pole_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


def test(start_angle=0.0):
    """Load a trained policy and run a little sim with it."""
    # Create an environment for evaluation
    print("Creating test environment...")
    envs.register_environment("cart_pole", CartPoleSwingupEnv)
    env = envs.get_environment("cart_pole")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data.qpos[1] = start_angle
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    # Load the saved policy
    print("Loading policy ...")
    params_path = "/tmp/cart_pole_params.pkl"
    networks_path = "/tmp/cart_pole_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy network
    print("Creating policy network...")
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    print(network_wrapper.policy_network)

    make_policy = make_inference_fn(ppo_networks)
    policy = make_policy(params, deterministic=True)
    jit_policy = jax.jit(policy)

    # Run a little sim
    rng = jax.random.PRNGKey(0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            act_rng, rng = jax.random.split(rng)

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = act
            obs = env.compute_obs(mjx.device_put(mj_data), {})

            # Step the simulation
            for _ in range(env._physics_steps_per_control_step):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly real time
            elapsed = time.time() - step_start
            dt = float(env.dt)
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    train()
    test()
