import functools
import pickle
import time
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import scipy
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import make_inference_fn
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import (
    MLP,
    HierarchyComposition,
    NestedLinearPolicy,
    ParallelComposition,
    SeriesComposition,
)
from ambersim.rl.cart_pole.swingup import CartPoleSwingupEnv
from ambersim.rl.helpers import BraxPPONetworksWrapper

"""
This example demonstrates using brax PPO + MJX to train a policy for a simple
cart-pole.
"""


def train():
    """Train a policy to swing up the cart-pole, then save the trained policy."""
    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", CartPoleSwingupEnv)
    env = envs.get_environment("cart_pole")

    # Use a custom network architecture
    print("Creating policy network...")
    num_modules = 7
    measurement_networks = [MLP for _ in range(num_modules)]
    measurement_network_kwargs = [{"layer_sizes": (128, 2)} for _ in range(num_modules)]
    linear_policy_kwargs = [{"features": 2} for _ in range(num_modules)]
    policy_network = NestedLinearPolicy(measurement_networks, measurement_network_kwargs, linear_policy_kwargs)

    value_network = MLP(layer_sizes=[256, 256, 1])
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
    )
    print(policy_network)

    # Set the number of training steps and evaluations
    num_timesteps = 5_000_000
    eval_every = 100_000
    num_evals = num_timesteps // eval_every

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        episode_length=200,
        reward_scaling=0.1,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=64,
        num_updates_per_batch=16,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-5,
        num_envs=1024,
        batch_size=512,
        clipping_epsilon=0.2,
        network_factory=network_wrapper.make_ppo_networks,
        seed=0,
    )

    # Set up tensorboard logging
    log_dir = f"/tmp/mjx_brax_logs/cart_pole_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def introspect():
    """Load a trained policy and inspect the output of various layers."""
    # Create an environment for evaluation
    print("Creating test environment...")
    envs.register_environment("cart_pole", CartPoleSwingupEnv)
    env = envs.get_environment("cart_pole")
    mj_model = env.model

    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[1] = 0.0
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    # Load the saved policy
    print("Loading saved params ...")
    params_path = "/tmp/cart_pole_params.pkl"
    networks_path = "/tmp/cart_pole_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy network
    print("Creating policy network...")
    assert isinstance(network_wrapper.policy_network, HierarchyComposition)

    # Define some functions for looking at various layers
    @jax.jit
    def get_final_output(obs):
        return network_wrapper.policy_network.apply(params[1], obs)

    @jax.jit
    def get_first_layer_output(obs):
        return network_wrapper.policy_network.apply(params[1], obs, method=lambda net, x: net.modules[0](x))

    print("Jitting policy functions...")
    u = get_final_output(obs)
    print("u:", u)
    u0 = get_first_layer_output(obs)
    print("u0:", u0)

    # Run a little simulation
    print("Running simulation...")
    num_steps = 150
    u0s = []
    us = []
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for _ in range(num_steps):
            step_start = time.time()

            # Compute and record network outputs
            u0 = get_first_layer_output(obs)
            u = get_final_output(obs)
            u0s.append(u0)
            us.append(u)

            # Apply the policy
            mj_data.ctrl = u[0]
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

    # Plot the network outputs
    u0s = np.array(u0s)
    us = np.array(us)
    t = float(env.dt) * np.arange(num_steps)

    plt.subplot(2, 1, 1)
    plt.plot(t, u0s[:, 0])
    plt.xlabel("time (s)")
    plt.ylabel("first layer output")

    plt.subplot(2, 1, 2)
    plt.plot(t, us[:, 0])
    plt.xlabel("time (s)")
    plt.ylabel("final output")

    # Make another plot with frequency analysis
    plt.figure()

    # Normalize u0s and us
    u0s = (u0s - np.mean(u0s)) / np.std(u0s)
    us = (us - np.mean(us)) / np.std(us)

    u0s_fft = scipy.fft.rfft(u0s[:, 0])
    u0s_freq = scipy.fft.rfftfreq(len(u0s), d=float(env.dt))
    us_fft = scipy.fft.rfft(us[:, 0])
    us_freq = scipy.fft.rfftfreq(len(us), d=float(env.dt))

    width = 0.2
    plt.bar(u0s_freq - width / 2, np.abs(u0s_fft), width=width, label="top layer")
    plt.bar(us_freq + width / 2, np.abs(us_fft), width=width, label="bottom layer")

    plt.xlabel("frequency (Hz)")
    plt.ylabel("amplitude")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
    # test()
    # introspect()
