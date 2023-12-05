import functools
import pickle
import time
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
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
    print_module_summary,
)
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.rl.humanoid.forward import HumanoidForwardEnv

"""
Demonstrates using brax PPO + MJX to train a humanoid forward locomotion task.

Adopted from the MJX tutorial: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
"""


def train():
    """Train a humanoid running policy, then save the trained policy."""
    print("Creating humanoid environment...")
    envs.register_environment("humanoid_forward", HumanoidForwardEnv)
    env = envs.get_environment("humanoid_forward")

    # Use a custom network architecture
    print("Creating policy network...")
    # policy_network = MLP(layer_sizes=[32, 32, 32, 32, 2*env.action_size])
    # policy_network = HierarchyComposition(
    #    MLP,
    #    num_modules=3,
    #    module_kwargs={"layer_sizes": (32, 2*env.action_size)},
    # )
    policy_network = NestedLinearPolicy(
        measurement_networks=[MLP, MLP, MLP, MLP, MLP],
        measurement_network_kwargs=[
            {"layer_sizes": (8, 4)},
            {"layer_sizes": (8, 4)},
            {"layer_sizes": (16, 8)},
            {"layer_sizes": (32, 16)},
            {"layer_sizes": (64, 32)},
        ],
        linear_policy_kwargs=[
            {"features": 4},
            {"features": 4},
            {"features": 8},
            {"features": 16},
            {"features": 2 * env.action_size},
        ],
    )

    value_network = MLP(layer_sizes=[256, 256, 256, 256, 256, 1])

    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
    )
    print_module_summary(network_wrapper.policy_network, env.observation_size)

    # Create the PPO agent
    num_timesteps = 90_000_000
    eval_every = 500_000
    num_evals = num_timesteps // eval_every
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        clipping_epsilon=0.3,
        network_factory=network_wrapper.make_ppo_networks,
        seed=0,
    )

    # Set up tensorboard logging
    logdir = f"/tmp/mjx_brax_logs/humanoid_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Setting up tensorboard at {logdir}...")
    writer = SummaryWriter(logdir)
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Helper function to log progress to tensorboard."""
        reward = metrics["eval/episode_reward"]
        std = metrics["eval/episode_reward_std"]
        print(f"    Step: {num_steps} | Reward: {reward} +/- {std}")

        # Record wall-clock time
        times.append(datetime.now())

        # Log to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)
            writer.add_scalar(key, val, num_steps)

    print("Training...")
    _, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f"  time to jit: {times[1] - times[0]}")
    print(f"  time to train: {times[-1] - times[1]}")

    # Save the trained policy
    print("Saving policy...")
    params_path = "/tmp/humanoid_params.pkl"
    networks_path = "/tmp/humanoid_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


def test():
    """Load a trained policy and run an interactive simulation."""
    # Create an environment for evaluation
    print("Creating test environment...")
    envs.register_environment("humanoid_forward", HumanoidForwardEnv)
    env = envs.get_environment("humanoid_forward")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    # Load the saved policy
    print("Loading policy ...")
    params_path = "/tmp/humanoid_params.pkl"
    networks_path = "/tmp/humanoid_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy function
    print("Creating policy network...")
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=obs.shape[0],
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        check_sizes=True,
    )
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
    # test()
