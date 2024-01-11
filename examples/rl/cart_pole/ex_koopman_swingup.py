import functools
import pickle
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import make_inference_fn
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP, BilinearSystemPolicy, LiftedInputLinearSystemPolicy, LinearSystemPolicy
from ambersim.learning.distributions import NormalDistribution
from ambersim.rl.cart_pole.swingup import CartPoleSwingupEnv
from ambersim.rl.env_wrappers import RecurrentWrapper
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.utils.io_utils import load_mj_model_from_file

"""
Perform cart-pole swingup with a Koopman linear system policy.
"""


def train():
    """Train a policy to swing up the cart-pole, then save the trained policy."""
    # Choose the dimension of the lifted state for the controller system
    nz = 32

    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", RecurrentWrapper.env_factory(CartPoleSwingupEnv, nz=nz))
    env = envs.get_environment("cart_pole")

    # Create the policy and value networks
    print("Creating policy network...")
    ny = 8  # observations are degree-2 polynomials of [cart_pos, cos(theta), sin(theta), cart_vel, theta_dot]
    nu = 1  # control input is [cart_force]
    # policy_network = MLP(layer_sizes=[128, 128, 2 * (nz + nu)])
    # policy_network = MLP(layer_sizes=[2 * (nz + nu)], bias=True)
    policy_network = LinearSystemPolicy(nz=nz, ny=ny, nu=nu)
    # policy_network = BilinearSystemPolicy(nz=nz, ny=ny, nu=nu)
    # policy_network = LiftedInputLinearSystemPolicy(nz=nz, ny=ny, nu=nu, phi_kwargs={"layer_sizes": [16, 16, nz]})

    value_network = MLP(layer_sizes=[128, 128, 1])
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalDistribution,
    )

    # Set the number of training steps and evaluations
    num_timesteps = 30_000_000
    eval_every = 1_000_000

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_timesteps // eval_every,
        episode_length=200,
        reward_scaling=0.1,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-5,
        num_envs=8192,
        batch_size=256,
        clipping_epsilon=0.3,
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
    _, params, _ = train_fn(environment=env, progress_fn=progress)

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
    nz = 32
    envs.register_environment("cart_pole", RecurrentWrapper.env_factory(CartPoleSwingupEnv, nz=nz))
    env = envs.get_environment("cart_pole")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data.qpos[1] = start_angle
    info = {"z": jnp.zeros(nz)}
    obs = env.compute_obs(mjx.device_put(mj_data), info)

    # Load the saved policy
    print("Loading policy ...")
    params_path = "/tmp/cart_pole_params.pkl"
    networks_path = "/tmp/cart_pole_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy function
    print("Creating policy network...")
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
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

            print("|z|: ", jnp.linalg.norm(info["z"]))

            # Mask some of the observations
            mask = jnp.array(
                [
                    1,  # cart_pos
                    1,  # cos(theta)
                    1,  # sin(theta)
                    1,  # cart_vel
                    1,  # theta_dot
                    1,  # cos(theta) * sin(theta)
                    1,  # cos(theta) * cart_vel
                    1,  # cos(theta) * theta_dot
                ]
            )
            mask = jnp.concatenate([jnp.ones(nz), mask])
            obs = obs * mask

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            info["z"] = act[:nz]
            mj_data.ctrl[:] = act[nz:]
            obs = env.compute_obs(mjx.device_put(mj_data), info)

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
    usage_message = "Usage: python ex_koopman_swingup.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        print(usage_message)
        sys.exit(1)
