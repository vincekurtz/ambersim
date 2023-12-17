import functools
import pickle
import sys
import time
from datetime import datetime

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP, LinearSystemPolicy
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
    nz = 10

    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", lambda *args: RecurrentWrapper(CartPoleSwingupEnv(*args), nz=nz))
    env = envs.get_environment("cart_pole")

    # Create the policy and value networks
    print("Creating policy network...")
    ny = 5  # observations are [cart_pos, cos(theta), sin(theta), cart_vel, dtheta]
    nu = 1  # control input is [cart_force]
    # policy_network = MLP(layer_sizes=[128, 128, 2*(nz + nu)])
    policy_network = LinearSystemPolicy(nz=nz, ny=ny, nu=nu)

    value_network = MLP(layer_sizes=[256, 256, 1])
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalDistribution,
    )

    # Set the number of training steps and evaluations
    num_timesteps = 50_000_000
    eval_every = 100_000

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_timesteps // eval_every,
        episode_length=200,
        reward_scaling=0.1,
        normalize_observations=False,
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
    # Load the trained policy
    print("Loading trained policy...")
    params_path = "/tmp/cart_pole_params.pkl"
    params = model.load_params(params_path)

    # Create the policy, which is just a linear system
    A = np.asarray(params[1]["params"]["A"])
    B = np.asarray(params[1]["params"]["B"])
    C = np.asarray(params[1]["params"]["C"])
    D = np.asarray(params[1]["params"]["D"])
    nz = A.shape[0]
    z = np.zeros(nz)

    # Initialize the environment
    mj_model = load_mj_model_from_file("models/cart_pole/cart_pole.xml")
    mj_data = mujoco.MjData(mj_model)
    dt = mj_model.opt.timestep

    print("Simulating...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            print("|z|: ", np.linalg.norm(z))

            # Get an observation
            p = mj_data.qpos[0]
            p_dot = mj_data.qvel[0]
            theta = mj_data.qpos[1]
            theta_dot = mj_data.qvel[1]
            y = np.array([p, np.cos(theta), np.sin(theta), p_dot, theta_dot])

            # Apply the policy
            u = C @ z + D @ y
            z = A @ z + B @ y

            mj_data.ctrl[:] = u

            # Step the simulation (one physics step per control step here)
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
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
