import functools
import pickle
from datetime import datetime

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.distribution import NormalTanhDistribution
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP, LinearSystemPolicy
from ambersim.learning.distributions import NormalDistribution
from ambersim.rl.env_wrappers import RecurrentWrapper
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.rl.quadruped.barkour import BarkourEnv
from ambersim.utils.io_utils import load_mj_model_from_file

"""
Run the quadruped "barkour" example from the MJX tutorials:

https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
"""


def train():
    """Train a quadruped barkour agent."""
    # Observation, action, and lifted state sizes for the controller system
    ny = 15
    nu = 12
    nz = 30

    # Initialize the environment
    envs.register_environment("barkour", lambda *args: RecurrentWrapper(BarkourEnv(*args), nz=nz))
    env = envs.get_environment("barkour")

    # Create policy and value networks
    policy_network = LinearSystemPolicy(nz=nz, ny=ny, nu=nu)
    # policy_network = MLP(layer_sizes=(128, 128, 2 * env.action_size))
    value_network = MLP(layer_sizes=(256, 256, 1))

    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalTanhDistribution,
    )

    num_timesteps = 60_000_000
    eval_every = 1_000_000

    # Define the training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=60_000_000,
        num_evals=num_timesteps // eval_every,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=8,
        gae_lambda=0.95,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-5,
        num_envs=4096,
        batch_size=1024,
        network_factory=network_wrapper.make_ppo_networks,
        clipping_epsilon=0.2,
        num_resets_per_eval=10,
        seed=0,
    )

    # Define a callback to log progress
    log_dir = f"/tmp/mjx_brax_logs/barkour_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Logging to {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

        # Log to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)
            writer.add_scalar(key, val, num_steps)

    # Do the training
    print("Training...")
    _, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save both the parameters and the networks to disk
    print("Saving...")
    params_path = "/tmp/barkour_params.pkl"
    networks_path = "/tmp/barkour_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


if __name__ == "__main__":
    train()
