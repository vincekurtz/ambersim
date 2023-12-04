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
    print_module_summary,
)
from ambersim.rl.humanoid.forward import HumanoidForwardEnv
from ambersim.rl.helpers import BraxPPONetworksWrapper

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
    policy_network = MLP(layer_sizes=[256, 256, 2*env.action_size])  # Policy outputs mean and std dev
    value_network = MLP(layer_sizes=[256, 256, 1])

    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
    )
    #print_module_summary(network_wrapper.policy_network, env.observation_size)

    # Create the PPO agent
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=10,   # 30 M
        num_evals=5, 
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
        seed=0,
    )

    # Set up tensorboard logging
    logdir = f"/tmp/mjx_brax_logs/humanoid_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Setting up tensorboard at {logdir}...")






if __name__=="__main__":
    train()