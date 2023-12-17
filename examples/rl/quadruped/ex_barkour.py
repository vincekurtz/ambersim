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
    # Initialize the environment
    envs.register_environment("barkour", BarkourEnv)
    env = envs.get_environment("barkour")
    print(env)


if __name__ == "__main__":
    train()
