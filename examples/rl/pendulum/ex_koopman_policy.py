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

from ambersim.learning.architectures import MLP, LinearSystemPolicy, BilinearSystemPolicy, LiftedInputLinearSystemPolicy
from ambersim.learning.distributions import NormalDistribution
from ambersim.rl.env_wrappers import RecurrentWrapper
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.rl.pendulum.swingup import PendulumSwingupEnv
from ambersim.utils.io_utils import load_mj_model_from_file

"""
Perform pendulum swingup training with a Koopman linear system policy.
"""


def train_swingup():
    """Train a pendulum swingup agent with custom network architectures."""
    # Choose the dimension of the lifted state for the controller system
    nz = 32

    # Initialize the environment with a recurrent wrapper
    envs.register_environment("pendulum_swingup", lambda *args: RecurrentWrapper(PendulumSwingupEnv(*args), nz=nz))
    env = envs.get_environment("pendulum_swingup")

    # Policy network takes as input [z, y]: the current lifted state and observations.
    # It outputs [z_next, u, σ(z_next), σ(u)]: the next lifted state, control input,
    # and their standard deviations.
    policy_network = LinearSystemPolicy(nz=nz, ny=3, nu=1)
    # policy_network = BilinearSystemPolicy(nz=nz, ny=3, nu=1)
    # policy_network = LiftedInputLinearSystemPolicy(nz=nz, ny=3, nu=1, phi_kwargs={"layer_sizes": (16, 16, nz)})
    # policy_network = MLP(layer_sizes=(32, 32, 2*(1+nz)))

    # Value network takes as input observations and the current lifted state,
    # and outputs a scalar value.
    value_network = MLP(layer_sizes=(128, 128, 1))
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalDistribution,
    )

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=250,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=False,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-4,
        num_envs=1024,
        batch_size=512,
        network_factory=network_wrapper.make_ppo_networks,
        seed=3,
    )

    # Define a callback to log progress
    log_dir = f"/tmp/mjx_brax_logs/koopman_pendulum_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"Setting up Tensorboard logging in {log_dir}")
    writer = SummaryWriter(log_dir)
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
    params_path = "/tmp/pendulum_params.pkl"
    networks_path = "/tmp/pendulum_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


def test_trained_swingup_policy():
    """Load a trained policy and run an interactive simulation."""
    # Load the trained policy
    print("Loading trained policy...")
    params_path = "/tmp/pendulum_params.pkl"
    params = model.load_params(params_path)

    # Create the policy, which is just a linear system
    A = np.asarray(params[1]["params"]["A"])
    B = np.asarray(params[1]["params"]["B"])
    C = np.asarray(params[1]["params"]["C"])
    D = np.asarray(params[1]["params"]["D"])
    nz = A.shape[0]
    z = np.zeros(nz)

    # Initialize the environment
    mj_model = load_mj_model_from_file("models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)
    dt = mj_model.opt.timestep

    print("Simulating...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            print("|z|: ", np.linalg.norm(z))

            # Get an observation
            theta = mj_data.qpos[0]
            theta_dot = mj_data.qvel[0]
            y = np.array([np.cos(theta), np.sin(theta), theta_dot])

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
