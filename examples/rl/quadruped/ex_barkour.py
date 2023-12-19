import functools
import pickle
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
from brax.training.distribution import NormalTanhDistribution
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP, LiftedInputLinearSystemPolicy, LinearSystemPolicy
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
    ny = 31
    nu = 12
    nz = 16

    # Initialize the environment
    envs.register_environment("barkour", lambda *args: RecurrentWrapper(BarkourEnv(*args), nz=nz))
    # envs.register_environment("barkour", BarkourEnv)
    env = envs.get_environment("barkour")

    # Create policy and value networks
    policy_network = LinearSystemPolicy(nz=nz, ny=ny, nu=nu)
    # policy_network = LiftedInputLinearSystemPolicy(nz=nz, ny=ny, nu=nu, phi_kwargs={"layer_sizes": [16, 16, nz]})
    # policy_network = MLP(layer_sizes=[128, 128, 2 * (nz + nu)])
    value_network = MLP(layer_sizes=(256, 256, 1))

    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalTanhDistribution,
    )

    num_timesteps = 60_000_000
    eval_every = 1_000_000

    # Domain randomization function
    def domain_randomize(sys, rng):
        """Randomize over friction and actuator gains."""

        @jax.vmap
        def rand(rng):
            _, key = jax.random.split(rng, 2)
            # friction
            friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
            friction = sys.geom_friction.at[:, 0].set(friction)
            # actuator
            _, key = jax.random.split(key, 2)
            gain_range = (-10, -5)
            param = (
                jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
            )
            gain = sys.actuator_gainprm.at[:, 0].set(param)
            bias = sys.actuator_biasprm.at[:, 1].set(-param)
            return friction, gain, bias

        friction, gain, bias = rand(rng)

        in_axes = jax.tree_map(lambda x: None, sys)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "actuator_gainprm": 0,
                "actuator_biasprm": 0,
            }
        )

        sys = sys.tree_replace(
            {
                "geom_friction": friction,
                "actuator_gainprm": gain,
                "actuator_biasprm": bias,
            }
        )

        return sys, in_axes

    # Define the training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_timesteps // eval_every,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        gae_lambda=0.95,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-4,
        num_envs=8192,
        batch_size=1024,
        network_factory=network_wrapper.make_ppo_networks,
        clipping_epsilon=0.3,
        num_resets_per_eval=10,
        # randomization_fn=domain_randomize,
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


def test():
    """Load a trained policy and run a little sim with it."""
    # Create an environment for evaluation
    print("Creating test environment...")
    nz = 0
    envs.register_environment("barkour", lambda *args: RecurrentWrapper(BarkourEnv(*args), nz=nz))
    env = envs.get_environment("barkour")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)

    # Set the command and initial state
    mj_data.qpos = mj_model.keyframe("standing").qpos
    info = {"z": jnp.zeros(nz), "obs_history": jnp.zeros(15), "command": jnp.array([0.0, 0.0, 0.0])}  # vx, vy, yaw rate
    obs = env.compute_obs(mjx.device_put(mj_data), info)

    # Load the saved policy
    print("Loading policy ...")
    params_path = "/tmp/barkour_params.pkl"
    networks_path = "/tmp/barkour_networks.pkl"
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

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = act[nz:]
            obs = env.compute_obs(mjx.device_put(mj_data), info)
            info["z"] = act[:nz]

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
