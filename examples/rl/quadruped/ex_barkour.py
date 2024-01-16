import functools
import pickle
import sys
import time
from datetime import datetime

import flax.linen as nn
import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import mujoco.viewer
import pygame
from brax import envs, math
from brax.base import Motion, Transform
from brax.io import model
from brax.mjx.base import State
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
    # Lifted state size and control dimension
    nz = 32
    nu = 12

    # Initialize the environment
    envs.register_environment("barkour", RecurrentWrapper.env_factory(BarkourEnv, nz=nz))

    # Create policy and value networks
    # policy_network = MLP(layer_sizes=(128,) * 4 + (2 * (nu + nz),))
    policy_network = MLP(layer_sizes=(2 * (nu + nz),))  # linear
    value_network = MLP(layer_sizes=(256,) * 5 + (1,))

    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=NormalDistribution,
    )

    # Domain randomization function
    def domain_randomize(sys, rng):
        """Randomize over friction and actuator gains."""
        friction_range = (0.6, 1.4)
        gain_range = (-5, 5)

        @jax.vmap
        def rand(rng):
            _, key = jax.random.split(rng, 2)
            # friction
            friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
            friction = sys.geom_friction.at[:, 0].set(friction)

            # actuator gains
            _, key = jax.random.split(key, 2)
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

    num_timesteps = 1_000_000_000
    eval_every = 10_000_000

    # Define the training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_timesteps // eval_every,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        network_factory=network_wrapper.make_ppo_networks,
        randomization_fn=domain_randomize,
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

    # Do the training. Note that for some reason it seems essential to have a separate
    # evaluation environment when using domain randomization.
    env = envs.get_environment("barkour")
    eval_env = envs.get_environment("barkour")

    print("Training...")
    _, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
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
    nz = 32
    envs.register_environment("barkour", RecurrentWrapper.env_factory(BarkourEnv, nz=nz))
    env = envs.get_environment("barkour")

    # Load the saved policy
    print("Loading policy ...")
    params_path = "/tmp/barkour_params.pkl"
    networks_path = "/tmp/barkour_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create a mujoco model
    # N.B. this is different from the scene_mjx.xml model used for training
    mj_model = load_mj_model_from_file("models/barkour/scene_terrain.xml")
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos = mj_model.keyframe("home").qpos

    # Set model parameters to roughly match the MJX model
    mj_model.opt.timestep = 0.004
    mj_model.dof_damping[6:] = 0.5
    mj_model.actuator_gainprm[:, 0] = 35.0
    mj_model.actuator_biasprm[:, 1] = -35.0

    # modify friction
    mj_model.geom_friction[:, 0] = 1.5

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

    # Define an observation function
    def get_obs(mj_data, command, z):
        """Returns the observation vector."""
        q_legs = jnp.array(mj_data.qpos[7:19] - env.env._default_pose)
        v_legs = jnp.array(mj_data.qvel[6:18])
        c = jnp.cos(q_legs)
        s = jnp.sin(q_legs)

        # Yaw rate and projected gravity
        x = Transform(pos=mj_data.xpos[1:], rot=mj_data.xquat[1:])
        cvel = Motion(vel=mj_data.cvel[1:, 3:], ang=mj_data.cvel[1:, :3])
        offset = mj_data.xpos[1:, :] - mj_data.subtree_com[mj_model.body_rootid[1:]]
        offset = Transform.create(pos=offset)
        xd = offset.vmap().do(cvel)
        inv_torso_rot = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_torso_rot)
        yaw_rate = jnp.array([local_rpyrate[2]])
        projected_gravity = math.rotate(jnp.array([0, 0, -1]), inv_torso_rot)

        return jnp.concatenate(
            [
                z,
                yaw_rate,
                projected_gravity,
                command,
                v_legs,
                c,
                s,
            ]
        )

    # Standard deviation of the observation noise
    obs_noise_std = jnp.concatenate(
        [
            jnp.zeros(nz),  # controller's lifted state is noise-free
            0.1 * jnp.ones(1),  # yaw rate is noisy
            0.1 * jnp.ones(3),  # projected gravity is noisy
            jnp.zeros(3),  # command is noise-free
            0.05 * jnp.ones(12),  # leg velocities are noisy
            0.001 * jnp.ones(12),  # cosines have small noise
            0.001 * jnp.ones(12),  # sines have small noise
        ]
    )

    # Define an initial command, controller state, and prior action
    z = jnp.zeros(nz)  # (lifted) state of the controller system
    command = jnp.array([0.0, 0.0, 0.0])  # commanded velocity

    # Set up a joystick to control the sim
    print("Looking for joystick...")
    pygame.init()
    has_joystick = False
    joystick = None

    if pygame.joystick.get_count() > 0 and pygame.joystick.Joystick(0).get_numaxes() >= 4:
        has_joystick = True
        joystick = pygame.joystick.Joystick(0)
        print("Connected to joystick", joystick.get_name())
    else:
        # If there's no joystick, just set the forward command to do
        # something kind of interesting
        print("No joystick detected: setting forward command to 1 m/s.")
        command = jnp.array([1.0, 0.3, -0.5])

    min_cmd = jnp.array([-0.6, -0.8, -0.7])
    max_cmd = jnp.array([1.0, 0.8, 0.7])

    # Run the sim
    print("Running...")
    rng = jax.random.PRNGKey(0)
    default_pose = mj_model.keyframe("home").qpos[7:19]
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            obs_rng, act_rng, rng = jax.random.split(rng, 3)

            # Update the command based on the joystick, if available
            if has_joystick:
                pygame.event.pump()
                yaw = -joystick.get_axis(0)
                sideways = -joystick.get_axis(2)
                forward = -joystick.get_axis(3)
                command = jnp.array([forward, sideways, yaw])
                command = jnp.clip(command, min_cmd, max_cmd)

            # Get an observation
            obs = get_obs(mj_data, command, z)
            obs += jax.random.normal(obs_rng, obs.shape) * obs_noise_std

            # Take an action
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = jnp.clip(default_pose + 0.3 * act[nz:], env.env.lowers, env.env.uppers)
            z = act[:nz]

            # Step the simulation
            for _ in range(env.env._n_frames):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly real time
            elapsed = time.time() - step_start
            dt = float(env.dt)
            if elapsed < dt:
                time.sleep(dt - elapsed)


def make_video():
    """Make a video of the trained policy."""
    # Create an environment for evaluation
    nz = 32
    envs.register_environment("barkour", RecurrentWrapper.env_factory(BarkourEnv, nz=nz))
    env = envs.get_environment("barkour")

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

    # Create step and reset functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    print("Initializing...")
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = jnp.array([1.0, 0.1, 0.2])
    rollout = [state.pipeline_state]

    # Simulate a trajectory
    print("Simulating...")
    n_steps = 500
    render_every = 2
    for _ in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, act)
        rollout.append(state.pipeline_state)

    fname = "/tmp/barkour.mp4"
    print(f"Writing to {fname}...")
    media.write_video(fname, env.render(rollout[::render_every], camera="track"), fps=1.0 / env.dt / render_every)


if __name__ == "__main__":
    usage_str = "Usage: python ex_barkour.py [train|test|video]"

    if len(sys.argv) < 2:
        print(usage_str)
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "video":
        make_video()
    else:
        print(usage_str)
        sys.exit(1)
