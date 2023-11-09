import functools
from datetime import datetime
from typing import Tuple

import jax
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax import envs
from brax.base import Motion, Transform
from brax.envs.base import Env, State
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from jax import numpy as jp
from mujoco import mjx

from ambersim.utils.io_utils import load_mj_model_from_file

"""
This example demonstrates using brax PPO + MJX to train a policy for a simple
cart-pole.
"""


class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ):
        """Initializes MjxEnv.

        Args:
          mj_model: mujoco.MjModel
          physics_steps_per_control_step: the number of times to step the physics
            pipeline for each environment step
        """
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return (
                mjx.step(self.sys, data),
                None,
            )

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation vector."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        """Returns the size of the action vector."""
        return self.sys.nu

    @property
    def backend(self) -> str:
        """Returns the backend used by this environment."""
        return "mjx"

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd


class CartPole(MjxEnv):
    """Training environment for a simple cart-pole."""

    def __init__(
        self,
        **kwargs,
    ):
        """Initializes the cart-pole environment."""
        path = "models/cart_pole/cart_pole.xml"
        mj_model = load_mj_model_from_file(path)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

        physics_steps_per_control_step = 1
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )

        super().__init__(mj_model=mj_model, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Steps the environment forward one timestep."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data, action)

        reward = 1.0
        done = jp.where(jp.abs(obs[1]) > 0.2, 1.0, 0.0)

        state.metrics.update(
            reward=reward,
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Return the observation, [cos(theta), sin(theta), theta_dot]."""
        return jp.concatenate([data.qpos, data.qvel])


def visualize_open_loop(start_angle=0.0):
    """Save a video of an open-loop trajectory, using just mujoco."""
    mj_model = load_mj_model_from_file("models/cart_pole/cart_pole.xml")
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)

    # Set the initial state
    mj_data.qpos[0] = start_angle

    # Simulate a trajectory
    num_steps = 500
    render_every = 2
    frames = []

    for k in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        renderer.update_scene(mj_data)

        if k % render_every == 0:
            frame = renderer.render()
            frames.append(frame)

    fps = 1.0 / (mj_model.opt.timestep * render_every)
    media.write_video("cart_pole_open_loop.mp4", frames, fps=fps)


def train():
    """Train a policy to swing up the cart-pole, then save the trained policy."""
    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", CartPole)
    env = envs.get_environment("cart_pole")

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100_000,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=128,
        batch_size=64,
        seed=0,
    )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Helper function for recording training progress."""
        print(
            "    Step:",
            num_steps,
            "Reward:",
            metrics["eval/episode_reward"],
            "Std:",
            metrics["eval/episode_reward_std"],
        )

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)

    print("Training...")
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

    print(f"  time to jit: {times[1] - times[0]}")
    print(f"  time to train: {times[-1] - times[1]}")

    print("  Metrics: ", metrics)

    # Save the trained policy
    print("Saving trained policy...")
    model_path = "/tmp/mjx_brax_policy"
    model.save_params(model_path, params)

    # Show the learning curves
    plt.show()


def test(start_angle=0.0):
    """Load a trained policy and run a little sim with it."""
    # Create an environment for evaluation
    print("Creating test environment...")
    envs.register_environment("cart_pole", CartPole)
    env = envs.get_environment("cart_pole")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)
    rng = jax.random.PRNGKey(0)
    ctrl = jp.zeros(mj_model.nu)

    # Load the saved policy weights
    print("Loading policy weights...")
    params = model.load_params("/tmp/mjx_brax_policy")

    # Create the policy network
    print("Creating policy network...")
    # TODO(vincekurtz): figure out how to save make_inference_fn when we train
    network_factory = ppo_networks.make_ppo_networks
    normalize = running_statistics.normalize
    ppo_network = network_factory(env.observation_size, env.action_size, preprocess_observations_fn=normalize)
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    policy = make_inference_fn(params)
    jit_policy = jax.jit(policy)

    # Set the initial state
    mj_data.qpos[0] = start_angle
    mj_data.qvel[0] = 0.0

    # Run a little sim
    print("Running sim...")
    num_steps = 200
    fps = 60
    render_every = int(1.0 / (env.dt * fps))
    frames = []

    for k in range(num_steps):
        act_rng, rng = jax.random.split(rng)

        # Get the last observation
        obs = env._get_obs(mjx.device_put(mj_data), ctrl)

        # Compute the control action for the next step
        ctrl, _ = jit_policy(obs, act_rng)
        mj_data.ctrl = ctrl

        # Step the simulation
        for _ in range(env._physics_steps_per_control_step):
            mujoco.mj_step(mj_model, mj_data)

        if k % render_every == 0:
            # Add an image of the scene to the video
            renderer.update_scene(mj_data)
            frame = renderer.render()
            frames.append(frame)

    print("Saving video...")
    media.write_video("cart_pole_controlled.mp4", frames, fps=fps)


if __name__ == "__main__":
    visualize_open_loop(0.0)
    # train()
    # test()
