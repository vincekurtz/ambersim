import functools
from datetime import datetime
from typing import Sequence, Tuple

import flax
import jax
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax import envs
from brax.base import Motion, Transform
from brax.envs.base import Env, State
from brax.io import model
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import linen
from jax import numpy as jp
from mujoco import mjx
from torch.utils.tensorboard import SummaryWriter

from ambersim.reinforcement_learning.policies import MLP
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
        upright_angle_cost: float = 1.0,
        center_cart_cost: float = 0.0,
        velocity_cost: float = 0.0,
        control_cost: float = 0.0,
        termination_threshold: float = 0.2,
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

        self._upright_angle_cost = upright_angle_cost
        self._center_cart_cost = center_cart_cost
        self._velocity_cost = velocity_cost
        self._control_cost = control_cost

        # Stop the episode if the pole falls over too far
        self._termination_threshold = termination_threshold

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to a new initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low = -(self._termination_threshold - 0.05)
        hi = self._termination_threshold - 0.05
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward": zero,
            "upright_reward": zero,
            "center_cart_reward": zero,
            "velocity_reward": zero,
            "control_reward": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Steps the environment forward one timestep."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        obs = self._get_obs(data, action)

        # Compute the reward
        upright_reward = 1.0
        # upright_reward = - self._upright_angle_cost * obs[1]**2
        center_cart_reward = -self._center_cart_cost * obs[0] ** 2
        velocity_reward = -self._velocity_cost * (obs[2] ** 2 + obs[3] ** 2)
        control_reward = -self._control_cost * action[0] ** 2
        reward = upright_reward + center_cart_reward + velocity_reward + control_reward

        # Check if the episode is done
        done = jp.where(jp.abs(obs[1]) > self._termination_threshold, 1.0, 0.0)

        state.metrics.update(
            reward=reward,
            upright_reward=upright_reward,
            center_cart_reward=center_cart_reward,
            velocity_reward=velocity_reward,
            control_reward=control_reward,
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
    mj_data.qpos[1] = start_angle

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


@flax.struct.dataclass
class CustomPPONetworks:
    """Custom PPO networks."""

    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


class SequentialMLP(linen.Module):
    """Your standard multi-layer perceptron."""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = linen.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jp.ndarray):
        """Run the thing."""
        output_size = self.layer_sizes[-1]
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                hidden
            )
            hidden = self.activation(hidden)
            hidden = linen.Dense(output_size, name=f"output_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                hidden
            )
        return hidden


class ParallelMLP(linen.Module):
    """A multi-layer perceptron where all layers are flattened into one."""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = linen.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jp.ndarray):
        """Run the thing."""
        output_size = self.layer_sizes[-1]

        # Evaluate each layer, mapping from the input to the output size
        hidden_outputs = []
        for i, hidden_size in enumerate(self.layer_sizes[:-1]):
            hidden = linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                data
            )
            hidden = self.activation(hidden)
            hidden = linen.Dense(output_size, name=f"output_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                hidden
            )
            hidden_outputs.append(hidden)

        # Total output is the summ of all the hidden outputs
        return sum(hidden_outputs)


class HierarchicalMLP(linen.Module):
    """MLP structure based on classical hierarchical control architectures."""

    layer_sizes: Sequence[int]
    activation: networks.ActivationFn = linen.relu
    kernel_init: networks.Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, x: jp.ndarray):
        """Run the thing."""
        output_size = self.layer_sizes[-1]

        # First layer maps input to output, [x, y]
        y = linen.Dense(self.layer_sizes[0], name="hidden_0", kernel_init=self.kernel_init, use_bias=self.bias)(x)
        y = self.activation(y)
        y = linen.Dense(output_size, name="output_0", kernel_init=self.kernel_init, use_bias=self.bias)(y)

        # Subsequent layers map [x, y] --> y
        for i, hidden_size in enumerate(self.layer_sizes[1:-1]):
            y = jp.concatenate([x, y], axis=-1)
            y = linen.Dense(hidden_size, name=f"hidden_{i + 1}", kernel_init=self.kernel_init, use_bias=self.bias)(y)
            y = self.activation(y)
            y = linen.Dense(output_size, name=f"output_{i + 1}", kernel_init=self.kernel_init, use_bias=self.bias)(y)
        return y


def make_custom_policy_network(
    output_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [output_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        """Apply the policy network."""
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_custom_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
) -> CustomPPONetworks:
    """Make some PPO networks (value network and policy network) with a custom architecture."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = make_custom_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    return CustomPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def train():
    """Train a policy to swing up the cart-pole, then save the trained policy."""
    print("Creating cart-pole environment...")
    envs.register_environment("cart_pole", CartPole)
    env = envs.get_environment("cart_pole")

    # Use a custom network architecture
    print("Creating policy network...")
    network_factory = functools.partial(
        make_custom_ppo_networks,
        policy_hidden_layer_sizes=(512,) * 1,
        value_hidden_layer_sizes=(256,) * 3,
    )

    # Create the PPO agent
    print("Creating PPO agent...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100_000,
        num_evals=20,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=256,
        batch_size=128,
        network_factory=network_factory,
        seed=3,
    )

    # Set up tensorboard logging
    print("Setting up tensorboard...")
    writer = SummaryWriter("/tmp/mjx_brax_logs/cart_pole")

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
    make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

    print(f"  time to jit: {times[1] - times[0]}")
    print(f"  time to train: {times[-1] - times[1]}")

    # Save the trained policy
    print("Saving trained policy...")
    model_path = "/tmp/mjx_brax_policy"
    model.save_params(model_path, params)


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
    mj_data.qpos[1] = start_angle

    # Run a little sim
    print("Running sim...")
    num_steps = 200
    render_every = 2
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
    fps = 1.0 / (env.dt * render_every)
    media.write_video("cart_pole_controlled.mp4", frames, fps=fps)


if __name__ == "__main__":
    # visualize_open_loop(0.0)
    train()
    # test(0.2)
