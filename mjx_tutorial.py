# @title Import packages for plotting and creating graphics
import functools
from datetime import datetime

import jax
import mediapy as media
import mujoco
import numpy as np
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training.agents.ppo import train as ppo
from jax import numpy as jp
from mujoco import mjx
from tensorboardX import SummaryWriter

from ambersim.learning.architectures import MLP
from ambersim.rl.base import State
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.rl.quadruped.barkour import BarkourEnv

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Domain randomization function


def domain_randomize(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-10, -5)
        param = jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
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


envs.register_environment("barkour", BarkourEnv)

env_name = "barkour"
env = envs.get_environment(env_name)

# re-instantiate the renderer
renderer = mujoco.Renderer(env.model)

"""## Train Policy

To train a policy with domain randomization, we pass in the domain randomization function into the brax train function; brax will call the domain randomization function when rolling out episodes. Training the quadruped takes about 14 minutes on a Tesla V100 GPU.
"""


policy_network = MLP(layer_sizes=(128, 128, 128, 128, 2 * 12))
value_network = MLP(layer_sizes=(256, 256, 256, 256, 256, 1))

network_wrapper = BraxPPONetworksWrapper(
    policy_network=policy_network,
    value_network=value_network,
    action_distribution=distribution.NormalTanhDistribution,
)

# make_networks_factory = functools.partial(
#    ppo_networks.make_ppo_networks,
#        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
    ppo.train,
    num_timesteps=6_000,
    num_evals=3,
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
    entropy_cost=1e-2,
    num_envs=8192,
    batch_size=1024,
    network_factory=network_wrapper.make_ppo_networks,
    num_resets_per_eval=10,
    randomization_fn=domain_randomize,
    seed=0,
)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 30, 0

# Define a pgrogress function
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


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, eval_env=eval_env)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Save and reload params.
print("Saving and loading params...")
model_path = "/tmp/mjx_brax_quadruped_policy"
model.save_params(model_path, params)
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

"""## Visualize Policy

For the Barkour Quadruped, the joystick commands can be set through `x_vel`, `y_vel`, and `ang_vel`. `x_vel` and `y_vel` define the linear forward and sideways velocities with respect to the quadruped torso. `ang_vel` defines the angular velocity of the torso in the z direction.
"""
print("Setting up visualizer...")
eval_env = envs.get_environment(env_name)


def get_image(state: State, camera: str) -> np.ndarray:
    """Renders the environment state."""
    d = mujoco.MjData(eval_env.model)
    # write the mjx.Data into an mjData object
    mjx.device_get_into(d, state.pipeline_state)
    mujoco.mj_forward(eval_env.model, d)
    # use the mjData object to update the renderer
    renderer.update_scene(d, camera=camera)
    return renderer.render()


jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# @markdown Commands **only used for Barkour Env**:
x_vel = 0.0  # @param {type: "number"}
y_vel = 0.0  # @param {type: "number"}
ang_vel = 0.0  # @param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info["command"] = the_command
rollout = [state]
images = [get_image(state, camera="track")]

# grab a trajectory
print("Running policy...")
n_steps = 500
render_every = 2

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if i % render_every == 0:
        images.append(get_image(state, camera="track"))

print("Rendering video...")
media.write_video("/tmp/mjx_brax_barkour.mp4", images, fps=1.0 / eval_env.dt / render_every)

print("Done!")
