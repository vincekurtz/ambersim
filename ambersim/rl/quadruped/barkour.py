from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import math
from brax.base import Base, Motion, Transform
from etils import epath
from flax import struct
from mujoco import mjx

from ambersim.rl.base import MjxEnv, State
from ambersim.utils.io_utils import load_mj_model_from_file


@struct.dataclass
class BarkourConfig:
    """Config dataclass for a quadruped locomotion task."""

    # Location of the model file
    model_path = Path(epath.resource_path("mujoco")) / ("mjx/benchmark/model/barkour_v0/assets/barkour_v0_mjx.xml")

    # Number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 10

    # Observation noise level
    obs_noise = 0.05

    # Scaling factor for actions (q_target = q_stand + action * action_scale)
    action_scale = 0.3

    # Time steps to take before terminating the episode
    reset_horizon = 500

    # Number of observations to stack through time
    obs_hist_len = 1  # 15

    # *********** Tracking Parameters ***********

    # Tracking reward = exp(-error^2/sigma).
    tracking_sigma = 0.25  # 0.25

    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel = 1.5  # 1.5

    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel = 0.8  # 0.8

    # ********* Other Reward Parameters *********

    # Penalize the base velocity in z direction, L2 penalty.
    lin_vel_z = -0.0  # -2.0

    # Penalize the base roll and pitch rate. L2 penalty.
    ang_vel_xy = -0.0  # -0.05

    # Penalize non-zero roll and pitch angles. L2 penalty.
    orientation = -1.0  # -5.0

    # Penalize height of the base ||base_height - default_base_height||^2.
    base_height = -1.0  # 0.0
    default_base_height = 0.21

    # L2 regularization of joint torques, |tau|^2.
    torques = -0.0  # -0.002

    # Penalize the change in the action and encourage smooth
    # actions. L2 regularization |action - last_action|^2
    action_rate = -0.1  # -0.1

    # Encourage long swing steps.  However, it does not
    # encourage high clearances.
    feet_air_time = 0.0  # 0.2

    # Encourage no motion at zero command, L2 regularization
    # |q - q_default|^2.
    stand_still = -0.0  # -0.5

    # Early termination penalty (for falling down)
    termination = -10.0  # -1.0

    # Penalizing foot slipping on the ground.
    foot_slip = -0.0  # -0.1


class BarkourEnv(MjxEnv):
    """Environment for training a quadruped to walk over flat ground.

    Adopted from the MJX tutorial: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
    """

    def __init__(self, config: Optional[BarkourConfig] = None) -> None:
        """Initialize the barkour env. See parent docstring."""
        if config is None:
            config = BarkourConfig()
        self.config = config

        # Load the model
        mj_model = load_mj_model_from_file(config.model_path)

        # Set solver parameters
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 6

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

        # Define some useful model quantities
        self.torso_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso")
        self._feet_index = jnp.array([3, 6, 9, 12])
        self._feet_pos = jnp.array(
            [
                [-0.191284, -0.0191638, 0.013],
                [-0.191284, -0.0191638, -0.013],
                [-0.191284, -0.0191638, 0.013],
                [-0.191284, -0.0191638, -0.013],
            ]
        )
        self._init_q = mj_model.keyframe("standing").qpos
        self._default_ap_pose = mj_model.keyframe("standing").qpos[7:]
        self.lowers = self._default_ap_pose - jnp.array([0.2, 0.8, 0.8] * 4)
        self.uppers = self._default_ap_pose + jnp.array([0.2, 0.8, 0.8] * 4)
        self._foot_radius = 0.014

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Generate a random user command (x velocity, y velocity, yaw rate)."""
        lin_vel_x = [-0.6, 1.0]  # min max [m/s]
        # lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        lin_vel_y = [ 0.0, 0.0]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
        new_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        """See parent docstring."""
        rng, key = jax.random.split(rng)

        qpos = jnp.array(self._init_q)
        qvel = jnp.zeros(self.model.nv)
        new_cmd = self.sample_command(key)
        data = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "last_act": jnp.zeros(12),
            "last_vel": jnp.zeros(12),
            "last_contact_buffer": jnp.zeros((20, 4), dtype=bool),
            "command": new_cmd,
            "last_contact": jnp.zeros(4, dtype=bool),
            "feet_air_time": jnp.zeros(4),
            "obs_history": jnp.zeros(31 * self.config.obs_hist_len),
            "reward_tuple": {
                "tracking_lin_vel": 0.0,
                "tracking_ang_vel": 0.0,
                "lin_vel_z": 0.0,
                "ang_vel_xy": 0.0,
                "orientation": 0.0,
                "torque": 0.0,
                "action_rate": 0.0,
                "stand_still": 0.0,
                "feet_air_time": 0.0,
                "foot_slip": 0.0,
                "base_height": 0.0,
            },
            "step": 0,
        }

        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state_info)
        reward, done = jnp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]
        state = State(data, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """See parent docstring."""
        rng, rng_noise, cmd_rng = jax.random.split(state.info["rng"], 3)

        # physics step
        cur_action = jnp.array(action)
        action = action[:12] * self.config.action_scale
        motor_targets = jnp.clip(action + self._default_ap_pose, self.lowers, self.uppers)
        data = self.pipeline_step(state.pipeline_state, motor_targets)

        # observation data
        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state.info)
        obs_noise = self.config.obs_noise * jax.random.uniform(rng_noise, obs.shape, minval=-1, maxval=1)
        qpos, qvel = data.qpos, data.qvel
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]

        # foot contact data based on z-position
        foot_contact_pos = self._get_feet_pos_vel(x, xd)[0][:, 2] - self._foot_radius
        contact = foot_contact_pos < 1e-3  # a mm or less off the floor
        contact_filt_mm = jnp.logical_or(contact, state.info["last_contact"])
        contact_filt_cm = jnp.logical_or(
            foot_contact_pos < 3e-2, state.info["last_contact"]
        )  # 3cm or less off the floor
        first_contact = (state.info["feet_air_time"] > 0) * (contact_filt_mm)
        state.info["feet_air_time"] += self.dt

        # reward
        reward_tuple = {
            "tracking_lin_vel": (
                self.dt * self._reward_tracking_lin_vel(state.info["command"], x, xd) * self.config.tracking_lin_vel
            ),
            "tracking_ang_vel": (
                self.dt * self._reward_tracking_ang_vel(state.info["command"], x, xd) * self.config.tracking_ang_vel
            ),
            "lin_vel_z": (self.dt * self._reward_lin_vel_z(xd) * self.config.lin_vel_z),
            "ang_vel_xy": (self.dt * self._reward_ang_vel_xy(xd) * self.config.ang_vel_xy),
            "orientation": (self.dt * self._reward_orientation(x) * self.config.orientation),
            "torque": (self.dt * self._reward_torques(data.qfrc_actuator) * self.config.torques),
            "action_rate": (
                self.dt * self._reward_action_rate(cur_action, state.info["last_act"]) * self.config.action_rate
            ),
            "stand_still": (
                self.dt
                * self._reward_stand_still(state.info["command"], joint_angles, self._default_ap_pose)
                * self.config.stand_still
            ),
            "feet_air_time": (
                self.dt
                * self._reward_feet_air_time(
                    state.info["feet_air_time"],
                    first_contact,
                    state.info["command"],
                )
                * self.config.feet_air_time
            ),
            "foot_slip": (self.dt * self._reward_foot_slip(x, xd, contact_filt_cm) * self.config.foot_slip),
            "base_height": (self.dt * self._reward_base_height(x) * self.config.base_height),
        }
        reward = sum(reward_tuple.values())
        # reward = jnp.clip(reward * self.dt, 0.0, 10000.0)

        # state management
        state.info["last_act"] = cur_action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["last_contact_buffer"] = jnp.roll(state.info["last_contact_buffer"], 1, axis=0)
        state.info["last_contact_buffer"] = state.info["last_contact_buffer"].at[0].set(contact)
        state.info["reward_tuple"] = reward_tuple
        state.info["step"] += 1
        state.info.update(rng=rng)

        # resetting logic if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[0]), up) < 0
        done |= jnp.any(joint_angles < 0.98 * self.lowers)
        done |= jnp.any(joint_angles > 0.98 * self.uppers)
        done |= x.pos[0, 2] < 0.1  # 0.18

        # termination reward
        reward += done * (state.info["step"] < self.config.reset_horizon) * self.config.termination

        # when done, sample new command if more than reset_horizon timesteps
        # achieved
        state.info["command"] = jnp.where(
            done & (state.info["step"] > self.config.reset_horizon), self.sample_command(cmd_rng), state.info["command"]
        )
        # reset the step counter when done
        state.info["step"] = jnp.where(done | (state.info["step"] > self.config.reset_horizon), 0, state.info["step"])

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self.torso_idx])[1]
        for k in state.info["reward_tuple"].keys():
            state.metrics[k] = state.info["reward_tuple"][k]

        state = state.replace(pipeline_state=data, obs=obs + obs_noise, reward=reward, done=done * 1.0)
        return state

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the observation from the state. See parent docstring."""
        x, xd = self._pos_vel(data)
        return self._get_obs(data.qpos, x, xd, info)

    def _get_obs(self, qpos: jax.Array, x: Transform, xd: Motion, state_info: Dict[str, Any]) -> jax.Array:
        # Get observations:
        # yaw_rate,  projected_gravity, command,  motor_angles, last_action

        inv_base_orientation = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)
        cmd = state_info["command"]

        obs_list = []
        # yaw rate
        obs_list.append(jnp.array([local_rpyrate[2]]) * 0.25)
        # projected gravity
        obs_list.append(math.rotate(jnp.array([0.0, 0.0, -1.0]), inv_base_orientation))
        # command
        obs_list.append(cmd * jnp.array([2.0, 2.0, 0.25]))
        # motor angles
        angles = qpos[7:19]
        obs_list.append(angles - self._default_ap_pose)
        # last action
        obs_list.append(state_info["last_act"])

        obs = jnp.clip(jnp.concatenate(obs_list), -100.0, 100.0)

        # stack observations through time
        single_obs_size = len(obs)
        state_info["obs_history"] = jnp.roll(state_info["obs_history"], single_obs_size)
        state_info["obs_history"] = jnp.array(state_info["obs_history"]).at[:single_obs_size].set(obs)
        return state_info["obs_history"]

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(act - last_act))

    def _reward_tracking_lin_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.config.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.config.tracking_sigma)

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= math.normalize(commands[:2])[1] > 0.05  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array, default_angles: jax.Array) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - default_angles)) * (math.normalize(commands[:2])[1] < 0.1)

    def _get_feet_pos_vel(self, x: Transform, xd: Motion) -> Tuple[jax.Array, jax.Array]:
        offset = Transform.create(pos=self._feet_pos)
        pos = x.take(self._feet_index).vmap().do(offset).pos
        world_offset = Transform.create(pos=pos - x.take(self._feet_index).pos)
        vel = world_offset.vmap().do(xd.take(self._feet_index)).vel
        return pos, vel

    def _reward_foot_slip(self, x: Transform, xd: Motion, contact_filt: jax.Array) -> jax.Array:
        # Get feet velocities
        _, foot_world_vel = self._get_feet_pos_vel(x, xd)
        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_world_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_base_height(self, x: Transform) -> jax.Array:
        # Penalize height of the base
        return jnp.square(x.pos[0, 2] - self.config.default_base_height)

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd
