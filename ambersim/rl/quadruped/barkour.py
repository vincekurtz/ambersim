from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import MjxEnv, State
from brax.io import mjcf
from flax import struct
from mujoco import mjx

from ambersim import ROOT


@struct.dataclass
class BarkourConfig:
    """Config dataclass for a quadruped locomotion task."""

    # Location of the model file
    model_path = ROOT + "/models/barkour/scene_mjx.xml"

    # Observation noise level
    obs_noise = 0.05

    # Scaling factor for actions (q_target = q_stand + action * action_scale)
    action_scale = 0.3

    # Velocity of random disturbances
    kick_vel = 0.05

    # Frequency of random disturbances
    kick_interval = 10

    # Time steps to take before terminating the episode
    reset_horizon = 500

    # Number of observations to stack through time
    obs_hist_len = 1

    # *********** Tracking Parameters ***********

    # Tracking reward = exp(-error^2/sigma).
    tracking_sigma = 0.25

    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel = 1.5

    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel = 0.8

    # ********* Other Reward Parameters *********

    # Penalize the base velocity in z direction, L2 penalty.
    lin_vel_z = -2.0

    # Penalize the base roll and pitch rate. L2 penalty.
    ang_vel_xy = -0.05

    # Penalize non-zero roll and pitch angles. L2 penalty.
    orientation = -5.0

    # Penalize height of the base ||base_height - default_base_height||^2.
    base_height = -0.0  # 0.0
    default_base_height = 0.21

    # L2 regularization of joint torques, |tau|^2.
    torques = -0.0002  # -0.002

    # Penalize the change in the action and encourage smooth
    # actions. L2 regularization |action - last_action|^2
    action_rate = -0.01  # -0.1

    # Encourage long swing steps.  However, it does not
    # encourage high clearances.
    feet_air_time = 0.2  # 0.2

    # Encourage no motion at zero command, L2 regularization
    # |q - q_default|^2.
    stand_still = -0.5  # -0.5

    # Early termination penalty (for falling down)
    termination = -1.0  # -1.0

    # Penalizing foot slipping on the ground.
    foot_slip = -0.1  # -0.1


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
        self._dt = 0.02
        self.brax_sys = mjcf.load(config.model_path).replace(dt=self._dt)
        model = self.brax_sys.get_model()
        model.opt.timestep = 0.004

        # Override some params for a smoother policy
        model.dof_damping[6:] = 0.5239
        model.actuator_gainprm[:, 0] = 35.0
        model.actuator_biasprm[:, 1] = -35.0

        # Number of physics steps per control step
        n_frames = int(self._dt / model.opt.timestep)

        # Create the mjx model
        super().__init__(model, n_frames)

        # Define some useful model quantities
        self._torso_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "torso")
        self._init_q = jnp.array(model.keyframe("home").qpos)
        self._default_pose = jnp.array(model.keyframe("home").qpos[7:])
        self.lowers = jnp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jnp.array([0.52, 2.1, 2.1] * 4)

        feet_site = [
            "foot_front_left",
            "foot_hind_left",
            "foot_front_right",
            "foot_hind_right",
        ]
        feet_site_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            "lower_leg_front_left",
            "lower_leg_hind_left",
            "lower_leg_front_right",
            "lower_leg_hind_right",
        ]
        lower_leg_body_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, l) for l in lower_leg_body]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = model.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Generate a random user command (x velocity, y velocity, yaw rate)."""
        lin_vel_x = [-0.6, 1.0]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
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

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jnp.zeros(12),
            "last_vel": jnp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jnp.zeros(4, dtype=bool),
            "feet_air_time": jnp.zeros(4),
            "rewards": {
                "tracking_lin_vel": 0.0,
                "tracking_ang_vel": 0.0,
                "lin_vel_z": 0.0,
                "ang_vel_xy": 0.0,
                "orientation": 0.0,
                "torques": 0.0,
                "action_rate": 0.0,
                "stand_still": 0.0,
                "feet_air_time": 0.0,
                "foot_slip": 0.0,
                "base_height": 0.0,
                "termination": 0.0,
            },
            "kick": jnp.array([0.0, 0.0]),
            "step": 0,
        }

        obs_history = jnp.zeros(103 * self.config.obs_hist_len)
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jnp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """See parent docstring."""
        rng, cmd_rng, kick_noise = jax.random.split(state.info["rng"], 3)

        # random kick
        kick_theta = jax.random.uniform(kick_noise, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info["step"], self.config.kick_interval) == 0
        qvel = state.pipeline_state.data.qvel
        qvel = qvel.at[:2].set(kick * self.config.kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.data.qvel": qvel})

        # physics step
        motor_targets = self._default_pose + action * self.config.action_scale
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.data.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.lowers)
        done |= jnp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            "tracking_lin_vel": (self._reward_tracking_lin_vel(state.info["command"], x, xd)),
            "tracking_ang_vel": (self._reward_tracking_ang_vel(state.info["command"], x, xd)),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.data.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
            "base_height": self._reward_base_height(x),
        }
        reward = jnp.clip(sum(rewards.values()), 0.0, 10000.0)

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jnp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jnp.where(done | (state.info["step"] > 500), 0, state.info["step"])

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jnp.float32(done)
        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(self, pipeline_state: State, state_info: dict[str, Any], obs_history: jax.Array) -> jax.Array:
        # base rotation and angular velocity
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        # Leg positions and velocities
        q_legs = pipeline_state.q[7:] - self._default_pose
        v_legs = pipeline_state.qd[6:]
        c = jnp.cos(q_legs)
        s = jnp.sin(q_legs)

        # Put together the observation vector
        obs = jnp.concatenate(
            [
                jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"] * jnp.array([2.0, 2.0, 0.25]),  # command
                q_legs,  # joint angles
                v_legs,  # joint velocities
                c,  # cos of joint angles
                s,  # sin of joint angles
                c * s,  # cos of joint angles times sin of joint angles
                c * v_legs,  # cos of joint angles times joint velocities
                s * v_legs,  # sin of joint angles times joint velocities
                state_info["last_act"],  # last action
            ]
        )

        # clip, noise
        obs = jnp.clip(obs, -100.0, 100.0) + self.config.obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2]) * self.dt * self.config.lin_vel_z

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2])) * self.dt * self.config.ang_vel_xy

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2])) * self.dt * self.config.orientation

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return (jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))) * self.dt * self.config.torques

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(act - last_act)) * self.dt * self.config.action_rate

    def _reward_tracking_lin_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.config.tracking_sigma)
        return lin_vel_reward * self.dt * self.config.tracking_lin_vel

    def _reward_tracking_ang_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.config.tracking_sigma) * self.dt * self.config.tracking_ang_vel

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= math.normalize(commands[:2])[1] > 0.05  # no reward for zero command
        return rew_air_time * self.dt * self.config.feet_air_time

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array) -> jax.Array:
        # Penalize motion at zero commands
        return (
            jnp.sum(jnp.abs(joint_angles - self._default_pose))
            * (math.normalize(commands[:2])[1] < 0.1)
            * self.dt
            * self.config.stand_still
        )

    def _reward_foot_slip(self, pipeline_state: State, contact_filt: jax.Array) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        pos = pipeline_state.data.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.data.xpos[self._lower_leg_body_id]
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1))) * self.dt * self.config.foot_slip

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return jnp.float32(done & (step < 500)) * self.dt * self.config.termination

    def _reward_base_height(self, x: Transform) -> jax.Array:
        # Penalize height of the base
        return jnp.square(x.pos[0, 2] - self.config.default_base_height) * self.dt * self.config.base_height

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd
