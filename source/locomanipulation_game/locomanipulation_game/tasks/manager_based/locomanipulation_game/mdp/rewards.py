"""Reward terms ported from ALMI-Open's h1_2_lower_env.py.

Each function names the ALMI method it came from so it can be checked against
the source. Two deliberate divergences from legged_gym:

  * legged_gym sets `only_positive_rewards = True`, clipping the total per-step
    reward at zero. Isaac Lab has no equivalent, so early returns here will be
    far more negative than ALMI's. If training diverges in the first few
    hundred iterations, suspect this first.
  * ALMI's `termination` scale is -0.0 (inherited from the legged_gym base and
    never overridden), so there is no terminal penalty. We match that.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


# --- gait clock (ALMI h1_2_lower_env.py, ~line 500) ---
GAIT_PERIOD = 0.8          # seconds per full stride
STANCE_THRESHOLD = 0.55    # phase below this is stance
ZERO_CMD_EPS = 0.1         # command norm below this counts as "standing"


def leg_phase(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Per-leg gait phase in [0, 1). Shape (num_envs, 2): left, right.

    Phase advances with episode time. It is pinned to 0 when the command is
    near zero, so a standing robot has no clock telling it to step. The right
    leg is offset half a cycle, which is what makes the gait alternate.
    """
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < ZERO_CMD_EPS

    t = env.episode_length_buf * env.step_dt
    phase = torch.where(
        standing, torch.zeros_like(t, dtype=torch.float), (t % GAIT_PERIOD) / GAIT_PERIOD
    )
    offset = torch.where(standing, 0.0, 0.5)

    return torch.stack([phase, (phase + offset) % 1.0], dim=-1)

def gait_phase_sin(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Observation term: sin of each leg's phase. Shape (num_envs, 2)."""
    return torch.sin(2.0 * math.pi * leg_phase(env, command_name))




def contact_matches_phase(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """ALMI `_reward_contact`: +1 per foot whose contact agrees with its phase.

    Pays for the foot being down during stance and up during swing. With the
    legs offset half a cycle this is what shapes an alternating walk.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    phase = leg_phase(env, command_name)

    res = torch.zeros(env.num_envs, device=env.device)
    for i, body_id in enumerate(sensor_cfg.body_ids):
        in_contact = forces[:, :, body_id, 2].max(dim=1)[0] > 1.0
        is_stance = phase[:, i] < STANCE_THRESHOLD
        res += (~(in_contact ^ is_stance)).float()
    return res



def feet_swing_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.08,
) -> torch.Tensor:
    """ALMI `_reward_feet_swing_height`: squared error to a target foot height,
    counted only while the foot is airborne.

    At weight -20.0 this is ALMI's largest single penalty. It is what stops the
    robot dragging its feet, and it is the term most likely to need retuning if
    the gait looks wrong.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.square(foot_z - target_height) * ~in_contact, dim=1)



def contact_no_vel(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """ALMI `_reward_contact_no_vel`: penalize a foot moving while in contact.

    The anti-skating term: a planted foot should be stationary. Skating is a
    classic sim artifact that transfers badly to hardware.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0

    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(foot_vel * in_contact.unsqueeze(-1)), dim=(1, 2))


def feet_contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_force: float = 700.0
) -> torch.Tensor:
    """ALMI `_reward_feet_contact_forces`: penalize contact force above a cap.
    Discourages stomping, which is hard on real hardware."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak = forces.norm(dim=-1).max(dim=1)[0]
    return torch.sum((peak - max_force).clip(min=0.0), dim=1)




def body_pair_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.3,
    max_dist: float = 0.6,
) -> torch.Tensor:
    """ALMI `_reward_feet_distance` and `_reward_knee_distance`, which share
    this shape.

    Returns ~1.0 when the horizontal separation of a body pair sits within
    [min_dist, max_dist], falling off sharply outside. Keeps the legs from
    either crossing or splaying. Expects exactly two matched bodies.

    Note this is a POSITIVE reward in ALMI (weights +1.0 and +0.2), not a
    penalty: the robot is paid for keeping a sane stance width.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)

    d_min = torch.clamp(dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(dist - max_dist, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0




def stand_still(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """ALMI `_reward_stand_still`: L1 joint deviation from default, applied only
    when the command is near zero.

    Without this the policy fidgets when given no input, which is obvious and
    unpleasant on hardware.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < ZERO_CMD_EPS

    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(dev), dim=1) * standing


def stance_base_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """ALMI `_reward_stance_base_vel`: penalize base xy velocity at zero command.

    Complements stand_still: that one keeps the joints put, this one keeps the
    robot from drifting.
    """
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < ZERO_CMD_EPS
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * standing




def joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ALMI `_reward_hip_pos`: squared deviation of selected joints from default.

    L2, not the L1 that Isaac Lab's built-in joint_deviation_l1 uses. ALMI
    applies it to hip yaw and roll (their dof indices 0, 2, 6, 8) to stop the
    legs splaying or toeing out.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(dev), dim=1)


def ankle_action_rate_l2(env: ManagerBasedRLEnv, action_ids: list[int]) -> torch.Tensor:
    """ALMI `_reward_ankle_action_rate`: action_rate restricted to the ankles.

    action_ids index the ACTION vector, not the joint array. With
    preserve_order=True and LOWER_BODY_JOINTS ordered hip_yaw, hip_pitch,
    hip_roll, knee, ankle_pitch, ankle_roll (left then right), the ankles are
    [4, 5, 10, 11] -- the same indices ALMI uses. VERIFY this against the
    action manager's ordering before trusting it; wrong indices here penalize
    the wrong joints and the error is silent.
    """
    a = env.action_manager.action[:, action_ids]
    prev = env.action_manager.prev_action[:, action_ids]
    return torch.sum(torch.square(prev - a), dim=1)



# ---------------------------------------------------------------------------
# End-effector pose tracking (upper-body rounds).
#
# Isaac Lab ships UniformPoseCommand, but the matching reward terms live inside
# the Franka reach task (isaaclab_tasks.manager_based.manipulation.reach.mdp),
# not in isaaclab.envs.mdp, so they are reimplemented here.
# ---------------------------------------------------------------------------


def ee_position_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Euclidean distance (m) between a body and its commanded position.

    `asset_cfg` must resolve to exactly one body -- pass body_names as a
    literal link name, not a pattern.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # The command is expressed in the asset's ROOT frame (pelvis on H1-2), so
    # lift it to world before comparing against a world-frame body position.
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3]
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=-1)


def ee_orientation_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Geodesic angle (rad) between a body's orientation and its command.

    command[:, 3:7] is w-first (qw, qx, qy, qz), the Isaac Lab convention --
    NOT the (qx, qy, qz, qw) order used by ROS / Eigen. Roll a ROS quaternion
    at the boundary, not here.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = quat_mul(asset.data.root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def track_ee_pos_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """exp(-d^2 / std^2) on the position error.

    Same kernel shape as track_lin_vel_xy_exp, so the two rounds' task rewards
    read on the same scale in TensorBoard. Positive-valued, which is what makes
    it safe to pair with a fall termination: an all-negative tracking reward
    would pay the policy to terminate early instead of reaching.
    """
    error = ee_position_error(env, command_name, asset_cfg)
    return torch.exp(-error.square() / std**2)


def track_ee_quat_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """exp(-theta^2 / std^2) on the orientation error, std in radians."""
    error = ee_orientation_error(env, command_name, asset_cfg)
    return torch.exp(-error.square() / std**2)