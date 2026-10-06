"""Observation terms for the arm targets."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ArmTargetsCommand


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)


def _xyzw_flat(env: ManagerBasedRLEnv, poses: torch.Tensor) -> torch.Tensor:
    return torch.cat([poses[..., :3], _wxyz_to_xyzw(poses[..., 3:])], dim=-1).view(env.num_envs, -1)


def arm_targets_in_root_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The arm command the policies act on: per wrist (x, y, z, qx, qy, qz, qw) in the pelvis frame.

    This is the believed command, kept on the world target by the odometry
    estimate (ArmTargetsCommand.apply_pelvis_motion), not the true one. The
    term stores w-first, the Isaac Lab convention; the policy sees x-y-z-w,
    the ROS / Eigen order the robot side speaks.
    """
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _xyzw_flat(env, term.believed_b)


def true_arm_targets_in_root_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The true targets in the pelvis frame, same layout. Privileged: for the critic, which scores the true point."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _xyzw_flat(env, term.true_targets_in_root())


def arm_goal_active(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 while the env has an arm goal, 0 while it navigates. Shape (num_envs, 1)."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return term.arm_mode.float().unsqueeze(1)


def arm_target_height_drop(env: ManagerBasedRLEnv, command_name: str, visible: bool = True) -> torch.Tensor:
    """How far the current arm goal was lowered below the standing workspace (m). Shape (num_envs, 1).

    visible=False returns zeros: the actors keep the input (same size, so
    older checkpoints load) but are no longer told to crouch; they have to
    read it from the targets' height. The critic keeps the true value.
    """
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return term.height_drop.unsqueeze(1) * float(visible)


def pelvis_height_above_ground(env: ManagerBasedRLEnv, sensor_name: str = "height_scanner") -> torch.Tensor:
    """Pelvis height over the mean of the height scan (m), 0 where every ray missed. Shape (num_envs, 1). Privileged."""
    from .commands import ground_height

    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_height(env.scene.sensors[sensor_name])
    return torch.nan_to_num(height, nan=0.0).unsqueeze(1)


def body_pose_in_root_xyzw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Each of asset_cfg's bodies as (x, y, z, qx, qy, qz, qw) in the root frame, concatenated."""
    asset: Articulation = env.scene[asset_cfg.name]
    poses = []
    for body_id in asset_cfg.body_ids:
        pos_b, quat_b = subtract_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w,
            asset.data.body_pos_w[:, body_id], asset.data.body_quat_w[:, body_id],
        )
        poses.append(torch.cat([pos_b, _wxyz_to_xyzw(quat_unique(quat_b))], dim=-1))
    return torch.cat(poses, dim=-1)


def applied_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """The agent's last action as applied: after the action term's bounds and rate limit, in action units.

    For a bounded, rate-limited term (mdp.RateLimitedJointPositionAction) the
    raw action and the applied one differ; the policy acts on the applied one.
    Any other term: its raw action, like mdp.last_action.
    """
    term = env.action_manager.get_term(action_name)
    return term.applied_actions if hasattr(term, "applied_actions") else term.raw_actions


def arm_target_error_in_root(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Per arm, the wrist's error to the command it acts on: (dx, dy, dz, rx, ry, rz) in the pelvis frame.

    Position: believed target minus wrist position. Rotation: the axis-angle
    rotation that takes the wrist to the believed target orientation. The
    actor's own wrist poses were missing from its observations (only the
    critic had them), so the arm policy had to learn its forward kinematics to
    know how far off it was. Shape (num_envs, 6 * arms).
    """
    from isaaclab.utils.math import compute_pose_error

    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    robot = term.robot
    errors = []
    for arm, body_id in enumerate(term.body_ids):
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]
        )
        target = term.believed_b[:, arm]
        pos_err, rot_err = compute_pose_error(pos_b, quat_b, target[:, :3], target[:, 3:], rot_error_type="axis_angle")
        errors.append(torch.cat([pos_err, rot_err], dim=-1))
    return torch.cat(errors, dim=-1)
