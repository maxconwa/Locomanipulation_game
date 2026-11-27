"""The task's own observation terms. Poses are (x, y, z, qx, qy, qz, qw) in the pelvis frame, the order the robot
side speaks; the arm command stores them w-first."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import compute_pose_error, quat_unique, subtract_frame_transforms

from ..terrain import ground_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ArmTargetsCommand


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)


def _xyzw_flat(env: ManagerBasedRLEnv, poses: torch.Tensor) -> torch.Tensor:
    return torch.cat([poses[..., :3], _wxyz_to_xyzw(poses[..., 3:])], dim=-1).view(env.num_envs, -1)


def arm_targets_in_root_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The arm command the policies act on: its targets in the pelvis frame."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _xyzw_flat(env, term.believed_b)


def true_arm_targets_in_root_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The true targets. Privileged: for the critic."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _xyzw_flat(env, term.true_targets_in_root())


def arm_goal_active(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 during an arm goal, 0 while walking. Shape (num_envs, 1)."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return term.arm_mode.float().unsqueeze(1)


def pelvis_height_above_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pelvis height over the ground under it (m, terrain.py). Shape (num_envs, 1). Privileged."""
    return (env.scene["robot"].data.root_pos_w[:, 2] - ground_height(env)).unsqueeze(1)


def joint_pos_rel_measured(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """joint_pos_rel as the encoders report it: plus each env's encoder offset (LocoManipMarlEnv.encoder_offset, the
    robot's joint order; none on an env that draws none)."""
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    offset = getattr(env, "encoder_offset", None)
    return rel if offset is None else rel + offset[:, asset_cfg.joint_ids]


def joint_vel_rel_reported(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """joint_vel_rel as /lowstate reports it: the joints the env low-passes (LocoManipMarlEnv.joint_vel_lowpass, the
    motor drivers' filter) at the filter's output, the rest as they are."""
    asset: Articulation = env.scene[asset_cfg.name]
    dq = asset.data.joint_vel
    lowpass = getattr(env, "joint_vel_lowpass", None)
    if lowpass is not None:
        dq = dq.clone()
        dq[:, env.joint_vel_lowpass_ids] = lowpass.output.to(dq.dtype)
    return dq[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def body_pose_in_root_xyzw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Each of asset_cfg's bodies in the root frame, concatenated."""
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
    """The agent's last action as its action term applied it, in action units."""
    return env.action_manager.get_term(action_name).applied_actions


def arm_target_error_in_root(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Per arm, the wrist's error to the believed target: (dx, dy, dz, axis-angle rx, ry, rz). Shape (num_envs, 6 * arms)."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _target_errors(term, term.believed_b)


def true_arm_target_error_in_root(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """arm_target_error_in_root against the true targets. Privileged: for the critic."""
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return _target_errors(term, term.true_targets_in_root())


def _target_errors(term: ArmTargetsCommand, targets_b: torch.Tensor) -> torch.Tensor:
    robot = term.robot
    errors = []
    for arm, body_id in enumerate(term.body_ids):
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]
        )
        target = targets_b[:, arm]
        pos_err, rot_err = compute_pose_error(pos_b, quat_b, target[:, :3], target[:, 3:], rot_error_type="axis_angle")
        errors.append(torch.cat([pos_err, rot_err], dim=-1))
    return torch.cat(errors, dim=-1)
