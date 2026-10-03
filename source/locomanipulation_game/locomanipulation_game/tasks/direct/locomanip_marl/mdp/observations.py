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


def arm_targets_in_root_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Each arm's target as (x, y, z, qx, qy, qz, qw) in the pelvis frame, concatenated.

    The command term stores w-first in the standing frame, the Isaac Lab
    convention the rewards use. The policy sees x-y-z-w in the pelvis frame,
    the ROS / Eigen order the robot side speaks.
    """
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    targets = term.targets_in_root()
    return torch.cat([targets[..., :3], _wxyz_to_xyzw(targets[..., 3:])], dim=-1).view(env.num_envs, -1)


def arm_target_height_drop(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """How far below the standing workspace the current goal is (m). Shape (num_envs, 1).

    The legs' base-height target is standing height minus this, so they need to
    see it: a low target in the pelvis frame alone doesn't say whether the arm
    or the legs should go lower.
    """
    term: ArmTargetsCommand = env.command_manager.get_term(command_name)
    return term.height_drop.unsqueeze(1)


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
