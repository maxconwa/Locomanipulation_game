"""Observation terms for the arm pose targets."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 1:4], quat[:, 0:1]], dim=-1)


def pose_command_xyzw(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """A pose command as (x, y, z, qx, qy, qz, qw) in the root (pelvis) frame.

    Commands are stored w-first, the Isaac Lab convention the reward terms use.
    The policy sees x-y-z-w, the ROS / Eigen order the robot side speaks.
    """
    command = env.command_manager.get_command(command_name)
    return torch.cat([command[:, :3], _wxyz_to_xyzw(command[:, 3:7])], dim=-1)


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
