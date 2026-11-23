"""The motion command's tracking errors, logged by Isaac Lab's command manager as ``Metrics/motion/error_*``.

The ten terms of ``MotionCommand._update_metrics`` in mjlab (HybridRobotics' mjlab_research, commit 6ea3e57,
Apache-2.0), ``src/mjlab/tasks/tracking/mdp/commands.py``, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from pretraining.tasks.commands import MotionCommand

ERROR_METRICS = (
    "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
    "error_body_pos", "error_body_rot", "error_body_lin_vel", "error_body_ang_vel",
    "error_joint_pos", "error_joint_vel",
)


def tracking_errors(command: MotionCommand) -> dict[str, torch.Tensor]:
    """Per-env errors between the reference and the robot: the anchor's (world frame), the tracked bodies' (mean
    over bodies; positions and orientations against the targets re-anchored to the robot), the joints' (L2)."""
    return {
        "error_anchor_pos": torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1),
        "error_anchor_rot": quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w),
        "error_anchor_lin_vel": torch.norm(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w, dim=-1),
        "error_anchor_ang_vel": torch.norm(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w, dim=-1),
        "error_body_pos": torch.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1).mean(dim=-1),
        "error_body_rot": quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w).mean(dim=-1),
        "error_body_lin_vel": torch.norm(command.body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1).mean(dim=-1),
        "error_body_ang_vel": torch.norm(command.body_ang_vel_w - command.robot_body_ang_vel_w, dim=-1).mean(dim=-1),
        "error_joint_pos": torch.norm(command.joint_pos - command.robot_joint_pos, dim=-1),
        "error_joint_vel": torch.norm(command.joint_vel - command.robot_joint_vel, dim=-1),
    }
