"""The tracking terminations ``amass_s1`` ran with: anchor height, anchor tilt and motion end (time-out is Isaac
Lab's).

Ported from mjlab (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0),
``src/mjlab/tasks/tracking/mdp/terminations.py``: ``bad_anchor_pos_z_only``, ``bad_anchor_ori``, ``motion_end``
(the end-effector terminations are left out: ``apply_gencoord`` removes ``ee_body_pos``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from pretraining.tasks.commands import MotionCommand


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = _command(env, command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    next_time_steps = command.time_steps + 1
    return torch.isin(next_time_steps, command.motion.motion_ends)
