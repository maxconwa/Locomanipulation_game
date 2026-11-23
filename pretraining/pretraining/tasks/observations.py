"""Observation terms of the tracking env: the actor's IMU and biased joint terms, the critic's tracking terms, the
full state and the goal window.

Ported from mjlab (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0): ``motion_anchor_pos_b``,
``motion_anchor_ori_b``, ``robot_body_pos_b``, ``robot_body_ori_b`` of ``src/mjlab/tasks/tracking/mdp/observations.py``
and ``joint_pos_rel(biased=True)`` of ``src/mjlab/envs/mdp/observations.py``; from SuperMimic (HybridRobotics, commit
7e94b07, Apache-2.0) ``src/supermimic/tasks/env_cfgs.py``: the full-state terms (``root_pos`` ... ``joint_vel``) and
the goal terms (``motion_command_window``, its windowed ``motion_anchor_pos_b`` / ``motion_anchor_ori_b``, here
``goal_anchor_pos_b`` / ``goal_anchor_ori_b``). MuJoCo sensors are read from the bodies they sit on: the torso IMU
(h12_train's ``imu_in_torso`` gyro and up vector) from ``torso_link``, the pelvis IMU (site at the pelvis origin)
from the root link. Joint terms are in ``JOINT_NAMES`` order through ``asset_cfg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

from pretraining.supermimic.goal_window import assemble_anchor_window, motion_anchor_window, motion_goal_window
from pretraining.tasks.events import encoder_bias

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from pretraining.tasks.commands import MotionCommand

_ROBOT = SceneEntityCfg("robot")


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


# --- the actor's sensors ---------------------------------------------------------------------------------------------


def body_projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gravity's direction in the frame of ``asset_cfg``'s one body (an IMU's up vector, negated)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    return quat_apply_inverse(quat, asset.data.GRAVITY_VEC_W)


def body_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angular velocity of ``asset_cfg``'s one body in its own frame (a gyro on it)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0]
    return quat_apply_inverse(asset.data.body_link_quat_w[:, body], asset.data.body_link_ang_vel_w[:, body])


def joint_pos_rel_biased(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """``q + encoder bias - default`` (mjlab's ``joint_pos_rel(biased=True)``); the bias is ``env.encoder_bias``."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return asset.data.joint_pos[:, ids] + encoder_bias(env) - asset.data.default_joint_pos[:, ids]


# --- the critic's ----------------------------------------------------------------------------------------------------


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """The pelvis IMU's velocimeter: the root link origin's velocity in the root frame."""
    return env.scene[asset_cfg.name].data.root_link_lin_vel_b


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """The pelvis IMU's gyro: the root link's angular velocity in the root frame."""
    return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def _robot_bodies_in_anchor_frame(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    num_bodies = len(command.cfg.body_names)
    return subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )


def robot_body_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    pos_b, _ = _robot_bodies_in_anchor_frame(_command(env, command_name))
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    _, ori_b = _robot_bodies_in_anchor_frame(_command(env, command_name))
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# --- the full state (the world model's) ------------------------------------------------------------------------------


def root_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_pos_w - env.scene.env_origins


def root_ori(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_quat_w


def root_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def root_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_ang_vel_w


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


# --- the goal --------------------------------------------------------------------------------------------------------


def motion_command_window(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Goal command: joint targets over a future window of reference frames."""
    command = _command(env, command_name)
    return motion_goal_window(command.motion, command.time_steps)


def _goal_anchor_rel(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Windowed motion anchor as ``(rel_pos, rel_ori)`` in the robot root frame.

    ``env_origins`` is added so the motion anchor shares ``robot_anchor_pos_w``'s
    world frame before the relative transform.
    """
    command = _command(env, command_name)
    target_pos, target_quat = motion_anchor_window(command.motion, command.time_steps)
    target_pos = target_pos + env.scene.env_origins[:, None, :]
    return assemble_anchor_window(command.robot_anchor_pos_w, command.robot_anchor_quat_w, target_pos, target_quat)


def goal_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Windowed anchor translation relative to the robot root (world-model only)."""
    return _goal_anchor_rel(env, command_name)[0]


def goal_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Windowed anchor orientation (rot6d) relative to the robot root."""
    return _goal_anchor_rel(env, command_name)[1]
