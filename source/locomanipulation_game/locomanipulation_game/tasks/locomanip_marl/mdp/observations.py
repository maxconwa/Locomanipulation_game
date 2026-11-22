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
    """The arm command the policies act on: the believed targets, kept on the world target by odometry."""
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


"""
The world model's state (MarlObservationsCfg.world): privileged and noise-free, the pelvis frame.
"""


def target_history_depth(action_delay_range: tuple[int, int] | None, decimation: int) -> int:
    """Policy steps before this one whose joint targets can still reach the robot: a delay of d substeps reaches back
    ceil(d / decimation) steps (LocoManipMarlEnv._target_history holds this many + 1). 0 without a delay."""
    if action_delay_range is None:
        return 0
    return -(-int(action_delay_range[1]) // int(decimation))


def joint_target_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Every action term's joint targets (the bounded ones it applies) for this policy step and the target_history_depth
    before it, less the default pose of those joints, in action-term order and per term newest first: (num_envs,
    sum over terms of joints x (depth + 1)). With the env's action delay these are the targets still on their way to
    the robot (LocoManipMarlEnv._target_history); without one, this step's alone.

    The observation manager calls every term once before the env has built its history, for the size: zeros then."""
    robot = env.scene["robot"]
    terms = [env.action_manager.get_term(name) for name in env.action_manager.active_terms]
    history = getattr(env, "_target_history", None)
    if history is None:
        steps = target_history_depth(env.cfg.action_delay_range, env.cfg.decimation) + 1
        return torch.zeros(env.num_envs, steps * sum(term._num_joints for term in terms), device=env.device)
    default = robot.data.default_joint_pos
    if not history:  # no delay: the targets as applied (processed_actions is never swapped for an older step's)
        return torch.cat([term.processed_actions - default[:, term._joint_ids] for term in terms], dim=1)
    return torch.cat([(targets - default[:, term._joint_ids].unsqueeze(1)).flatten(1)
                      for term, targets in zip(terms, history)], dim=1)


def body_pos_in_root(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Each of asset_cfg's bodies' position in the root frame (m), concatenated: (num_envs, 3 x bodies)."""
    asset: Articulation = env.scene[asset_cfg.name]
    data = asset.data
    positions = [subtract_frame_transforms(data.root_pos_w, data.root_quat_w, data.body_pos_w[:, body_id])[0]
                 for body_id in asset_cfg.body_ids]
    return torch.cat(positions, dim=-1)


def contact_force_z_mean(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Each of sensor_cfg's bodies' vertical contact force (N, pushing only), the mean over the policy step's physics
    substeps, as mdp.feet_load_balance_band reads it: (num_envs, bodies)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].clamp(min=0.0).mean(dim=1)


def cycle_mean_velocity(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """The pelvis velocity's mean over the last gait cycle, (vx, vy, wz) in the pelvis frame: the velocity command's
    own window (mdp.ModalVelocityCommand), the same samples the tracking rewards' windows hold (mdp._CycleMean, on
    root_lin_vel_b / root_ang_vel_b). Theirs are private to each reward term and gone where the legs' rewards are off;
    the command's differs only after a reset, where it holds the reset state's sample and theirs nothing yet."""
    term = env.command_manager.get_term(command_name)
    return term._vel_samples.sum(dim=1) / term._count.clamp(min=1.0).unsqueeze(-1)


def gantry_support_share(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The share of the robot's weight a pre-pose gantry carries now (LocoManipMarlEnv.support_share): (num_envs, 1)."""
    if getattr(env, "_prepose", None) is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env.support_share().unsqueeze(1)


def gantry_hook_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(num_envs, 3) m, world frame: the pre-pose rope's hook less its anchor now (mdp.prepose_reset.anchor_at), what
    the rope's spring and tether pull on; 0 while no rope is attached (and without the prepose event)."""
    prepose = getattr(env, "_prepose", None)
    if prepose is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    t = env.episode_length_buf * env.step_dt
    offset = prepose._hook()[0] - prepose.anchor_at(t)
    return torch.where((prepose.support_share(t) > 0.0).unsqueeze(1), offset, 0.0)


def episode_time(env: ManagerBasedRLEnv, max_time: float) -> torch.Tensor:
    """The episode's time (s) clipped at max_time: (num_envs, 1). What depends on it (the gantry's release, the
    pre-pose's settling) is over by then."""
    return (env.episode_length_buf * env.step_dt).clamp(max=max_time).unsqueeze(1)


def action_term_state(env: ManagerBasedRLEnv, action_name: str, attr: str,
                      relative_to_default: bool = False) -> torch.Tensor:
    """An action term's internal state tensor `attr` (num_envs, k), e.g. mdp.IKPostureArmAction's _ik_q or _integral;
    relative_to_default: less the term's joints' default positions."""
    term = env.action_manager.get_term(action_name)
    value = getattr(term, attr)
    if relative_to_default:
        value = value - env.scene["robot"].data.default_joint_pos[:, term._joint_ids]
    return value
