"""Joint-position actions with bounded targets.

Both terms bound each target to [hard lower - target_margin, hard upper + target_margin] (a negative margin keeps it
inside GOLEM's clip) and then to where the PD torque it asks for at the measured state, stiffness (target - q) -
damping qd, stays within torque_headroom of the joint's effort limit (torque_bounds): GOLEM's safety layer e-stops
on torque. The policy observes what was applied (applied_actions, action units), and beyond_bounds (action units)
is what the position bound cut off, for the legs' action_beyond_clip penalty.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "BoundedJointPositionAction",
    "BoundedJointPositionActionCfg",
    "IKResidualArmAction",
    "IKResidualArmActionCfg",
]


def torque_bounds(asset, joint_ids, headroom: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per joint, the targets whose PD torque at the measured state stays within headroom x the effort limit:
    [q + (-h tau + kd qd) / kp, q + (h tau + kd qd) / kp], with the motor's own damping (the randomized passive
    part is joint friction, not motor torque)."""
    data = asset.data
    q, qd = data.joint_pos[:, joint_ids], data.joint_vel[:, joint_ids]
    kp, kd = data.joint_stiffness[:, joint_ids], data.default_joint_damping[:, joint_ids]
    tau = headroom * data.joint_effort_limits[:, joint_ids]
    return q + (kd * qd - tau) / kp, q + (kd * qd + tau) / kp


class BoundedJointPositionAction(JointPositionAction):
    """JointPositionAction (target = action * scale + offset) with the position and torque bounds."""

    cfg: BoundedJointPositionActionCfg

    def __init__(self, cfg: BoundedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        hard = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower = hard[..., 0] - cfg.target_margin
        self._upper = hard[..., 1] + cfg.target_margin
        self._applied = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._beyond = torch.zeros_like(self._raw_actions)
        print(f"[INFO] {type(self).__name__}: targets within the hard limits +- {cfg.target_margin} rad")

    @property
    def applied_actions(self) -> torch.Tensor:
        return (self._applied - self._offset) / self._scale

    @property
    def beyond_bounds(self) -> torch.Tensor:
        return self._beyond

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = self._raw_actions * self._scale + self._offset
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._beyond = (target - bounded).abs() / abs(self._scale)
        low, high = torque_bounds(self._asset, self._joint_ids, self.cfg.torque_headroom)
        self._applied = torch.maximum(torch.minimum(bounded, high), low)
        self._processed_actions = self._applied

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._beyond[ids] = 0.0


@configclass
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = BoundedJointPositionAction

    target_margin: float = MISSING
    """rad past each hard joint limit a target may go (negative: inside it)."""
    torque_headroom: float = MISSING
    """Share of the effort limit the PD torque of a target may ask for at the measured state."""


class IKResidualArmAction(JointPositionAction):
    """Arm joint targets: one damped-least-squares IK step toward the arm command, plus the policy's residual.

    Each policy step and arm, Isaac Lab's DifferentialIKController steps the arm from its measured joint angles
    toward the command the policies see (ArmTargetsCommand.believed_b, pelvis frame), capped at max_ik_step rad
    per joint. The policy's action, low-passed at residual_cutoff_hz (one-pole, per policy step) and times scale,
    is added in joint space. The policy observes the filtered residual.
    """

    cfg: IKResidualArmActionCfg

    def __init__(self, cfg: IKResidualArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        hard = robot.data.joint_pos_limits[:, self._joint_ids]
        self._lower = hard[..., 0] - cfg.target_margin
        self._upper = hard[..., 1] + cfg.target_margin
        self._arm_cols, self._jac_cols, self._jac_body, self._body_ids, self._ik = [], [], [], [], []
        names = list(self._joint_names)
        for body, arm_joints in zip(cfg.body_names, cfg.arm_joint_names):
            joint_ids, _ = robot.find_joints(arm_joints, preserve_order=True)
            body_id = robot.find_bodies(body)[0][0]
            self._arm_cols.append([names.index(j) for j in arm_joints])  # columns in this term's action
            # PhysX Jacobians of a floating base have 6 leading root columns
            self._jac_cols.append([j + 6 for j in joint_ids])
            self._jac_body.append(body_id)
            self._body_ids.append(body_id)
            self._ik.append(DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                            ik_params={"lambda_val": cfg.ik_damping}),
                num_envs=self.num_envs, device=self.device,
            ))
        self._applied = robot.data.joint_pos[:, self._joint_ids].clone()
        self._beyond = torch.zeros_like(self._raw_actions)
        self._filter_alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.residual_cutoff_hz * env.step_dt)
        self._residual = torch.zeros_like(self._raw_actions)  # the filtered residual, action units
        print(f"[INFO] {type(self).__name__}: IK on {cfg.body_names}, damping {cfg.ik_damping}, step cap"
              f" {cfg.max_ik_step} rad, residual scale {cfg.scale}, bounds +- {cfg.target_margin} rad past the limits,"
              f" residual low-pass {cfg.residual_cutoff_hz} Hz (alpha {self._filter_alpha:.3f})")

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._residual

    @property
    def beyond_bounds(self) -> torch.Tensor:
        return self._beyond

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # detached: outside skrl's no_grad the recursion would chain the policy's graph across steps
        self._residual = self._residual + self._filter_alpha * (actions.detach() - self._residual)
        robot = self._asset
        command = self._env.command_manager.get_term(self.cfg.command_name).believed_b  # (N, arms, 7), wxyz
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        to_root = matrix_from_quat(quat_inv(root_quat))
        jacobians = robot.root_physx_view.get_jacobians()
        q = robot.data.joint_pos[:, self._joint_ids]
        ik = q.clone()
        for arm, (cols, jac_cols, jac_body, body) in enumerate(zip(self._arm_cols, self._jac_cols, self._jac_body, self._body_ids)):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
            )
            jac = jacobians[:, jac_body][:, :, jac_cols]  # (N, 6, arm joints), world frame
            jac = torch.cat([torch.bmm(to_root, jac[:, :3]), torch.bmm(to_root, jac[:, 3:])], dim=1)
            self._ik[arm].set_command(command[:, arm])
            q_arm = q[:, cols]
            step = self._ik[arm].compute(ee_pos, ee_quat, jac, q_arm) - q_arm
            ik[:, cols] = ik[:, cols] + step.clamp(-self.cfg.max_ik_step, self.cfg.max_ik_step)
        target = ik + self._residual * self._scale
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._beyond = (target - bounded).abs() / abs(self._scale)
        low, high = torque_bounds(robot, self._joint_ids, self.cfg.torque_headroom)
        self._applied = torch.maximum(torch.minimum(bounded, high), low)
        self._processed_actions = self._applied

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._residual[ids] = 0.0
        self._beyond[ids] = 0.0


@configclass
class IKResidualArmActionCfg(JointPositionActionCfg):
    """joint_names must cover both arms' joints; scale is the residual's rad per action unit."""

    class_type: type = IKResidualArmAction

    body_names: list[str] = MISSING
    """One end-effector body per arm, in the order of the arm command's bodies."""
    arm_joint_names: list[list[str]] = MISSING
    """Per arm, its joints, in kinematic order."""
    command_name: str = "arm_targets"
    ik_damping: float = 0.05
    """Damped-least-squares lambda."""
    max_ik_step: float = 0.1
    """rad per joint per policy step."""
    residual_cutoff_hz: float = MISSING
    target_margin: float = MISSING
    torque_headroom: float = MISSING
