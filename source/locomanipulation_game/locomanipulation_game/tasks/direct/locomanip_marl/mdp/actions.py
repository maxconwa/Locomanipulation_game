"""Joint-position actions with bounded, optionally rate-limited targets.

Run 12's policy commanded targets far past the joint stops (knee actions of
-300 against a clip of 10). This term keeps the plain JointPositionAction
mapping (target = action * scale + offset) and then, once per policy step:

  1. bounds each target to [hard lower - target_margin, hard upper + target_margin]:
     the margin lets the PD controller push against a limit (holding a deep
     squat needs the knee target ~0.37 rad past the knee angle), but no further;
  2. if max_rate is set, limits how far each target may move from the last
     applied one: max_rate (rad/s) during arm goals and settle segments,
     max_rate_navigation while walking, per env from the arm command's mode.
     Flat run A had rate caps (arms 1.0-1.5 rad/s, knee and hip pitch 1.2 during
     arm goals) for slower, safer moves; the user reverted them for the restart.

The policy sees what was applied (applied_actions, in action units, through
mdp.applied_action), and beyond_bounds (also in action units) is what the
bound cut off, for the legs' action_beyond_clip penalty.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "IKResidualArmAction",
    "IKResidualArmActionCfg",
    "RateLimitedJointPositionAction",
    "RateLimitedJointPositionActionCfg",
]


class RateLimitedJointPositionAction(JointPositionAction):
    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        hard = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower = hard[..., 0] - cfg.target_margin
        self._upper = hard[..., 1] + cfg.target_margin
        self._rate = self._per_joint(cfg.max_rate) if cfg.max_rate else None
        self._rate_navigation = self._per_joint(cfg.max_rate_navigation) if cfg.max_rate_navigation else self._rate
        self._applied = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._beyond = torch.zeros_like(self._raw_actions)
        caps = "no rate caps"
        if self._rate is not None:
            caps = f"rate caps (rad/s, arm goals and settle) {dict(zip(self._joint_names, self._rate.tolist()))}"
            if cfg.max_rate_navigation:
                caps += f"; walking {dict(zip(self._joint_names, self._rate_navigation.tolist()))}"
        print(f"[INFO] {type(self).__name__}: targets within the hard limits +- {cfg.target_margin} rad; {caps}")

    def _per_joint(self, rates: dict[str, float]) -> torch.Tensor:
        index_list, _, value_list = string_utils.resolve_matching_names_values(rates, self._joint_names)
        values = torch.full((self._num_joints,), float("inf"), device=self.device)
        values[index_list] = torch.tensor(value_list, device=self.device)
        if torch.isinf(values).any():
            missing = [n for n, v in zip(self._joint_names, values.tolist()) if v == float("inf")]
            raise ValueError(f"{type(self).__name__}: no rate cap for joints {missing}")
        return values

    @property
    def applied_actions(self) -> torch.Tensor:
        """The applied targets in action units: what the policy's action amounted to after the bound and the rate."""
        return (self._applied - self._offset) / self._scale

    @property
    def beyond_bounds(self) -> torch.Tensor:
        """How far each raw action was past its joint's target bound, in action units (0 inside)."""
        return self._beyond

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = self._raw_actions * self._scale + self._offset
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._beyond = (target - bounded).abs() / abs(self._scale)
        if self._rate is None:
            self._applied = bounded
            self._processed_actions = self._applied
            return
        rate = self._rate.expand_as(bounded)
        if self.cfg.max_rate_navigation:
            arm = self._env.command_manager.get_term(self.cfg.mode_command_name)
            walking = ~(arm.arm_mode | arm.settling)
            rate = torch.where(walking.unsqueeze(1), self._rate_navigation, rate)
        step = rate * self._env.step_dt
        self._applied = torch.maximum(torch.minimum(bounded, self._applied + step), self._applied - step)
        self._processed_actions = self._applied

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # the reset events have written the new joint state: start the rate limit from it
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._beyond[ids] = 0.0


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RateLimitedJointPositionAction

    max_rate: dict[str, float] | None = None
    """rad/s per joint (regex keys, every joint covered): the cap during arm goals and settle segments,
    or always when max_rate_navigation is None. None: no rate limit, bounds only."""
    max_rate_navigation: dict[str, float] | None = None
    """rad/s per joint while walking (not in an arm goal or a settle)."""
    mode_command_name: str = "arm_targets"
    """The ArmTargetsCommand whose arm_mode / settling pick the cap."""
    target_margin: float = 0.4
    """rad past each hard joint limit a target may go."""


class IKResidualArmAction(JointPositionAction):
    """Arm joint targets from one differential-IK step toward the arm command, plus the policy's residual.

    Each policy step, for each arm, a damped-least-squares step (Isaac Lab's
    DifferentialIKController) moves the arm's joint targets from the current
    joint angles toward the command the policies see (ArmTargetsCommand's
    believed pose, pelvis frame), with the wrist Jacobian rotated into the
    pelvis frame. The policy's action, times scale, is added in joint space,
    and each target is bounded to [hard lower - target_margin, hard upper +
    target_margin]. The IK step is capped at max_ik_step rad per joint per
    policy step.

    Run F's arms, learning the whole map from joint angles and a target to
    joint targets, stopped at ~12 cm and reached ~15% of goals (5 cm / 0.35
    rad), so its reach curriculum never moved. The IK step gets near a
    target from the start; the policy learns what IK can't: gravity sag,
    self-collision, joint limits, the moving pelvis. A target the arms can't
    reach (a lowered goal) leaves an error that only the legs can close.

    With residual_cutoff_hz set, the residual is low-passed first (one-pole,
    per policy step) and the policy observes the filtered residual. Run H's
    residual chattered at ~12 Hz while holding a goal: every arm joint
    reversed 21-30 times a second, against 5-10 with the residual off.

    applied_actions (the residual, action units) is what mdp.applied_action
    observes; beyond_bounds as in RateLimitedJointPositionAction.
    """

    cfg: IKResidualArmActionCfg

    def __init__(self, cfg: IKResidualArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        robot = self._asset
        hard = robot.data.joint_pos_limits[:, self._joint_ids]
        self._lower = hard[..., 0] - cfg.target_margin
        self._upper = hard[..., 1] + cfg.target_margin
        self._arm_cols, self._jac_cols, self._jac_body, self._body_ids, self._ik = [], [], [], [], []
        names = list(self._joint_names)
        for body, arm_joints in zip(cfg.body_names, cfg.arm_joint_names):
            cols = [names.index(j) for j in arm_joints]  # columns in this term's action
            joint_ids, _ = robot.find_joints(arm_joints, preserve_order=True)
            body_id = robot.find_bodies(body)[0][0]
            fixed = robot.is_fixed_base
            self._arm_cols.append(cols)
            # PhysX Jacobians: a floating base adds 6 leading columns; a fixed one drops the root link
            self._jac_cols.append([j + (0 if fixed else 6) for j in joint_ids])
            self._jac_body.append(body_id - 1 if fixed else body_id)
            self._body_ids.append(body_id)
            self._ik.append(DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                            ik_params={"lambda_val": cfg.ik_damping}),
                num_envs=self.num_envs, device=self.device,
            ))
        self._applied = robot.data.joint_pos[:, self._joint_ids].clone()
        self._beyond = torch.zeros_like(self._raw_actions)
        self.ik_targets = self._applied.clone()  # the IK command alone, before the residual
        # one-pole low-pass of the residual: weight of the new action per policy step (1: no filter)
        self._filter_alpha = 1.0
        if cfg.residual_cutoff_hz is not None:
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
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

        self._raw_actions[:] = actions
        # detached: outside skrl's no_grad (eval scripts) the recursion would chain the policy's graph across steps
        self._residual = self._residual + self._filter_alpha * (actions.detach() - self._residual)
        robot = self._asset
        command = self._env.command_manager.get_term(self.cfg.command_name).believed_b  # (N, arms, 7), wxyz
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        to_root = matrix_from_quat(quat_inv(root_quat))
        jacobians = robot.root_physx_view.get_jacobians()
        q = robot.data.joint_pos[:, self._joint_ids]
        # integrate: step from the previous IK command (holds against gravity sag); else from the measured angles
        ik = self.ik_targets.clone() if self.cfg.ik_integrate else q.clone()
        for arm, (cols, jac_cols, jac_body, body) in enumerate(zip(self._arm_cols, self._jac_cols, self._jac_body, self._body_ids)):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
            )
            jac = jacobians[:, jac_body][:, :, jac_cols]  # (N, 6, arm joints), world frame
            jac = torch.cat([torch.bmm(to_root, jac[:, :3]), torch.bmm(to_root, jac[:, 3:])], dim=1)
            self._ik[arm].set_command(command[:, arm])
            q_arm = q[:, cols]
            step = self.cfg.ik_gain * (self._ik[arm].compute(ee_pos, ee_quat, jac, q_arm) - q_arm)
            ik[:, cols] = ik[:, cols] + step.clamp(-self.cfg.max_ik_step, self.cfg.max_ik_step)
        self.ik_targets = torch.maximum(torch.minimum(ik, self._upper), self._lower)  # bounded, so no windup past them
        target = ik + self._residual * self._scale
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._beyond = (target - bounded).abs() / abs(self._scale)
        self._applied = bounded
        self._processed_actions = bounded

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self.ik_targets[ids] = self._applied[ids]
        self._residual[ids] = 0.0
        self._beyond[ids] = 0.0


@configclass
class IKResidualArmActionCfg(JointPositionActionCfg):
    """Joint names must cover both arms' joints; scale is the residual's rad per action unit."""

    class_type: type = IKResidualArmAction

    body_names: list[str] = MISSING
    """One end-effector body per arm, in the order of the arm command's bodies."""
    arm_joint_names: list[list[str]] = MISSING
    """Per arm, its joints, in kinematic order."""
    command_name: str = "arm_targets"
    ik_damping: float = 0.05
    """Damped-least-squares lambda (Isaac Lab's default is 0.01; more damping near singular poses)."""
    max_ik_step: float = 0.1
    """rad per joint per policy step: the IK step's cap, not a speed limit on the residual. Pure IK (residual 0,
    run F's legs), easiest goals: 0.05 reached 6% (9.5 cm closest: too small a step to hold the arm up against
    gravity, PD torque being stiffness x step), 0.1 reached 72% (2.8 cm), 0.2 reached 69% (2.6 cm)."""
    ik_integrate: bool = False
    """Step from the previous IK command instead of the measured joint angles, so the command can stay ahead
    of a sagging arm. Tested and worse: 6-7% of the easiest goals reached and more falls (gain 0.3-0.5)."""
    ik_gain: float = 1.0
    """Share of the IK step taken per policy step."""
    residual_cutoff_hz: float | None = None
    """Low-pass the residual at this cut-off (one-pole, per policy step) before adding it; None: unfiltered."""
    target_margin: float = 0.4
