"""Joint-position actions with bounded targets.

Both terms bound each target to [hard lower - target_margin, hard upper + target_margin] (a negative margin keeps it
inside GOLEM's clip), narrowed to joint_limits where the cfg gives them (GOLEM's position e-stop range), and then to
where the PD torque it asks for at the measured state, stiffness (target - q) - damping qd, stays within
torque_headroom of the joint's effort limit (torque_bounds): GOLEM's safety layer e-stops on torque. What the policies
observe of their action: the legs the applied target in action units, the arms their low-passed residual. The legs'
beyond_bounds (action units) is what the position bound cut off, for their action_beyond_clip penalty.
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
from isaaclab.utils.math import (compute_pose_error, matrix_from_quat, quat_error_magnitude, quat_inv,
                                 subtract_frame_transforms)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "BoundedJointPositionAction",
    "BoundedJointPositionActionCfg",
    "IKResidualArmAction",
    "IKResidualArmActionCfg",
    "IKPostureArmAction",
    "IKPostureArmActionCfg",
]


def position_bounds(asset, joint_ids, joint_names, margin: float, joint_limits: dict | None):
    """(lower, upper) of each target, (N, J): the hard limits widened by margin, narrowed to joint_limits[name]."""
    hard = asset.data.joint_pos_limits[:, joint_ids]
    lower, upper = hard[..., 0] - margin, hard[..., 1] + margin
    for i, name in enumerate(joint_names):
        if joint_limits and name in joint_limits:
            low, high = joint_limits[name]
            lower[:, i] = lower[:, i].clamp(min=low)
            upper[:, i] = upper[:, i].clamp(max=high)
    return lower, upper


def torque_bounds(asset, joint_ids, headroom: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per joint, the targets whose PD torque at the measured state stays within headroom x the effort limit:
    [q + (-h tau + kd qd) / kp, q + (h tau + kd qd) / kp], with the configured gains (explicit actuators zero the
    sim's)."""
    data = asset.data
    q, qd = data.joint_pos[:, joint_ids], data.joint_vel[:, joint_ids]
    kp, kd = data.default_joint_stiffness[:, joint_ids], data.default_joint_damping[:, joint_ids]
    tau = headroom * data.joint_effort_limits[:, joint_ids]
    return q + (kd * qd - tau) / kp, q + (kd * qd + tau) / kp


class _BoundedTargets(JointPositionAction):
    """JointPositionAction whose targets are bounded (the module docstring); subclasses compute the target."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._lower, self._upper = position_bounds(
            self._asset, self._joint_ids, self._joint_names, cfg.target_margin, cfg.joint_limits
        )
        self._applied = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def _describe(self, extra: str = "") -> None:
        print(f"[INFO] {type(self).__name__}: {extra}targets within the hard limits +- {self.cfg.target_margin} rad"
              f"{' and joint_limits' if self.cfg.joint_limits else ''}: " + ", ".join(
                  f"{n} [{lo:.3f}, {hi:.3f}]" for n, lo, hi in zip(self._joint_names, self._lower[0].tolist(), self._upper[0].tolist())))

    def _apply_bounds(self, target: torch.Tensor) -> torch.Tensor:
        """Bound the target, apply it; returns the position-bounded target (before the torque bound)."""
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        low, high = torque_bounds(self._asset, self._joint_ids, self.cfg.torque_headroom)
        self._applied = torch.maximum(torch.minimum(bounded, high), low)
        self._processed_actions = self._applied
        return bounded

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]


class BoundedJointPositionAction(_BoundedTargets):
    """JointPositionAction (target = action * scale + offset, offset the default pose) with the bounds. A scale of 0
    on every joint holds the default pose whatever the action (LocoManip-ArmsOnly-v0's legs); its applied action and
    its overshoot past the bounds then read 0, not the division by 0."""

    cfg: BoundedJointPositionActionCfg

    def __init__(self, cfg: BoundedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._beyond = torch.zeros_like(self._raw_actions)
        self._held = not bool(torch.as_tensor(self._scale).any())
        self._describe("the default pose held (scale 0): " if self._held else "")

    @property
    def applied_actions(self) -> torch.Tensor:
        if self._held:
            return torch.zeros_like(self._raw_actions)
        return (self._applied - self._offset) / self._scale

    @property
    def beyond_bounds(self) -> torch.Tensor:
        return self._beyond

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = self._raw_actions * self._scale + self._offset
        bounded = self._apply_bounds(target)
        self._beyond = torch.zeros_like(target) if self._held else (target - bounded).abs() / abs(self._scale)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._beyond[slice(None) if env_ids is None else env_ids] = 0.0


@configclass
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = BoundedJointPositionAction

    target_margin: float = MISSING
    """rad past each hard joint limit a target may go (negative: inside it)."""
    joint_limits: dict[str, tuple[float, float]] | None = None
    """joint name -> (low, high) the target must also stay inside (golem_safety.position_envelope)."""
    torque_headroom: float = MISSING
    """Share of the effort limit the PD torque of a target may ask for at the measured state."""


class IKResidualArmAction(_BoundedTargets):
    """Arm joint targets: one damped-least-squares IK step toward the arm command, plus the policy's residual.

    Each policy step and arm, Isaac Lab's DifferentialIKController steps the arm from its measured joint angles
    toward the command the policies see (ArmTargetsCommand.believed_b, pelvis frame), capped at max_ik_step rad per
    joint. The policy's action, low-passed at residual_cutoff_hz (one pole, per policy step) and times scale, is added
    in joint space, times cfg.residual_gain; with cfg.residual_ramp, also times the arm command's goal_ramp (0 until the warm start ends, then
    up to 1 with the arm-goal share). The policy observes the filtered residual.
    """

    cfg: IKResidualArmActionCfg

    def __init__(self, cfg: IKResidualArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        robot = self._asset
        self._arm_cols, self._jac_cols, self._body_ids, self._jac_body_ids, self._ik = [], [], [], [], []
        names = list(self._joint_names)
        # PhysX Jacobians of a floating base have 6 leading root columns; a fixed base's (the pelvis pinned,
        # LocoManip-ArmsOnly-v0) have none and leave the root body out of their rows
        root_cols, root_rows = (0, 1) if robot.is_fixed_base else (6, 0)
        for body, arm_joints in zip(cfg.body_names, cfg.arm_joint_names):
            joint_ids, _ = robot.find_joints(arm_joints, preserve_order=True)
            self._arm_cols.append([names.index(j) for j in arm_joints])  # columns in this term's action
            self._jac_cols.append([j + root_cols for j in joint_ids])
            self._body_ids.append(robot.find_bodies(body)[0][0])
            self._jac_body_ids.append(self._body_ids[-1] - root_rows)
            self._ik.append(DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls",
                                            ik_params={"lambda_val": cfg.ik_damping}),
                num_envs=self.num_envs, device=self.device,
            ))
        self._filter_alpha = 1.0 - math.exp(-2.0 * math.pi * cfg.residual_cutoff_hz * env.step_dt)
        self._residual = torch.zeros_like(self._raw_actions)  # the filtered residual, action units
        self._describe(f"IK on {cfg.body_names}, damping {cfg.ik_damping}, step cap {cfg.max_ik_step} rad, residual"
                       f" scale {cfg.scale}, residual low-pass {cfg.residual_cutoff_hz} Hz (alpha"
                       f" {self._filter_alpha:.3f}); ")

    @property
    def applied_actions(self) -> torch.Tensor:
        return self._residual

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # detached: outside skrl's no_grad the recursion would chain the policy's graph across steps
        self._residual = self._residual + self._filter_alpha * (actions.detach() - self._residual)
        robot = self._asset
        term = self._env.command_manager.get_term(self.cfg.command_name)
        command = term.believed_b  # (N, arms, 7), wxyz
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        to_root = matrix_from_quat(quat_inv(root_quat))
        jacobians = robot.root_physx_view.get_jacobians()
        q = robot.data.joint_pos[:, self._joint_ids]
        ik = q.clone()
        for arm, (cols, jac_cols, body, jac_body) in enumerate(
                zip(self._arm_cols, self._jac_cols, self._body_ids, self._jac_body_ids)):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
            )
            jac = jacobians[:, jac_body][:, :, jac_cols]  # (N, 6, arm joints), world frame
            jac = torch.cat([torch.bmm(to_root, jac[:, :3]), torch.bmm(to_root, jac[:, 3:])], dim=1)
            self._ik[arm].set_command(command[:, arm])
            q_arm = q[:, cols]
            step = self._ik[arm].compute(ee_pos, ee_quat, jac, q_arm) - q_arm
            ik[:, cols] = ik[:, cols] + step.clamp(-self.cfg.max_ik_step, self.cfg.max_ik_step)
        gain = (term.goal_ramp if self.cfg.residual_ramp else 1.0) * self.cfg.residual_gain
        self._apply_bounds(ik + self._residual * (self._scale * gain))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._residual[slice(None) if env_ids is None else env_ids] = 0.0


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
    max_ik_step: float = 0.05
    """rad per joint per policy step (5 rad/s at 100 Hz)."""
    residual_cutoff_hz: float = MISSING
    residual_ramp: bool = False
    """Scale the residual by the arm command's goal_ramp: the IK alone holds the arms until the warm start ends."""
    residual_gain: float = 1.0
    """Times scale (one number, so a run can override it; scale is per joint): 0 holds the arms by the IK alone."""
    target_margin: float = MISSING
    joint_limits: dict[str, tuple[float, float]] | None = None
    """joint name -> (low, high) the target must also stay inside (golem_safety.position_envelope)."""
    torque_headroom: float = MISSING


class IKPostureArmAction(IKResidualArmAction):
    """IKResidualArmAction's arms with a limit-aware, posture-seeking IK step and a policy that also sets the posture.

    The action has two halves per joint (joint_names order): the residual, as IKResidualArmAction's, then a posture
    offset (posture_actions False: the residual alone). Both are low-passed at residual_cutoff_hz; the policy observes both. Each policy step and arm, from the
    measured joint angles:
    - the task: the 6-D pose error to the arm command in the pelvis frame, [position, rot_weight x axis-angle], and the
      wrist Jacobian's rows weighted alike, so a metre of position error counts as 1 / rot_weight rad of rotation;
    - the step: damped least squares on the joints still free, plus the posture pull projected into the task's null
      space, null_gain x (posture - q), where posture = the default joint angles (with posture_from_command, the arm
      command's seed_q: the joint angles of the goal table's row nearest the command) + posture_scale x posture_gain x
      the posture offset, inside the IK's bounds. A joint whose step would take it within limit_margin of its target
      bounds (golem_safety's envelope) is locked and the step solved again without it, ik_passes times;
    - the target: the step capped at max_ik_step per joint, plus the residual times scale x residual_gain, bounded as
      IKResidualArmAction's.
    The plain DLS step pinned joints on their clips (the goal tables' poses come from random joint angles; a 7-joint
    arm's redundancy left the IK free to wander into a limit) and weighted a radian like a metre.

    With trajectory (LocoManip-ArmsOnly-v0) there is no IK step: the target moves from its own last value to the
    posture, max_ik_step per joint per step (with trajectory_max_vel, each arm's joints together, speed- and
    acceleration-limited), and the residual corrects the rest (the cfg fields' docs say why).

    With freeze_residual_on_goal (trajectory only), an env's residual is latched once its target has stopped at the
    posture and both wrists have been inside freeze_pos_tol / freeze_rot_tol of the command the policy sees (pelvis
    frame) for freeze_settle_s: the policy's output is ignored and its correction (the gravity sag's, among others)
    held as it was, until the posture moves (the next goal) or the arm goal ends.

    With gravity_offset, the target also carries each arm joint's gravity torque at the target (the arm command's
    ArmKinematics, the URDF's masses, gravity from the pelvis's projected gravity) over its nominal kp, so the PD holds
    the target's angles instead of sagging under the arms' weight; the residual corrects what the model misses.

    With residual_integrate, the residual isn't low-passed but integrated: each step it leaks toward 0 (time constant
    residual_leak_s), gains residual_integrate_gain x the action and is clamped to +-residual_clamp action units; the
    policy observes it as it observes the filtered one. With integral_gain (trajectory only), the target also carries
    an integral of the measured wrists' pose error: each step the target has stopped at the posture, the weighted DLS
    step that would close the error (no posture pull, no locks) times integral_gain is added to it; it leaks toward 0
    (integral_leak_s) and is clamped to +-integral_clamp rad per joint.
    """

    cfg: IKPostureArmActionCfg

    def __init__(self, cfg: IKPostureArmActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        n = self._num_joints
        robot = self._asset
        self._rest = robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._ik_lower = self._lower + cfg.limit_margin
        self._ik_upper = self._upper - cfg.limit_margin
        self._row_weight = torch.tensor([1.0, 1.0, 1.0, cfg.rot_weight, cfg.rot_weight, cfg.rot_weight],
                                        device=self.device)
        self._eye6 = torch.eye(6, device=self.device) * cfg.ik_damping ** 2
        self._posture = self._rest.clone()  # the posture each step's null-space pull aims for
        self._locked = torch.zeros(self.num_envs, n, dtype=torch.bool, device=self.device)  # last step's, for logging
        self._ik_q = robot.data.joint_pos[:, self._joint_ids].clone()  # trajectory: the target's own last value
        self._traj_v = torch.zeros_like(self._ik_q)  # trajectory_max_vel: the target's velocity (rad/s)
        if cfg.freeze_residual_on_goal and not cfg.trajectory:
            raise ValueError("freeze_residual_on_goal needs trajectory")
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # the residual latched
        self._settled_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._settle_steps = max(1, int(round(cfg.freeze_settle_s / env.step_dt)))
        self._kp_nominal = robot.data.default_joint_stiffness[:, self._joint_ids].clone()
        self._gravity = torch.zeros_like(self._ik_q)  # gravity_offset: the last step's offset (rad)
        if cfg.integral_gain > 0.0 and not cfg.trajectory:
            raise ValueError("integral_gain needs trajectory")
        self._integral = torch.zeros_like(self._ik_q)  # integral_gain: the error's integral (rad)
        self._integral_leak = math.exp(-env.step_dt / cfg.integral_leak_s)
        self._residual_leak = math.exp(-env.step_dt / cfg.residual_leak_s)
        print(f"[INFO] IKPostureArmAction: rot_weight {cfg.rot_weight}, null_gain {cfg.null_gain}, posture_scale"
              f" {cfg.posture_scale} rad, limit_margin {cfg.limit_margin} rad, {cfg.ik_passes} passes"
              + (f"; residual frozen on the goal ({cfg.freeze_pos_tol} m, {cfg.freeze_rot_tol} rad, after"
                 f" {cfg.freeze_settle_s} s)" if cfg.freeze_residual_on_goal else "")
              + ("; gravity offset on the target (URDF masses / nominal kp)" if cfg.gravity_offset else "")
              + (f"; residual integrated (gain {cfg.residual_integrate_gain}, leak {cfg.residual_leak_s} s, clamp"
                 f" {cfg.residual_clamp})" if cfg.residual_integrate else "")
              + (f"; pose-error integral (gain {cfg.integral_gain}, leak {cfg.integral_leak_s} s, clamp"
                 f" {cfg.integral_clamp} rad)" if cfg.integral_gain > 0.0 else ""))

    @property
    def action_dim(self) -> int:
        return (2 if self.cfg.posture_actions else 1) * self._num_joints

    @property
    def posture(self) -> torch.Tensor:
        """(N, joints): the posture the last step aimed for."""
        return self._posture

    @property
    def locked(self) -> torch.Tensor:
        """(N, joints): the joints the last step locked at their bounds."""
        return self._locked

    @property
    def frozen(self) -> torch.Tensor:
        """(N,): the envs whose residual is latched (freeze_residual_on_goal)."""
        return self._frozen

    @property
    def gravity(self) -> torch.Tensor:
        """(N, joints): the gravity offset the last step added to the target (rad; 0 without gravity_offset)."""
        return self._gravity

    def _gravity_offset(self, term, q: torch.Tensor) -> torch.Tensor:
        """(N, joints): minus each arm joint's gravity torque at q over its nominal kp (gravity_offset), else 0."""
        if not self.cfg.gravity_offset:
            return self._gravity
        kin = term._kin
        if kin is None:
            raise ValueError("gravity_offset needs the arm command's ArmKinematics (its seed_solve or collision_check)")
        gravity_b = self._asset.data.projected_gravity_b * abs(self._env.sim.cfg.gravity[2])
        q_arms = torch.stack([q[:, cols] for cols in self._arm_cols], dim=1)
        tau = kin.gravity_torques(q_arms.to(kin.dtype), gravity_b).to(q.dtype)
        for arm, cols in enumerate(self._arm_cols):
            self._gravity[:, cols] = -tau[:, arm] / self._kp_nominal[:, cols]
        return self._gravity

    def process_actions(self, actions: torch.Tensor):
        cfg, n = self.cfg, self._num_joints
        self._raw_actions[:] = actions
        if cfg.residual_integrate:
            stepped = (self._residual_leak * self._residual + cfg.residual_integrate_gain * actions.detach()).clamp(
                -cfg.residual_clamp, cfg.residual_clamp)
        else:
            stepped = self._residual + self._filter_alpha * (actions.detach() - self._residual)
        self._residual = torch.where(self._frozen.unsqueeze(1), self._residual, stepped)
        residual = self._residual[:, :n]
        offset = self._residual[:, n:] if cfg.posture_actions else 0.0
        scale = self._scale[:, :n] if isinstance(self._scale, torch.Tensor) else self._scale
        robot = self._asset
        term = self._env.command_manager.get_term(cfg.command_name)
        gain = (term.goal_ramp if cfg.residual_ramp else 1.0)
        base = self._rest
        if cfg.posture_from_command:
            base = self._rest.clone()
            for arm, cols in enumerate(self._arm_cols):
                base[:, cols] = term.seed_q[:, arm]
        self._posture = torch.maximum(torch.minimum(
            base + cfg.posture_scale * cfg.posture_gain * gain * offset, self._ik_upper), self._ik_lower)
        command = term.believed_b  # (N, arms, 7), wxyz
        root_pos, root_quat = robot.data.root_pos_w, robot.data.root_quat_w
        to_root = matrix_from_quat(quat_inv(root_quat))
        jacobians = robot.root_physx_view.get_jacobians()
        q = robot.data.joint_pos[:, self._joint_ids]
        if cfg.trajectory:
            # the target moves from its own last value toward the posture: no IK
            if cfg.trajectory_max_vel is None:
                ik = self._ik_q + (self._posture - self._ik_q).clamp(-cfg.max_ik_step, cfg.max_ik_step)
            else:
                ik = self._synchronized_step()
            self._ik_q = torch.maximum(torch.minimum(ik, self._upper), self._lower)
            if cfg.integral_gain > 0.0:
                self._update_integral(command, jacobians, to_root, root_pos, root_quat)
            self._apply_bounds(self._ik_q + self._gravity_offset(term, self._ik_q) + self._integral
                               + residual * (scale * cfg.residual_gain * gain))
            if cfg.freeze_residual_on_goal:
                self._update_freeze(term, root_pos, root_quat)
            return
        ik = q.clone()
        for arm, (cols, jac_cols, body, jac_body) in enumerate(
                zip(self._arm_cols, self._jac_cols, self._body_ids, self._jac_body_ids)):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body]
            )
            jac = jacobians[:, jac_body][:, :, jac_cols]  # (N, 6, arm joints), world frame
            jac = torch.cat([torch.bmm(to_root, jac[:, :3]), torch.bmm(to_root, jac[:, 3:])], dim=1)
            pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, command[:, arm, :3], command[:, arm, 3:],
                                                  rot_error_type="axis_angle")
            step = self._step(q[:, cols], jac, torch.cat([pos_err, rot_err], dim=1), cols)
            ik[:, cols] = q[:, cols] + step
        self._ik_q = torch.maximum(torch.minimum(ik, self._upper), self._lower)  # where trajectory would start from
        self._apply_bounds(ik + self._gravity_offset(term, ik) + residual * (scale * cfg.residual_gain * gain))

    def _update_freeze(self, term, root_pos: torch.Tensor, root_quat: torch.Tensor):
        """Latch (class doc) or release each env's residual for the next step."""
        cfg = self.cfg
        arrived = self._arrived()
        self._frozen &= arrived & term.arm_mode  # a new goal moves the posture
        on_goal = term.arm_mode.clone()
        robot = self._asset
        for arm, body in enumerate(self._body_ids):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body])
            on_goal &= torch.norm(ee_pos - term.believed_b[:, arm, :3], dim=-1) < cfg.freeze_pos_tol
            on_goal &= quat_error_magnitude(ee_quat, term.believed_b[:, arm, 3:]) < cfg.freeze_rot_tol
        settling = arrived & on_goal & ~self._frozen
        self._settled_steps = torch.where(settling, self._settled_steps + 1, torch.zeros_like(self._settled_steps))
        self._frozen |= self._settled_steps >= self._settle_steps

    def _arrived(self) -> torch.Tensor:
        """(N,): the trajectory's target has stopped at the posture."""
        return ((self._ik_q - self._posture).abs().amax(dim=1) < 1e-4) & (self._traj_v.abs().amax(dim=1) < 1e-3)

    def _update_integral(self, command, jacobians, to_root, root_pos, root_quat):
        """Leak the pose-error integral, add integral_gain x each arm's weighted DLS step on its measured wrist's error
        where the target has stopped at the posture, clamp (class doc)."""
        cfg, robot = self.cfg, self._asset
        arrived = self._arrived().float().unsqueeze(1)
        self._integral *= self._integral_leak
        for arm, (cols, jac_cols, body, jac_body) in enumerate(
                zip(self._arm_cols, self._jac_cols, self._body_ids, self._jac_body_ids)):
            ee_pos, ee_quat = subtract_frame_transforms(
                root_pos, root_quat, robot.data.body_pos_w[:, body], robot.data.body_quat_w[:, body])
            jac = jacobians[:, jac_body][:, :, jac_cols]  # (N, 6, arm joints), world frame
            jac_w = torch.cat([torch.bmm(to_root, jac[:, :3]), torch.bmm(to_root, jac[:, 3:])], dim=1) \
                * self._row_weight.view(1, 6, 1)
            pos_err, rot_err = compute_pose_error(ee_pos, ee_quat, command[:, arm, :3], command[:, arm, 3:],
                                                  rot_error_type="axis_angle")
            err_w = (torch.cat([pos_err, rot_err], dim=1) * self._row_weight).unsqueeze(2)
            step = (jac_w.transpose(1, 2) @ torch.linalg.solve(jac_w @ jac_w.transpose(1, 2) + self._eye6, err_w))
            self._integral[:, cols] += cfg.integral_gain * step.squeeze(2) * arrived
        self._integral.clamp_(-cfg.integral_clamp, cfg.integral_clamp)

    def _synchronized_step(self) -> torch.Tensor:
        """(N, joints): the trajectory's next target with trajectory_max_vel: per arm, all joints along the straight
        joint-space line to the posture, the fastest at trajectory_max_vel, slowed to stop at the posture and the
        velocity changed by at most trajectory_max_acc (the fastest-changing joint's)."""
        cfg, dt = self.cfg, self._env.step_dt
        vmax, amax = cfg.trajectory_max_vel, cfg.trajectory_max_acc
        ik = self._ik_q.clone()
        for cols in self._arm_cols:
            delta = self._posture[:, cols] - self._ik_q[:, cols]
            dist = delta.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
            cap = (2.0 * amax * dist).sqrt().clamp(max=vmax) if amax is not None else torch.full_like(dist, vmax)
            v_want = delta / dt * (cap * dt / dist).clamp(max=1.0)
            v = self._traj_v[:, cols]
            if amax is not None:
                dv = v_want - v
                v = v + dv * (amax * dt / dv.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)
            else:
                v = v_want
            self._traj_v[:, cols] = v
            ik[:, cols] = self._ik_q[:, cols] + v * dt
        return ik

    def _step(self, q: torch.Tensor, jac: torch.Tensor, error: torch.Tensor, cols: list[int]) -> torch.Tensor:
        """One arm's joint step (N, k): the weighted DLS task step and the null-space posture pull, joints locked at
        their bounds solved out (class doc), capped at max_ik_step."""
        cfg = self.cfg
        jac_w = jac * self._row_weight.view(1, 6, 1)
        err_w = (error * self._row_weight).unsqueeze(2)
        to_posture = self._posture[:, cols] - q
        pull = cfg.null_gain * to_posture
        low, high = self._ik_lower[:, cols], self._ik_upper[:, cols]
        free = torch.ones_like(q, dtype=torch.bool)
        eye_k = torch.eye(q.shape[1], device=self.device)
        for _ in range(cfg.ik_passes):
            f = free.float()
            jf = jac_w * f.unsqueeze(1)
            pinv = jf.transpose(1, 2) @ torch.linalg.inv(jf @ jf.transpose(1, 2) + self._eye6)  # (N, k, 6)
            null = eye_k - pinv @ jf
            step = ((pinv @ err_w).squeeze(2) + (null @ (pull * f).unsqueeze(2)).squeeze(2)) * f
            step = step.clamp(-cfg.max_ik_step, cfg.max_ik_step)
            new = q + step
            hit = ((new < low) & (step < 0)) | ((new > high) & (step > 0))
            if not (hit & free).any():
                break
            free = free & ~hit
        self._locked[:, cols] = ~free
        return step

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._posture[ids] = self._rest[ids]
        self._locked[ids] = False
        self._ik_q[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._traj_v[ids] = 0.0
        self._frozen[ids] = False
        self._settled_steps[ids] = 0
        self._integral[ids] = 0.0


@configclass
class IKPostureArmActionCfg(IKResidualArmActionCfg):
    """IKPostureArmAction's: the residual's settings as IKResidualArmActionCfg's, and the IK's and the posture's."""

    class_type: type = IKPostureArmAction

    rot_weight: float = 0.3
    """The rotation error's and the Jacobian's rotation rows' weight against position's (1 per m)."""
    null_gain: float = 0.05
    """Share of the posture error the null-space pull asks for per policy step."""
    posture_scale: float = 0.5
    """rad of posture offset per action unit."""
    posture_gain: float = 1.0
    """Times posture_scale (0: the posture is the default joint angles whatever the action)."""
    limit_margin: float = 0.05
    """rad inside the target bounds where the IK locks a joint moving toward them."""
    ik_passes: int = 3
    """Solves per step, each with the joints the last one would have taken past their bounds locked."""
    trajectory: bool = False
    """No IK step: the target moves from its own last value straight to the posture (with posture_from_command, the
    goal table's nearest row's joint angles plus the policy's offset), max_ik_step per joint per step. An IK step from
    the measured angles, capped at max_ik_step, can't hold a joint whose sag under gravity needs more PD error than
    the cap: the shoulders stalled 0.3 rad short of the goals' angles, and the IK alone reached 10-17% of goals; the
    goals' table angles commanded directly reached 94%. Also tried, with the posture a true solution: the task step
    with a null-space pull (12.5% reached, the arm 0.93 rad off the posture), with a full-space pull of 0.05-0.2 per
    step (6-9%), going to the posture in joint space and fading the task step in near it (16-17%; the same step cap
    from the measured angles), and stepping from the IK's own last target (integral action; it wound up: 0%)."""
    posture_from_command: bool = False
    """The posture's base is the arm command's seed_q (its table_joints on), not the default joint angles."""
    posture_actions: bool = True
    """The action's posture half; False: the action is the residual alone and the posture its base."""
    trajectory_max_vel: float | None = None
    """rad/s: with trajectory, each arm's joints move together along the straight joint-space line to the posture, the
    fastest at this speed (IKPostureArmAction._synchronized_step); None: each joint capped at max_ik_step per step on
    its own. In MuJoCo (the IK alone on table goals), per-joint caps of 0.05 rad tripped GOLEM's torque e-stop in 46%
    of goals; 0.02 rad (2 rad/s), 1%; synchronized at 2 rad/s, 0-1%, and the links' overlap 24% -> 17% of goals."""
    trajectory_max_acc: float | None = None
    """rad/s^2 of the fastest-changing joint, with trajectory_max_vel (None: no limit). At 10: no e-stops, overlap
    16% of goals."""
    freeze_residual_on_goal: bool = False
    """With trajectory: latch the residual on the goal (class doc). ArmsOnlyV6_s42's policy (11-11, 1 Hz residual)
    kept the settled wrist swaying 2.8-3.9 mm RMS 4-8 s into a goal, its residual moving 0-10 Hz with the command and
    the arm still; the IK alone, 0.07 mm."""
    freeze_pos_tol: float = 0.05
    """m: each wrist's distance from the command (pelvis frame) to count as on the goal."""
    freeze_rot_tol: float = 0.35
    """rad: each wrist's rotation from the command to count as on the goal."""
    freeze_settle_s: float = 0.5
    """s on the goal with the target stopped before the latch: the residual's own settling (1 Hz: 0.16 s)."""
    gravity_offset: bool = False
    """Add each arm joint's gravity torque at the target over its nominal kp to the target (class doc). The IK alone
    (ArmsOnly, pinned, 11-12) held the wrists 26 mm off, 23 mm low, all of it the arm joints' sag (shoulder pitch
    2.1 deg); the policy's residual was the only gravity compensation."""
    residual_integrate: bool = False
    """Integrate the action into the residual instead of low-passing it (class doc; residual_cutoff_hz unused)."""
    residual_integrate_gain: float = 0.02
    """Action units added to the residual per step per action unit (|a| = 1: the clamp in 0.5 s at 100 Hz)."""
    residual_leak_s: float = 5.0
    """s: the integrated residual's leak toward 0."""
    residual_clamp: float = 1.0
    """Action units: the integrated residual's bound (x scale: rad)."""
    integral_gain: float = 0.0
    """Share of the DLS step on the measured pose error added to the target's integral per step (class doc); 0: off."""
    integral_leak_s: float = 5.0
    """s: the integral's leak toward 0."""
    integral_clamp: float = 0.05
    """rad per joint: the integral's bound."""
