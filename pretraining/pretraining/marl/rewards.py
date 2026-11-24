# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/rewards.py.
# Changes: imports; JointLimits.for_joints reads pretraining.constants (KP, KD, EFFORT, ESTOP_LIMITS); stage 1 only
# (command_rewards, _penalties, foot_clearance_cost and command_errors left out, and max_step_rewards without its
# stage-2 ``command`` argument and branch); rot6d_from_matrix, ee_in_pelvis and ee_features moved here from
# src/h12_train/marl/goals.py (stage 2's module, not ported), unchanged.
"""Differentiable per-agent rewards on the world model's predicted state.

Every reward is ``dt * sum(terms)`` like ``supermimic.utils.tracking_reward``; kernels are
``weight * exp(-err / std^2)`` with ``err`` a squared error averaged over bodies. Each
agent's reward also carries its estop penalties (``estop_penalties``) on its own joints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pretraining.marl.config import (
    AGENTS,
    GRASP_OFFSET,
    JOINT_SLICES,
    LEG_BODIES,
    PELVIS,
    UPPER_BODIES,
    WRISTS,
    EstopPenaltyCfg,
    ImitationRewardCfg,
    Kernel,
)
from pretraining.math import matrix_from_quat
from pretraining.supermimic.utils.helpers import rotation_error_magnitude
from pretraining.supermimic.utils.tracking_reward import _reanchor_targets, tracking_reward


def rot6d_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    """Return the first two columns of each rotation matrix, row-major, (..., 6)."""
    return mat[..., :2].flatten(-2)


def ee_in_pelvis(body_pos: torch.Tensor, body_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both tool frames' positions (..., 2, 3) and rotations (..., 2, 3, 3) in the pelvis frame.

    ``body_pos`` (..., B, 3) and ``body_rot`` (..., B, 3, 3) are the tracked bodies in any
    common frame (world for the robot or the reference).
    """
    rp = body_rot[..., PELVIS, :, :]
    pp = body_pos[..., PELVIS, :]
    rw = body_rot[..., list(WRISTS), :, :]
    pw = body_pos[..., list(WRISTS), :]
    offset = torch.tensor(GRASP_OFFSET, device=body_pos.device, dtype=body_pos.dtype)
    p_ee = pw + (rw @ offset)
    rel_p = (rp.transpose(-1, -2).unsqueeze(-3) @ (p_ee - pp.unsqueeze(-2)).unsqueeze(-1))[..., 0]
    rel_r = rp.transpose(-1, -2).unsqueeze(-3) @ rw
    return rel_p, rel_r


def ee_features(rel_p: torch.Tensor, rel_r: torch.Tensor) -> torch.Tensor:
    """Return tool-frame poses as position + rot6d, (..., 2, 9)."""
    return torch.cat([rel_p, rot6d_from_matrix(rel_r)], dim=-1)


@dataclass(frozen=True)
class JointLimits:
    """Per-joint tensors (policy joint order) of the PD gains and golem_safety's estop limits."""

    kp: torch.Tensor  # N m/rad
    kd: torch.Tensor  # N m s/rad
    dq: torch.Tensor  # rad/s, estop |dq|
    tau: torch.Tensor  # N m, estop |tau|
    effort: torch.Tensor  # N m, the actuator's force limit
    q_low: torch.Tensor  # rad, estop position range
    q_high: torch.Tensor

    @staticmethod
    def for_joints(names: Sequence[str], device, plant=None) -> JointLimits:
        """Return the limits of joints ``names``: the gains and effort limits of ``plant`` (a ``plants.Plant``;
        None: ``pretraining.constants``', the pretrain plant's), golem_safety's estop limits whatever the plant."""
        from pretraining.constants import EFFORT, ESTOP_LIMITS, KD, KP

        if plant is not None:
            KP, KD, EFFORT = plant.kp, plant.kd, plant.effort  # noqa: N806

        def t(values):
            """Return ``values`` as a float tensor on ``device``."""
            return torch.tensor(values, dtype=torch.float32, device=device)

        return JointLimits(
            kp=t([KP[n] for n in names]),
            kd=t([KD[n] for n in names]),
            dq=t([ESTOP_LIMITS[n].dq for n in names]),
            tau=t([ESTOP_LIMITS[n].tau for n in names]),
            effort=t([EFFORT[n] for n in names]),
            q_low=t([ESTOP_LIMITS[n].q_low for n in names]),
            q_high=t([ESTOP_LIMITS[n].q_high for n in names]),
        )


def estop_ratios(
    limits: JointLimits,
    q: torch.Tensor,
    dq: torch.Tensor,
    dq_next: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``|dq_next| / dq_estop`` and the PD torque demand ``|kp (target - q) - kd dq| / tau_estop``, per joint.

    ``q``, ``dq`` are the state before the step (the first substep's), ``dq_next`` the
    state after it, ``target`` the clamped position target of the step (rad).
    """
    demand = limits.kp * (target - q) - limits.kd * dq
    return dq_next.abs() / limits.dq, demand.abs() / limits.tau


def estop_position_depth(limits: JointLimits, q: torch.Tensor, margin: float) -> torch.Tensor:
    """Return, per joint, how deep ``q`` is in the ``margin`` rad band before its nearer estop position limit.

    0 when the joint is more than ``margin`` inside the range, 1 at the limit, above 1 past it.
    """
    room = torch.minimum(q - limits.q_low, limits.q_high - q)
    return (margin - room).clamp(min=0) / margin


def estop_penalties(
    cfg: EstopPenaltyCfg,
    vel_ratio: torch.Tensor,
    tau_ratio: torch.Tensor,
    pos_depth: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each agent's estop penalty (B, 2), on its own joints: weighted excess past the margin.

    ``pos_depth`` is ``estop_position_depth`` of the predicted next positions (None: no position term).
    """
    vel = (vel_ratio - cfg.margin).clamp(min=0)
    tau = (tau_ratio - cfg.margin).clamp(min=0)
    cost = cfg.velocity * vel + cfg.torque * tau
    if pos_depth is not None and cfg.position != 0.0:
        cost = cost + cfg.position * pos_depth
    return torch.stack([cost[..., JOINT_SLICES[a]].sum(-1) for a in AGENTS], dim=-1)


def _k(err: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    """Return ``weight * exp(-err / std^2)``."""
    return kernel.weight * torch.exp(-err / kernel.std**2)


def _in_pelvis(pos: torch.Tensor, rot: torch.Tensor, ids: tuple[int, ...]):
    """Return bodies ``ids`` in the pelvis frame: positions (..., n, 3), rotations (..., n, 3, 3)."""
    rp_t = rot[..., PELVIS, :, :].transpose(-1, -2).unsqueeze(-3)
    rel_p = (rp_t @ (pos[..., list(ids), :] - pos[..., PELVIS : PELVIS + 1, :]).unsqueeze(-1))[..., 0]
    return rel_p, rp_t @ rot[..., list(ids), :, :]


def imitation_rewards(
    cfg: ImitationRewardCfg,
    aux_state: torch.Tensor,
    pos: torch.Tensor,
    rot: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    joint_state: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    target_lin_vel: torch.Tensor,
    target_ang_vel: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    dt: float,
    estop: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(B, 3)``: the legs', arms' and team (SuperMimic tracking) rewards.

    ``pos``/``rot``/``lin_vel``/``ang_vel`` are the predicted tracked bodies (world), the
    ``target_*`` the reference's; ``aux_state`` is the predicted root pose; ``estop``
    (B, 2) each agent's ``estop_penalties``, added to its own reward.
    """
    team = tracking_reward(
        aux_state,
        pos,
        rot,
        lin_vel,
        ang_vel,
        joint_state,
        action,
        last_action,
        target_pos,
        target_quat,
        target_lin_vel,
        target_ang_vel,
        soft_joint_pos_limits,
        dt,
    )
    tgt_pos, tgt_rot = _reanchor_targets(target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7])
    ref_rot = matrix_from_quat(target_quat)
    legs = list(LEG_BODIES)
    legs_r = (
        _k(
            torch.sum((pos[..., PELVIS, :] - target_pos[..., PELVIS, :]) ** 2, -1),
            cfg.legs_root_pos,
        )
        + _k(
            rotation_error_magnitude(rot[..., PELVIS, :, :], ref_rot[..., PELVIS, :, :]) ** 2,
            cfg.legs_root_ori,
        )
        + _k(
            torch.sum((pos[..., legs, :] - tgt_pos[..., legs, :]) ** 2, -1).mean(-1),
            cfg.legs_body_pos,
        )
        + _k(
            (rotation_error_magnitude(rot[..., legs, :, :], tgt_rot[..., legs, :, :]) ** 2).mean(-1),
            cfg.legs_body_ori,
        )
        + _k(
            torch.sum((lin_vel[..., legs, :] - target_lin_vel[..., legs, :]) ** 2, -1).mean(-1),
            cfg.legs_body_lin_vel,
        )
        + _k(
            torch.sum((ang_vel[..., legs, :] - target_ang_vel[..., legs, :]) ** 2, -1).mean(-1),
            cfg.legs_body_ang_vel,
        )
    )
    rob_p, rob_r = _in_pelvis(pos, rot, UPPER_BODIES)
    ref_p, ref_r = _in_pelvis(target_pos, ref_rot, UPPER_BODIES)
    rob_ee_p, rob_ee_r = ee_in_pelvis(pos, rot)
    ref_ee_p, ref_ee_r = ee_in_pelvis(target_pos, ref_rot)
    arms_r = (
        _k(torch.sum((rob_p - ref_p) ** 2, -1).mean(-1), cfg.arms_body_pos)
        + _k((rotation_error_magnitude(rob_r, ref_r) ** 2).mean(-1), cfg.arms_body_ori)
        + _k(torch.sum((rob_ee_p - ref_ee_p) ** 2, -1).mean(-1), cfg.arms_ee_pos)
        + _k((rotation_error_magnitude(rob_ee_r, ref_ee_r) ** 2).mean(-1), cfg.arms_ee_ori)
    )
    if estop is not None:
        legs_r = legs_r + estop[..., 0]
        arms_r = arms_r + estop[..., 1]
    return torch.stack([dt * legs_r, dt * arms_r, team], dim=-1)


def max_step_rewards(imitation: ImitationRewardCfg, dt: float) -> torch.Tensor:
    """Return the analytic per-step reward ceiling of each agent's return (own + team), (2,)."""
    team = 0.5 + 0.5 + 4 * 1.0
    legs = sum(getattr(imitation, f).weight for f in imitation.__dataclass_fields__ if f.startswith("legs"))
    arms = sum(getattr(imitation, f).weight for f in imitation.__dataclass_fields__ if f.startswith("arms"))
    return dt * torch.tensor([legs + team, arms + team])
