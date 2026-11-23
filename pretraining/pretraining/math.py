"""Quaternion and frame helpers (wxyz) under the names of ``mjlab.utils.lab_api.math``, in plain torch.

Ported from mjlab (HybridRobotics' fork mjlab_research, commit 6ea3e57) ``src/mjlab/utils/lab_api/math.py``, itself
Isaac Lab's ``isaaclab/utils/math.py`` (Copyright (c) 2022-2025, The Isaac Lab Project Developers, BSD-3-Clause).
``isaaclab.utils.math`` imports ``pxr``, which needs the running app, so the helpers the ported files use are restated
here with the same names and conventions; this module imports without Isaac. Shapes are ``(..., 4)`` / ``(..., 3)``.
"""

from __future__ import annotations

import torch


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product; the shapes must match, as Isaac Lab's."""
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}.")
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vec)
    return vec + quat[..., :1] * t + torch.linalg.cross(xyz, t)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    t = 2 * torch.linalg.cross(xyz, vec)
    return vec - quat[..., :1] * t + torch.linalg.cross(xyz, t)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    theta = (angle / 2).unsqueeze(-1)
    return normalize(torch.cat([theta.cos(), normalize(axis) * theta.sin()], dim=-1))


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dim=-1,
    )


def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    half = torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])
    angle = 2.0 * half
    ratio = torch.where(angle.abs() > eps, torch.sin(half) / angle, 0.5 - angle * angle / 48)
    return quat[..., 1:4] / ratio.unsqueeze(-1)


def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2)))


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.norm(quat_box_minus(q1, q2), dim=-1)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    zero = torch.zeros_like(yaw)
    return normalize(torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1))


def combine_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t12: torch.Tensor | None = None, q12: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    q02 = quat_mul(q01, q12) if q12 is not None else q01
    t02 = t01 + quat_apply(q01, t12) if t12 is not None else t01
    return t02, q02


def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    return t12, q12


def sample_uniform(
    lower: torch.Tensor | float, upper: torch.Tensor | float, size: int | tuple[int, ...], device: str | torch.device
) -> torch.Tensor:
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (upper - lower) + lower
