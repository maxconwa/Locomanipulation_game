"""Poses and pose deltas in numpy, the benchmark's one convention.

A pose is (..., 7): position (m) then a unit quaternion (w, x, y, z), Isaac Lab's order. A delta is (..., 6): a
translation (m) and a rotation vector (rad), both in the frame of the pose they are applied to:

    perturb(c, d) = (p + R d[:3], q * exp(d[3:]))       relative(a, b) = the d with perturb(a, d) = b

Translation and rotation are kept apart (not the SE(3) exponential, whose translation part depends on the rotation),
so a command step of 1 mm along an axis is exactly 1 mm whatever rotation step comes with it.
"""

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """v rotated by q."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    half = 0.5 * angle
    # sin(half) / angle, its limit 0.5 at 0
    scale = np.where(angle > 1e-12, np.sin(half) / np.where(angle > 1e-12, angle, 1.0), 0.5)
    return np.concatenate([np.cos(half), r * scale], axis=-1)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.where(q[..., :1] < 0.0, -q, q)  # the shorter of the two rotations
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    scale = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0)
    return q[..., 1:] * scale


def perturb(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """pose moved by delta, in pose's own frame (module doc)."""
    p, q = pose[..., :3], pose[..., 3:]
    return np.concatenate([p + quat_rotate(q, delta[..., :3]), quat_mul(q, quat_from_rotvec(delta[..., 3:]))], axis=-1)


def relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The delta that takes a to b, in a's frame (module doc)."""
    qa_inv = quat_conj(a[..., 3:])
    return np.concatenate([quat_rotate(qa_inv, b[..., :3] - a[..., :3]),
                           rotvec_from_quat(quat_mul(qa_inv, b[..., 3:]))], axis=-1)


def mean_pose(poses: np.ndarray, axis: int = 0) -> np.ndarray:
    """The mean of poses along axis: positions averaged, rotations averaged as rotation vectors about the first
    (exact enough for the sub-degree spreads of a settled hold)."""
    poses = np.moveaxis(poses, axis, 0)
    ref = poses[0]
    rot = rotvec_from_quat(quat_mul(quat_conj(ref[..., 3:])[None], poses[..., 3:])).mean(axis=0)
    return np.concatenate([poses[..., :3].mean(axis=0), quat_mul(ref[..., 3:], quat_from_rotvec(rot))], axis=-1)


def identity(shape: tuple[int, ...] = ()) -> np.ndarray:
    pose = np.zeros(shape + (7,))
    pose[..., 3] = 1.0
    return pose
