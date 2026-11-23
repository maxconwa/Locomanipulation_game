"""Procedurally generated stand-in motion clips in the pretraining's ``.npz`` format (no mocap data in tests)."""

from pathlib import Path

import numpy as np

from pretraining.constants import DEFAULT_JOINT_POS, DEFAULT_ROOT_HEIGHT, JOINT_NAMES, JOINT_RANGES

FPS = 50.0


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=-1)


def write_clips(dir: Path, n: int = 3, frames: int = 120, seed: int = 0) -> list[Path]:
    """Write ``n`` clips of ``frames`` frames to ``dir``: joint sinusoids about the default pose inside the joint
    ranges, a pelvis that sways in x/y and yaws; velocities are the analytic derivatives. Returns the paths."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / FPS
    default = np.array([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES])
    lo, hi = np.array([JOINT_RANGES[j] for j in JOINT_NAMES]).T
    room = np.minimum(default - lo, hi - default)  # the largest swing that stays inside the range
    paths = []
    for i in range(n):
        amp = room * rng.uniform(0.2, 0.6, 27)
        freq = rng.uniform(0.3, 1.0, 27) * 2 * np.pi
        phase = rng.uniform(0, 2 * np.pi, 27)
        arg = freq * t[:, None] + phase
        joint_pos = default + amp * np.sin(arg)
        joint_vel = amp * freq * np.cos(arg)
        sway, w = rng.uniform(0.02, 0.08), rng.uniform(0.5, 1.5) * 2 * np.pi
        yaw_rate = rng.uniform(-0.5, 0.5)
        root_pos = np.stack([sway * np.sin(w * t), sway * np.cos(w * t), DEFAULT_ROOT_HEIGHT + 0 * t], axis=-1)
        root_lin_vel = np.stack([sway * w * np.cos(w * t), -sway * w * np.sin(w * t), 0 * t], axis=-1)
        root_quat = _yaw_quat(yaw_rate * t)
        root_ang_vel = np.stack([0 * t, 0 * t, yaw_rate + 0 * t], axis=-1)
        path = dir / f"standin_{i:03d}.npz"
        np.savez(
            path,
            fps=np.array([FPS]),
            joint_names=np.array(JOINT_NAMES),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            root_pos_w=root_pos.astype(np.float32),
            root_quat_w=root_quat.astype(np.float32),
            root_lin_vel_w=root_lin_vel.astype(np.float32),
            root_ang_vel_w=root_ang_vel.astype(np.float32),
        )
        paths.append(path)
    return paths
