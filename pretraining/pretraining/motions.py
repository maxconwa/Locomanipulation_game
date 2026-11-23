"""Reference motions: checks a motion directory and loads its clips, concatenated, with the tracked bodies' poses
and velocities computed by forward kinematics at load.

``MotionLoader`` keeps the attributes of mjlab's (mjlab_research, commit 6ea3e57,
``src/mjlab/tasks/tracking/mdp/commands.py``) and loads ``.npz`` clips the same way (sorted, concatenated, cumulative
``motion_ends``); the fps and joint-name checks follow h12_train's ``scripts/motion_files.py`` (this repository's own
code, branch ``supermimic``, commit b175f1c). Importable without Isaac.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pretraining.constants import DECIMATION, JOINT_NAMES, SIM_DT
from pretraining.kinematics import ForwardKinematics

FPS = 1.0 / (SIM_DT * DECIMATION)  # one motion frame per policy step: 50 Hz
_FIELDS = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")


def check_motion_dir(path: str | Path) -> list[Path]:
    """The ``.npz`` clips of ``path``, sorted; ValueError naming the path or file if the directory is missing or
    empty, or a clip is not at 50 Hz or its ``joint_names`` are not ``JOINT_NAMES``."""
    path = Path(path).expanduser()
    if not path.is_dir():
        raise ValueError(f"motion directory {path} does not exist")
    files = sorted(path.glob("*.npz"))
    if not files:
        raise ValueError(f"motion directory {path} holds no .npz clips")
    for f in files:
        _check_clip(f)
    return files


def _check_clip(f: Path) -> None:
    with np.load(f) as clip:
        fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
        names = [str(n) for n in clip["joint_names"]]
    if abs(fps - FPS) > 1e-6 * FPS:
        raise ValueError(f"motion {f}: {fps:g} Hz, not the env's {FPS:g} Hz")
    if names != list(JOINT_NAMES):
        raise ValueError(f"motion {f}: joint_names {names} are not the robot's JOINT_NAMES {list(JOINT_NAMES)}")


class MotionLoader:
    """Clips of ``motion_dir`` (or the given files), concatenated per frame on ``device``, with the world poses and
    velocities of ``fk``'s bodies. ``motion_starts`` / ``motion_ends`` delimit the clips (end exclusive)."""

    def __init__(self, motion_dir: str | Path | list[Path], fk: ForwardKinematics, device: str = "cpu") -> None:
        if isinstance(motion_dir, list):
            files = [Path(f) for f in motion_dir]
            for f in files:
                _check_clip(f)
        else:
            files = check_motion_dir(motion_dir)
        clips = [dict(np.load(f)) for f in files]

        def field(name: str) -> torch.Tensor:
            arr = np.concatenate([clip[name] for clip in clips])
            return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

        self.files = files
        self.fps = FPS
        self.fk = fk
        self.joint_pos, self.joint_vel, self.root_pos_w, self.root_quat_w, self.root_lin_vel_w, self.root_ang_vel_w = (
            field(name) for name in _FIELDS)
        self.time_step_total = self.joint_pos.shape[0]
        lengths = torch.tensor([clip["joint_pos"].shape[0] for clip in clips], device=device)
        self.motion_ends = torch.cumsum(lengths, 0)
        self.motion_starts = self.motion_ends - lengths
        self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w = fk(
            self.root_pos_w, self.root_quat_w, self.joint_pos, self.root_lin_vel_w, self.root_ang_vel_w,
            self.joint_vel)
