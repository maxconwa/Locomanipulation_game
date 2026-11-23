"""Reference motions: checks a motion directory and loads its clips, concatenated; the tracked bodies' poses and
velocities are computed by forward kinematics on demand, for the frames asked for (the full dataset's bodies would
not fit in GPU memory).

``MotionLoader`` keeps the attributes and ``body_state`` of mjlab's (mjlab_research, commit 6ea3e57,
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
        lengths = [clip["joint_pos"].shape[0] for clip in clips]

        def field(name: str) -> torch.Tensor:  # assembled on the CPU, moved to the device once
            arr = np.concatenate([clip[name] for clip in clips]).astype(np.float32, copy=False)
            return torch.from_numpy(arr).to(device=device)

        self.files = files
        self.fps = FPS
        self.fk = fk
        self.joint_pos, self.joint_vel, self.root_pos_w, self.root_quat_w, self.root_lin_vel_w, self.root_ang_vel_w = (
            field(name) for name in _FIELDS)
        del clips
        self.time_step_total = self.joint_pos.shape[0]
        lengths = torch.tensor(lengths, device=device)
        self.motion_ends = torch.cumsum(lengths, 0)
        self.motion_starts = self.motion_ends - lengths
        bodies = len(fk.body_names)
        self.body_pos_w = LazyBodyField(self, 0, "body_pos_w", (self.time_step_total, bodies, 3))
        self.body_quat_w = LazyBodyField(self, 1, "body_quat_w", (self.time_step_total, bodies, 4))
        self.body_lin_vel_w = LazyBodyField(self, 2, "body_lin_vel_w", (self.time_step_total, bodies, 3))
        self.body_ang_vel_w = LazyBodyField(self, 3, "body_ang_vel_w", (self.time_step_total, bodies, 3))

    def body_state(self, frames: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World positions, wxyz quaternions, linear and angular velocities of the
        tracked bodies at ``frames``, without env origins."""
        return self.fk(
            self.root_pos_w[frames],
            self.root_quat_w[frames],
            self.joint_pos[frames],
            self.root_lin_vel_w[frames],
            self.root_ang_vel_w[frames],
            self.joint_vel[frames],
        )


class LazyBodyField:
    """One of ``body_state``'s outputs, indexed as the precomputed ``(frames, bodies, d)`` tensor would be, but
    computed by forward kinematics for the indexed frames only. Supports ``[frames]`` and ``[frames, body]`` with
    ``frames`` an int or an integer tensor of any shape and ``body`` an int or a 0-d integer tensor; anything else
    raises TypeError. Every indexing runs FK: callers reading several fields on the same frames call ``body_state``
    once instead."""

    def __init__(self, motion: MotionLoader, output: int, name: str, shape: tuple[int, int, int]):
        self._motion, self._output, self._name, self.shape = motion, output, name, torch.Size(shape)

    def __getitem__(self, key) -> torch.Tensor:
        frames, body = (key if isinstance(key, tuple) and len(key) == 2 else (key, None))
        if isinstance(key, tuple) and len(key) != 2:
            self._refuse(key)
        if not _is_index(frames) or (body is not None and not (_is_index(body) and torch.as_tensor(body).dim() == 0)):
            self._refuse(key)
        value = self._motion.body_state(frames)[self._output]
        return value if body is None else value[..., body, :]

    def _refuse(self, key):
        raise TypeError(f"motion.{self._name}[{key!r}]: only [frames] and [frames, body] are computed lazily, with "
                        "integer frame indices (int or integer tensor) and a single body index")


def _is_index(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    return isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dtype != torch.bool and not x.is_complex()
