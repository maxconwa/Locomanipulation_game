"""Run a schedule on a SteerableArms: per segment, the settled pose over its last window_s and how flat it was there,
and (trace) every step's pose as a delta from the segment's command."""

from dataclasses import dataclass

import numpy as np

from . import se3
from .interface import SteerableArms
from .schedules import Segment


@dataclass
class RunResult:
    settled: np.ndarray  # (S, N, arms, 7): mean achieved pose over each segment's last window
    drift: np.ndarray  # (S, N, arms, 2): the window's slope, |position| m/s and |rotation| rad/s
    start: np.ndarray  # (S, N, arms, 7): the pose at each segment's first step (the last step of the one before)
    reset: np.ndarray  # (S, N): the instance reset (fell, timed out) during the segment (a system's "reset" extra)
    traces: list[np.ndarray] | None  # per segment (steps, N, arms, 6): achieved relative to the segment's command
    extras: list[dict[str, np.ndarray]] | None  # per segment, the system's extras at its last step
    dt: float


def _np(value) -> np.ndarray:
    """A system's extra as numpy (Isaac's are torch tensors on the GPU)."""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _slope(series: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares slope along axis 0 of (T, ...) samples dt apart."""
    t = (np.arange(len(series)) - (len(series) - 1) / 2) * dt
    return np.tensordot(t, series - series.mean(axis=0), axes=(0, 0)) / (t @ t)


def run(system: SteerableArms, segments: list[Segment], window_s: float = 0.5, trace: bool = False) -> RunResult:
    dt = system.dt
    log = system.reset()
    last = log.achieved
    settled, drift, start, resets = [], [], [], []
    traces, extras = ([], []) if trace else (None, None)
    for seg in segments:
        steps = int(round(seg.duration_s / dt))
        window = min(steps, max(2, int(round(window_s / dt))))
        system.set_command(seg.poses, seg.goal)
        start.append(last)
        rel = np.empty((steps,) + seg.poses.shape[:2] + (6,))
        tail = []
        reset = None
        for k in range(steps):
            log = system.step()
            rel[k] = se3.relative(seg.poses, log.achieved)
            if k >= steps - window:
                tail.append(log.achieved)
            if "reset" in log.extras:
                reset = log.extras["reset"] if reset is None else reset | log.extras["reset"]
        last = log.achieved
        settled.append(se3.mean_pose(np.stack(tail)))
        slope = _slope(rel[-window:], dt)
        drift.append(np.stack([np.linalg.norm(slope[..., :3], axis=-1), np.linalg.norm(slope[..., 3:], axis=-1)], -1))
        resets.append(np.zeros(seg.poses.shape[0], bool) if reset is None else _np(reset).astype(bool))
        if trace:
            traces.append(rel.astype(np.float32))
            extras.append({key: _np(value) for key, value in log.extras.items()})
    return RunResult(np.stack(settled), np.stack(drift), np.stack(start), np.stack(resets), traces, extras, dt)
