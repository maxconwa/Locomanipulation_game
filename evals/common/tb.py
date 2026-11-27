"""A run's TensorBoard scalars, with each event's wall time: the training curves and the wall-clock axis.

skrl logs on the env-policy-step axis into the run folder. A run's start is its first event's wall time; a
checkpoint's wall-clock hours are those of the first event at or after its step.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# tags present in every run, used to place checkpoints in wall-clock time
CLOCK_TAGS = ("Reward / Total reward (mean)", "Episode / Total timesteps (mean)", "Reward / Instantaneous reward (mean)")


@lru_cache(maxsize=32)
def _accumulator(run_dir: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def tags(run_dir: str | Path) -> set[str]:
    try:
        return set(_accumulator(str(run_dir)).Tags()["scalars"])
    except Exception:  # noqa: BLE001  (no events, a corrupt file)
        return set()


def scalars(run_dir: str | Path, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(steps, values, wall times) of one tag, or None."""
    if tag not in tags(run_dir):
        return None
    ev = _accumulator(str(run_dir)).Scalars(tag)
    return (np.array([e.step for e in ev], float), np.array([e.value for e in ev], float),
            np.array([e.wall_time for e in ev], float))


def start_time(run_dir: str | Path) -> float | None:
    times = [s[2].min() for t in CLOCK_TAGS if (s := scalars(run_dir, t)) is not None and len(s[2])]
    return min(times) if times else None


def hours_at(run_dir: str | Path, steps) -> dict[int, float]:
    """{step: wall-clock hours since the run started} for the steps the events reach."""
    t0 = start_time(run_dir)
    clock = next((s for t in CLOCK_TAGS if (s := scalars(run_dir, t)) is not None), None)
    if t0 is None or clock is None:
        return {}
    x, _, w = clock
    out = {}
    for step in steps:
        i = np.searchsorted(x, step)
        if i < len(x):
            out[int(step)] = float((w[i] - t0) / 3600.0)
    return out


def curve(run_dir: str | Path, tag: str, smooth: int = 25, hours: bool = False):
    """(x, y) smoothed by a moving average over `smooth` points; x = env steps, or hours since the start."""
    s = scalars(run_dir, tag)
    if s is None:
        return None
    x, y, w = s
    ok = np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    if hours:
        t0 = start_time(run_dir)
        x = (w - (t0 if t0 is not None else w.min())) / 3600.0
    if len(y) > smooth:
        k = np.ones(smooth) / smooth
        y, x = np.convolve(y, k, mode="valid"), x[smooth - 1:]
    return x, y
