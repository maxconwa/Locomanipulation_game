"""Trial schedules: pure functions of the bases, the config and a seed, so a run can be rebuilt from its record.

A schedule is a list of Segments, each one command per (instance, arm) held for duration_s, run in lockstep. Every
instance and arm gets the same number of segments of the same durations; what differs is which axis and scale each
one steps, in an order shuffled per instance and per arm (the arms' trials are independent).
"""

from dataclasses import dataclass, field

import numpy as np

from . import se3


@dataclass
class Segment:
    duration_s: float
    poses: np.ndarray  # (N, arms, 7)
    goal: np.ndarray  # (N, arms) bool: a goal there, else a nudge
    tags: dict[str, np.ndarray] = field(default_factory=dict)  # (N, arms) labels


@dataclass
class StepPlan:
    """step_schedule's trials: item m's segments are 1 + 4m + phase, phase 0 the step out by +magnitude along axis,
    1 back to the base, 2 out by -magnitude, 3 back; segment 0 is the first approach to the base."""
    segments: list[Segment]
    bases: np.ndarray  # (N, arms, 7)
    axis: np.ndarray  # (M, N, arms): 0-2 translation x, y, z; 3-5 rotation about x, y, z (the base's frame)
    scale: np.ndarray  # (M, N, arms): index into scales, -1 for a sham (no step)
    magnitude: np.ndarray  # (M, N, arms): m or rad, 0 for a sham
    scales: np.ndarray  # (S, 2): (translation m, rotation rad) per scale index


def step_schedule(bases: np.ndarray, scales: np.ndarray, repeats: int, sham_repeats: int, settle_s: float,
                  approach_s: float, goal_mode: bool, seed: int) -> StepPlan:
    """Each axis at each scale, repeats times, plus sham_repeats shams per axis: per trial, + step, back, - step, back
    (StepPlan). The central difference gives the local gain, the backs from either side the hysteresis, the shams the
    no-step distribution."""
    rng = np.random.default_rng(seed)
    scales = np.asarray(scales, dtype=float)
    N, arms = bases.shape[:2]
    items = [(k, s) for k in range(6) for s in range(len(scales)) for _ in range(repeats)]
    items += [(k, -1) for k in range(6) for _ in range(sham_repeats)]
    items = np.array(items)  # (M, 2)
    M = len(items)
    order = np.stack([np.stack([rng.permutation(M) for _ in range(arms)], axis=1) for _ in range(N)], axis=1)
    axis, scale = items[order, 0], items[order, 1]  # (M, N, arms)
    per_axis = np.where(axis[..., None] < 3, scales[None, None, None, :, 0], scales[None, None, None, :, 1])
    magnitude = np.where(scale >= 0, np.take_along_axis(per_axis, np.maximum(scale, 0)[..., None], -1)[..., 0], 0.0)
    goal = np.full((N, arms), goal_mode)
    segments = [Segment(approach_s, bases.copy(), np.ones((N, arms), bool), {"phase": np.full((N, arms), -1)})]
    for m in range(M):
        for phase, sign in enumerate((1.0, 0.0, -1.0, 0.0)):
            delta = np.zeros((N, arms, 6))
            np.put_along_axis(delta, axis[m][..., None], (sign * magnitude[m])[..., None], -1)
            segments.append(Segment(settle_s, se3.perturb(bases, delta), goal.copy(),
                                    {"phase": np.full((N, arms), phase), "item": np.full((N, arms), m)}))
    return StepPlan(segments, bases.copy(), axis, scale, magnitude, scales)


@dataclass
class RepeatPlan:
    """repeat_schedule's trials: segment 2k approaches starts[k], 2k + 1 the base after it."""
    segments: list[Segment]
    bases: np.ndarray
    starts: np.ndarray  # (K, N, arms, 7)


def repeat_schedule(bases: np.ndarray, starts: np.ndarray, settle_s: float, approach_s: float) -> RepeatPlan:
    """The base reached K times as a goal, each time from starts[k] (all equal: a matched start; different: varied)."""
    N, arms = bases.shape[:2]
    goal = np.ones((N, arms), bool)
    segments = []
    for k, start in enumerate(starts):
        segments.append(Segment(approach_s, start.copy(), goal.copy(), {"repeat": np.full((N, arms), k)}))
        segments.append(Segment(settle_s, bases.copy(), goal.copy(), {"repeat": np.full((N, arms), k)}))
    return RepeatPlan(segments, bases.copy(), np.asarray(starts).copy())
