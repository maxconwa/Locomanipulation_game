"""The one interface every system under test is driven through: SE(3) wrist commands in, achieved poses out, one
control step at a time, for num_envs robot instances with num_arms arms each, all in lockstep.

A command change is either a goal (the system treats it as a new goal: the arms stack re-solves its seed posture and
plans a trajectory to it) or a nudge (only the pose the controller and policy see changes; the outer-loop interface).
Poses are in the frame the commands are given in (the pinned pelvis's), as se3's (..., 7).
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class StepLog:
    achieved: np.ndarray  # (N, arms, 7): each wrist's pose after the step
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # system-specific state, logged as given


class SteerableArms(Protocol):
    num_envs: int
    num_arms: int
    dt: float  # s per step

    def reset(self) -> StepLog:
        """Start every instance afresh; the first command must then be a goal."""
        ...

    def set_command(self, poses: np.ndarray, goal: np.ndarray) -> None:
        """poses (N, arms, 7) from the next step on; goal (N, arms) bool: a new goal there, a nudge elsewhere."""
        ...

    def step(self) -> StepLog:
        ...
