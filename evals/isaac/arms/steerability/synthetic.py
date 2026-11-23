"""A synthetic SteerableArms whose steerability is known in closed form: the estimators' test bench.

Per instance and arm, with x the command as a delta from the last goal's pose (se3.relative):
- backlash: an internal command x_b follows x with a play of backlash (per axis, full width): hysteresis of that width,
  and no response to a reversal smaller than it;
- the static map: y* = gain @ x_b + bias(command position) + offset + drift * t, where bias_j = bias_amp_j
  sin(2 pi command_x / bias_wavelength + j), so the local Jacobian is gain plus bias's gradient along x, and offset is
  drawn N(0, goal_std) at every goal (repeatability from one approach to the next);
- dynamics: y follows y* as a second-order system (natural frequency omega, damping zeta);
- the output: the last goal's pose moved by y plus white noise N(0, white_std) every step.
"""

from dataclasses import dataclass, field

import numpy as np

from . import se3
from .interface import StepLog


def _six(value) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), (6,)).copy()


@dataclass
class SyntheticParams:
    gain: np.ndarray = field(default_factory=lambda: np.eye(6))
    backlash: float | np.ndarray = 0.0
    bias_amp: float | np.ndarray = 0.0
    bias_wavelength: float = 0.1  # m
    goal_std: float | np.ndarray = 0.0
    white_std: float | np.ndarray = 0.0
    drift: float | np.ndarray = 0.0  # per s
    omega: float = 2 * np.pi * 2.0  # rad/s
    zeta: float = 1.0


class SyntheticArms:
    def __init__(self, num_envs: int, params: SyntheticParams, num_arms: int = 2, dt: float = 0.01, seed: int = 0):
        self.num_envs, self.num_arms, self.dt = num_envs, num_arms, dt
        self.p = params
        self.gain = np.asarray(params.gain, dtype=float)
        self.backlash, self.bias_amp = _six(params.backlash), _six(params.bias_amp)
        self.goal_std, self.white_std, self.drift = _six(params.goal_std), _six(params.white_std), _six(params.drift)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def bias(self, position: np.ndarray) -> np.ndarray:
        """(..., 6) at command positions (..., 3)."""
        phase = 2 * np.pi * position[..., :1] / self.p.bias_wavelength + np.arange(6)
        return self.bias_amp * np.sin(phase)

    def bias_gradient_x(self, position: np.ndarray) -> np.ndarray:
        """(..., 6): d bias / d command x."""
        phase = 2 * np.pi * position[..., :1] / self.p.bias_wavelength + np.arange(6)
        return self.bias_amp * 2 * np.pi / self.p.bias_wavelength * np.cos(phase)

    def reset(self) -> StepLog:
        shape = (self.num_envs, self.num_arms)
        self.goal_pose = se3.identity(shape)
        self.command = se3.identity(shape)
        self.x_b, self.y, self.y_dot, self.offset = (np.zeros(shape + (6,)) for _ in range(4))
        self.t = 0.0
        return StepLog(self.goal_pose.copy())

    def set_command(self, poses: np.ndarray, goal: np.ndarray) -> None:
        goal = np.asarray(goal, dtype=bool)
        if goal.any():
            achieved = se3.perturb(self.goal_pose, self.y)
            self.goal_pose[goal] = poses[goal]
            # the motion continues from where the wrist is, now measured from the new goal
            self.y[goal] = se3.relative(self.goal_pose, achieved)[goal]
            self.x_b[goal] = 0.0
            self.offset[goal] = self.rng.normal(size=(int(goal.sum()), 6)) * self.goal_std
        self.command = poses.copy()

    def step(self) -> StepLog:
        x = se3.relative(self.goal_pose, self.command)
        half = 0.5 * self.backlash
        self.x_b = np.clip(self.x_b, x - half, x + half)
        target = (self.x_b @ self.gain.T + self.bias(self.command[..., :3]) + self.offset + self.drift * self.t)
        w, z = self.p.omega, self.p.zeta
        self.y_dot += (w * w * (target - self.y) - 2 * z * w * self.y_dot) * self.dt
        self.y += self.y_dot * self.dt
        self.t += self.dt
        noise = self.rng.normal(size=self.y.shape) * self.white_std
        return StepLog(se3.perturb(self.goal_pose, self.y + noise))
