"""The outer loop around a SteerableArms (design doc, metric 7): a proportional pose corrector that reads a simulated
sensor and nudges the command.

Each correction: measured = the achieved pose latency_s ago, moved by Gaussian noise (noise_pos m, noise_rot rad per
axis); command <- command moved by gain x relative(measured, target). Look-then-move waits settle_s between
corrections; continuous corrects every 1 / rate_hz s.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import se3
from .interface import SteerableArms


@dataclass
class OuterLoop:
    gain: float = 1.0
    look_then_move: bool = True
    settle_s: float = 3.0  # look-then-move: wait between corrections
    rate_hz: float = 10.0  # continuous: corrections per second
    latency_s: float = 0.0
    noise_pos: float = 0.0  # m, per axis
    noise_rot: float = 0.0  # rad, per axis
    corrections: int = 10


def run_closed_loop(system: SteerableArms, targets: np.ndarray, loop: OuterLoop, approach_s: float, seed: int
                    ) -> dict[str, np.ndarray]:
    """targets (N, arms, 7), commanded as goals first and held approach_s. Returns error (corrections + 1, N, arms,
    6): the true error relative(target, achieved) just before each correction and after the last period; and command
    (same shape): each correction's command as a delta from the target."""
    rng = np.random.default_rng(seed)
    dt = system.dt
    system.reset()
    goal = np.ones(targets.shape[:2], bool)
    system.set_command(targets, goal)
    log = None
    for _ in range(int(round(approach_s / dt))):
        log = system.step()
    period = int(round((loop.settle_s if loop.look_then_move else 1.0 / loop.rate_hz) / dt))
    lag = int(round(loop.latency_s / dt))
    history = deque([log.achieved], maxlen=lag + 1)
    command = targets.copy()
    errors, commands = [], []
    noise_std = np.array([loop.noise_pos] * 3 + [loop.noise_rot] * 3)
    for c in range(loop.corrections + 1):
        for _ in range(period):
            log = system.step()
            history.append(log.achieved)
        errors.append(se3.relative(targets, log.achieved))
        commands.append(se3.relative(targets, command))
        if c == loop.corrections:
            break
        measured = se3.perturb(history[0], rng.normal(size=targets.shape[:2] + (6,)) * noise_std)
        command = se3.perturb(command, loop.gain * se3.relative(measured, targets))
        system.set_command(command, ~goal)
    return {"error": np.stack(errors), "command": np.stack(commands)}
