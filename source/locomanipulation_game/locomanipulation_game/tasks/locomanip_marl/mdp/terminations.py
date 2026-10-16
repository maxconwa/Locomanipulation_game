"""Terminations for the two-agent task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import bad_orientation

from .commands import ground_height

if TYPE_CHECKING:
    from ..locomanip_marl_env import LocoManipMarlEnv

__all__ = ["fell_or_knelt", "golem_estop"]


def fell_or_knelt(env: LocoManipMarlEnv, limit_angle: float, min_pelvis_height: float,
                  sensor_name: str = "height_scanner") -> torch.Tensor:
    """bad_orientation, or the pelvis under min_pelvis_height (m) above the height scan's mean ground (not where
    every ray missed). Tilt alone let almitrain_gtodom_200k kneel upright, its pelvis 60 cm down, through pushes."""
    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_height(env.scene.sensors[sensor_name])
    return bad_orientation(env, limit_angle) | (height < min_pelvis_height)  # nan (no ground) compares False


def golem_estop(env: LocoManipMarlEnv) -> torch.Tensor:
    """GOLEM's safety layer would have e-stopped the robot during this policy step (golem_safety.py). Ends the
    episode only while env.estops_end_episodes (cfg.golem_estop_ends_episode, with the arm goals); the trips are
    logged either way.

    The env's monitor has checked every physics substep but the last; this checks the last, keeps the step's log
    and clears the flags.
    """
    monitor = env.golem_monitor
    if monitor is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    monitor.update()
    tripped = monitor.tripped().clone()
    env.golem_log = monitor.log()
    monitor.clear()
    return tripped if env.estops_end_episodes else torch.zeros_like(tripped)
