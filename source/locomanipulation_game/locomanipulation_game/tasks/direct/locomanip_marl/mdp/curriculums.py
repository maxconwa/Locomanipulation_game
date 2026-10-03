"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_target_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> torch.Tensor:
    """At episode end, a fall during an arm goal costs a level (ArmTargetsCommand.update_levels). Logs the mean level.

    The rest of the arm curriculum moves at goal events, inside the command.
    """
    term = env.command_manager.get_term(command_name)
    term.update_levels(env_ids, fell=env.termination_manager.terminated[env_ids])
    return torch.mean(term.level.float())


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    min_commanded_path: float = 2.0,
    promote_ratio: float = 0.8,
    demote_ratio: float = 0.5,
) -> torch.Tensor:
    """Terrain levels moved on how well the navigation commands were followed, not on distance walked.

    terrain_levels_vel promotes after 4 m from the env origin in one episode.
    Here only part of an episode is navigation, and the commands change
    direction, so few episodes ever got there: the level sat at ~0.43 (and at
    0.41 in the IBR game). This term reads the episode's commanded_path and
    tracked_path from ModalVelocityCommand: with at least min_commanded_path
    metres commanded, up if >= promote_ratio of it was covered along the
    commanded direction, down below demote_ratio. Down on a fall too. Runs
    before the command reset zeroes the two paths.
    """
    term = env.command_manager.get_term(command_name)
    commanded = term.metrics["commanded_path"][env_ids]
    tracked = term.metrics["tracked_path"][env_ids]
    fell = env.termination_manager.terminated[env_ids]
    judged = commanded >= min_commanded_path
    ratio = tracked / commanded.clamp(min=1e-6)
    move_up = judged & ~fell & (ratio >= promote_ratio)
    move_down = fell | (judged & (ratio < demote_ratio))
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down & ~move_up)
    return torch.mean(terrain.terrain_levels.float())
