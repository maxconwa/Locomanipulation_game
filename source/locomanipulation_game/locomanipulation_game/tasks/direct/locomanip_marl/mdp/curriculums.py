"""Curriculum terms for the arm targets."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_target_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> torch.Tensor:
    """Moves each ending episode's arm-target level (ArmTargetsCommand.update_levels). Logs the mean level.

    Runs at reset, before the command manager resets: it reads the episode's
    goal counts, which the command's reset then zeroes.
    """
    term = env.command_manager.get_term(command_name)
    term.update_levels(env_ids, fell=env.termination_manager.terminated[env_ids])
    return torch.mean(term.level.float())
