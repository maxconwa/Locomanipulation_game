"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_target_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, torch.Tensor]:
    """At episode end, a fall during an arm goal costs a level (ArmTargetsCommand.update_levels; the rest of the
    reach curriculum moves at goal events). Logs the mean level and what it means: the lowest goal height and the
    goals' sideways half-width."""
    term = env.command_manager.get_term(command_name)
    term.update_levels(env_ids, fell=env.termination_manager.terminated[env_ids])
    return {
        "level": torch.mean(term.level.float()),
        "min_height": torch.mean(term.min_height(term.level)),
        "half_width": torch.mean(term.half_width(term.level)),
    }
