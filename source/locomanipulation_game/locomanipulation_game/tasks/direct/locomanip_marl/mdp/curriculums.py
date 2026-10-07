"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_target_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, torch.Tensor]:
    """At episode end, a fall during an arm goal costs a level (ArmTargetsCommand.update_levels; the rest of the
    reach curriculum moves at goal events). Logs the mean spread and drop levels."""
    term = env.command_manager.get_term(command_name)
    term.update_levels(env_ids, fell=env.termination_manager.terminated[env_ids])
    return {"spread": torch.mean(term.spread_level.float()), "drop": torch.mean(term.drop_level.float())}
