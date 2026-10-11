"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_target_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, torch.Tensor]:
    """At episode end, a fall during an arm goal drops the depth level (ArmTargetsCommand.update_levels; the rest of
    the depth curriculum moves at goal events). Logs the mean depth level and z_min, and the tracking-kernel stage,
    its stds and the share of the last judged window that came within them."""
    term = env.command_manager.get_term(command_name)
    term.update_levels(env_ids, fell=env.termination_manager.terminated[env_ids])
    return {
        "depth_level": torch.mean(term.level.float()),
        "z_min": torch.mean(term.z_min(term.level)),
        "kernel_stage": torch.tensor(float(term.kernel_stage), device=env.device),
        "fine_pos_std": torch.tensor(term.fine_pos_std, device=env.device),
        "rot_std": torch.tensor(term.rot_std, device=env.device),
        "kernel_window_share": term.kernel_window_share,
    }
