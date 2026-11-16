"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_kernels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, torch.Tensor]:
    """Logs the arm goals' tracking-kernel stage (ArmTargetsCommand._judge_kernels), its stds and the share of the
    last judged window that came within them, and the share of tasks that are arm goals."""
    term = env.command_manager.get_term(command_name)
    return {
        "kernel_stage": torch.tensor(float(term.kernel_stage), device=env.device),
        "fine_pos_std": torch.tensor(term.fine_pos_std, device=env.device),
        "rot_std": torch.tensor(term.rot_std, device=env.device),
        "kernel_window_share": term.kernel_window_share,
        "arm_goal_prob": torch.tensor(term.arm_goal_prob if term.goals_enabled else 0.0, device=env.device),
    }


def velocity_kernels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, torch.Tensor]:
    """Logs the velocity command's tracking-kernel stages (ModalVelocityCommand._judge_kernels), their stds and the share
    of the last judged window of each that came within them."""
    term = env.command_manager.get_term(command_name)
    return {
        "lin_stage": torch.tensor(float(term.lin_kernel_stage), device=env.device),
        "lin_std": torch.tensor(term.lin_vel_std, device=env.device),
        "lin_window_share": term.lin_kernel_window_share,
        "ang_stage": torch.tensor(float(term.ang_kernel_stage), device=env.device),
        "ang_std": torch.tensor(term.ang_vel_std, device=env.device),
        "ang_window_share": term.ang_kernel_window_share,
    }
