"""Curriculum terms for the two-agent task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

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


def _tracked_episode(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, max_error: float,
                     fall_term: str, time_out_term: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(tracked, fell) for the envs whose episode is ending: tracked is a time-out without a fall whose mean velocity
    tracking error while navigating (command_name's per-env sums, read before its reset) was under max_error (m/s)."""
    command = env.command_manager.get_term(command_name)
    steps = command._nav_steps[env_ids]
    error = command._nav_err_xy[env_ids] / steps.clamp(min=1.0)
    fell = env.termination_manager.get_term(fall_term)[env_ids]
    timed_out = env.termination_manager.get_term(time_out_term)[env_ids]
    return timed_out & ~fell & (steps > 0) & (error < max_error), fell


def terrain_levels_tracking(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, max_error: float,
                            fall_term: str = "fell", time_out_term: str = "time_out") -> dict[str, torch.Tensor]:
    """At episode end, an env moves down a terrain row after an episode that ended in fall_term, and up one after a
    time-out whose mean velocity tracking error while navigating (command_name's per-env sums, read before its reset)
    was under max_error (m/s); otherwise it stays, as after an e-stop or an episode with no navigation. In place of
    terrain_levels_vel, which wants the robot to end far from its start: ours turn every few seconds and stop for arm
    goals, so it demoted the robots that tracked best. Robots past the last row go to a random one (TerrainImporter)."""
    terrain = env.scene.terrain
    move_up, fell = _tracked_episode(env, env_ids, command_name, max_error, fall_term, time_out_term)
    terrain.update_env_origins(env_ids, move_up, fell)
    return {"terrain_level": terrain.terrain_levels.float().mean(), "promoted": move_up.float().mean(),
            "demoted": fell.float().mean()}


def gated_terrain_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, max_error: float,
                         arm_command_name: str, gated: bool = True, max_fall_share: float = 0.02, min_tracked_share: float = 0.7,
                         min_depth_level: float = 4.0, min_crouch_ratio: float = 0.75, steps_per_row: int = 8000,
                         episode_decay: float = 0.999, fall_term: str = "fell",
                         time_out_term: str = "time_out") -> dict[str, torch.Tensor]:
    """terrain_levels_tracking under a ceiling that starts at row 0 and rises slowly once navigation is robust and the
    crouch has emerged; the robots start on row 0 (max_init_terrain_level 0), the flattest.

    Over the ending episodes, exponential means with decay episode_decay per episode: the share that fell and the
    share that tracked their velocity commands (as terrain_levels_tracking promotes); over the arm goals' steps, the
    pelvis drop and the target drop (arm_command_name's metrics). The gate wants falls under max_fall_share, tracked
    over min_tracked_share, the arm depth level's mean at least min_depth_level and the pelvis dropping at least
    min_crouch_ratio x as far as the targets (the legs helping the arms; nothing rewards a pelvis height). It opens
    once, raising the ceiling to row 1, then one row more every steps_per_row policy steps while the gate's conditions
    still hold; a row up is never forced, each env still has to earn it. Logs the ceiling and the gate's inputs.
    Not gated: terrain_levels_tracking (no ceiling)."""
    terrain = env.scene.terrain
    if not gated:
        return terrain_levels_tracking(env, env_ids, command_name, max_error, fall_term, time_out_term)
    if not hasattr(env, "terrain_gate"):
        zero = lambda: torch.zeros((), device=env.device)  # noqa: E731
        env.terrain_gate = {"fell": torch.ones((), device=env.device), "tracked": zero(), "pelvis": zero(),
                            "target": zero(), "ceiling": 0, "raised_at": 0}
    gate = env.terrain_gate
    move_up, fell = _tracked_episode(env, env_ids, command_name, max_error, fall_term, time_out_term)
    n = len(move_up)
    if n:
        keep = episode_decay ** n
        gate["fell"] = keep * gate["fell"] + (1.0 - keep) * fell.float().mean()
        gate["tracked"] = keep * gate["tracked"] + (1.0 - keep) * move_up.float().mean()
    arm = env.command_manager.get_term(arm_command_name)
    steps = arm._crouch_n[env_ids]
    goals = int((steps > 0).sum())
    if goals:
        keep = episode_decay ** goals
        gate["pelvis"] = keep * gate["pelvis"] + (1.0 - keep) * arm._crouch_p[env_ids].sum() / steps.sum()
        gate["target"] = keep * gate["target"] + (1.0 - keep) * arm._crouch_t[env_ids].sum() / steps.sum()
    ratio = gate["pelvis"] / gate["target"].clamp(min=1e-3)
    depth = arm.level.float().mean()
    ready = bool((gate["fell"] <= max_fall_share) & (gate["tracked"] >= min_tracked_share)
                 & (depth >= min_depth_level) & (ratio >= min_crouch_ratio))
    last_row = terrain.max_terrain_level - 1
    step = env.common_step_counter
    if ready and gate["ceiling"] < last_row and (gate["ceiling"] == 0 or step - gate["raised_at"] >= steps_per_row):
        gate["ceiling"] += 1
        gate["raised_at"] = step
    move_up &= terrain.terrain_levels[env_ids] < gate["ceiling"]
    terrain.update_env_origins(env_ids, move_up, fell)
    return {"terrain_level": terrain.terrain_levels.float().mean(), "ceiling": torch.tensor(float(gate["ceiling"])),
            "promoted": move_up.float().mean(), "demoted": fell.float().mean(), "gate_ready": torch.tensor(float(ready)),
            "fell_share": gate["fell"], "tracked_share": gate["tracked"], "crouch_ratio": ratio}


def _push_levels(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Each env's push level (long, (num_envs,)), starting at 0: no push."""
    if not hasattr(env, "push_level"):
        env.push_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env.push_level


def push_levels(env: ManagerBasedRLEnv, env_ids: Sequence[int], num_levels: int, max_speed: float,
                command_name: str, max_error: float, fall_term: str = "fell",
                time_out_term: str = "time_out") -> dict[str, torch.Tensor]:
    """At episode end, as terrain_levels_tracking: an env's push level goes down one after an episode that ended in
    fall_term and up one after a time-out that tracked its velocity commands (mean navigation error under max_error,
    m/s), within [0, num_levels]; otherwise it stays. push_by_level pushes it at up to level / num_levels x max_speed.
    Survival alone used to raise it, and lstm_run's robots climbed to 0.86 m/s standing still. Logs the mean level,
    push and the shares promoted and demoted."""
    level = _push_levels(env)
    move_up, fell = _tracked_episode(env, env_ids, command_name, max_error, fall_term, time_out_term)
    level[env_ids] = (level[env_ids] + move_up.long() - fell.long()).clamp(0, num_levels)
    return {"push_level": level.float().mean(), "max_push_speed": level.float().mean() * max_speed / num_levels,
            "promoted": move_up.float().mean(), "demoted": fell.float().mean()}


def push_by_level(env: ManagerBasedRLEnv, env_ids: torch.Tensor, num_levels: int, max_speed: float,
                  fixed: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """push_by_setting_velocity with each env's own range: x and y velocity changes uniform in
    +-level / num_levels x max_speed (m/s), the level from push_levels. With fixed, +-max_speed for every env whatever
    its level (every run up to feetodom_knee_200k: 0.4 m/s every 8-12 s)."""
    asset = env.scene[asset_cfg.name]
    if fixed:
        speed = torch.full((len(env_ids),), max_speed, device=asset.device)
    else:
        speed = _push_levels(env)[env_ids].float() * (max_speed / num_levels)
    vel_w = asset.data.root_vel_w[env_ids]
    vel_w[:, :2] += (2.0 * torch.rand(len(env_ids), 2, device=asset.device) - 1.0) * speed.unsqueeze(1)
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
