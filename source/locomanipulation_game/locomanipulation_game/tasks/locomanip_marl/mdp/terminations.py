"""Terminations for the two-agent task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from ..locomanip_marl_env import LocoManipMarlEnv

__all__ = ["episode_limit_reached", "golem_estop", "pelvis_contact_or_tilt"]


def episode_limit_reached(env: LocoManipMarlEnv) -> torch.Tensor:
    """The time-out with a length drawn per episode (cfg.episode_length_mean_s; LocoManipMarlEnv.episode_limit, in
    policy steps), never past max_episode_length (episode_length_s); Isaac Lab's time_out without a draw."""
    limit = getattr(env, "episode_limit", None)
    if limit is None:
        return env.episode_length_buf >= env.max_episode_length
    return env.episode_length_buf >= torch.clamp(limit, max=env.max_episode_length)


def pelvis_contact_or_tilt(env: LocoManipMarlEnv, sensor_cfg: SceneEntityCfg, max_roll: float,
                           max_pitch: float) -> torch.Tensor:
    """unitree_rl_gym's fall check (LeggedRobot.check_termination): a contact force over 1 N on sensor_cfg's bodies
    (the pelvis) in this step's last substep, or the pelvis rolled past max_roll or pitched past max_pitch (rad)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0, dim=1)
    roll, pitch, _ = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    return contact | (wrap_to_pi(roll).abs() > max_roll) | (wrap_to_pi(pitch).abs() > max_pitch)


def golem_estop(env: LocoManipMarlEnv) -> torch.Tensor:
    """Never ends the episode: a termination term so it runs once per step, after the physics substeps. The env's
    GOLEM e-stop monitor (golem_safety.py) has checked every substep but the last; this checks the last, keeps the
    step's trips (env.golem_tripped, which the env charges cfg.golem_estop_penalty on a trip's first step) and its log,
    and clears the flags."""
    monitor = env.golem_monitor
    monitor.update()
    env.golem_tripped = monitor.tripped().clone()
    env.golem_log = monitor.log()
    monitor.clear()
    return torch.zeros_like(env.golem_tripped)
