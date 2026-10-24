"""Terminations for the two-agent task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import bad_orientation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .commands import ground_height

if TYPE_CHECKING:
    from ..locomanip_marl_env import LocoManipMarlEnv

__all__ = ["episode_limit_reached", "fell_or_knelt", "golem_estop", "pelvis_contact_or_tilt"]


def episode_limit_reached(env: LocoManipMarlEnv) -> torch.Tensor:
    """The time-out with a length drawn per episode (cfg.episode_length_mean_s; LocoManipMarlEnv.episode_limit, in
    policy steps), never past max_episode_length (episode_length_s); Isaac Lab's time_out without a draw."""
    limit = getattr(env, "episode_limit", None)
    if limit is None:
        return env.episode_length_buf >= env.max_episode_length
    return env.episode_length_buf >= torch.clamp(limit, max=env.max_episode_length)


def fell_or_knelt(env: LocoManipMarlEnv, limit_angle: float, min_pelvis_height: float,
                  sensor_name: str = "height_scanner") -> torch.Tensor:
    """bad_orientation, or the pelvis under min_pelvis_height (m) above the height scan's mean ground (not where
    every ray missed). Tilt alone let almitrain_gtodom_200k kneel upright, its pelvis 60 cm down, through pushes."""
    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_height(env.scene.sensors[sensor_name])
    return bad_orientation(env, limit_angle) | (height < min_pelvis_height)  # nan (no ground) compares False


def pelvis_contact_or_tilt(env: LocoManipMarlEnv, sensor_cfg: SceneEntityCfg, max_roll: float,
                           max_pitch: float) -> torch.Tensor:
    """unitree_rl_gym's fall check (LeggedRobot.check_termination): a contact force over 1 N on sensor_cfg's bodies
    (the pelvis) in this step's last substep, or the pelvis rolled past max_roll or pitched past max_pitch (rad)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0, dim=1)
    roll, pitch, _ = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w)
    return contact | (wrap_to_pi(roll).abs() > max_roll) | (wrap_to_pi(pitch).abs() > max_pitch)


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
    env.golem_tripped = tripped  # for cfg.golem_estop_penalty (LocoManipMarlEnv._get_rewards)
    env.golem_log = monitor.log()
    monitor.clear()
    return tripped if env.estops_end_episodes else torch.zeros_like(tripped)
