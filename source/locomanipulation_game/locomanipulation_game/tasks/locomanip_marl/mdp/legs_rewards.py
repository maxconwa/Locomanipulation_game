"""The legs' gait clock and the reward terms ported from ALMI-Open's h1_2_lower_env.py."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- gait clock (ALMI h1_2_lower_env.py) ---
# The stride period and the stand/walk gate are the env cfg's gait_period (s per full stride) and zero_cmd_eps (a
# command norm over vx, vy, wz below it counts as standing).
STANCE_THRESHOLD = 0.55    # phase below this is stance


def standing_command(env: ManagerBasedRLEnv, cmd: torch.Tensor) -> torch.Tensor:
    """Per env, whether the command's norm over (vx, vy, wz) is under the env's zero_cmd_eps."""
    return torch.norm(cmd[:, :3], dim=1) < env.cfg.zero_cmd_eps


def leg_phase(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Each leg's phase in [0, 1), (num_envs, 2): the legs half a stride apart while a walk is commanded, both 0
    while standing."""
    standing = standing_command(env, env.command_manager.get_command(command_name))
    period = env.cfg.gait_period
    t = env.episode_length_buf * env.step_dt
    phase = torch.where(standing, torch.zeros_like(t, dtype=torch.float), (t % period) / period)
    offset = torch.where(standing, 0.0, 0.5)
    return torch.stack([phase, (phase + offset) % 1.0], dim=-1)


def gait_phase_sin(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Observation term: sin of each leg's phase. Shape (num_envs, 2)."""
    return torch.sin(2.0 * math.pi * leg_phase(env, command_name))


def gait_phase_cos(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Observation term: cos of each leg's phase. Shape (num_envs, 2)."""
    return torch.cos(2.0 * math.pi * leg_phase(env, command_name))


def contact_matches_phase(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Number of feet whose contact (over the policy step's substeps) matches the gait clock's stance or swing."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    phase = leg_phase(env, command_name)
    res = torch.zeros(env.num_envs, device=env.device)
    for i, body_id in enumerate(sensor_cfg.body_ids):
        in_contact = forces[:, :, body_id, 2].max(dim=1)[0] > 1.0
        is_stance = phase[:, i] < STANCE_THRESHOLD
        res += (~(in_contact ^ is_stance)).float()
    return res


def contact_no_vel(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Squared velocity of the feet in contact: a planted foot should not slide."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(foot_vel * in_contact.unsqueeze(-1)), dim=(1, 2))


def feet_contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_force: float = 700.0, window: int = 3
) -> torch.Tensor:
    """Each foot's contact force above max_force, summed (N): per foot the peak over the policy step's physics
    substeps of the force averaged over `window` consecutive substeps (6 ms at 500 Hz). The raw per-substep peak
    caught every landing's spike, and a policy charged with it stopped stepping."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    if window > 1:
        force = force.unfold(1, window, 1).mean(dim=-1)  # (N, T - window + 1, B)
    return torch.sum((force.max(dim=1)[0] - max_force).clip(min=0.0), dim=1)


def body_pair_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.3,
    max_dist: float = 0.6,
) -> torch.Tensor:
    """In [0, 1]: the two bodies' horizontal distance within [min_dist, max_dist] scores 1, falling off outside."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)
    d_min = torch.clamp(dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(dist - max_dist, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0


def stance_base_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Squared pelvis xy velocity while standing is commanded."""
    asset: Articulation = env.scene["robot"]
    standing = standing_command(env, env.command_manager.get_command(command_name))
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * standing


def joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Squared deviation of asset_cfg's joints from their defaults, summed."""
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(dev), dim=1)


def ankle_action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Squared change of the ankles' raw actions since the last step. action_name's term must come first in the
    action vector (its joints are found by their index in that term)."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    action_order = {n: i for i, n in enumerate(term._joint_names)}
    ids = [action_order[asset.joint_names[i]] for i in asset_cfg.joint_ids]
    a = env.action_manager.action[:, ids]
    prev = env.action_manager.prev_action[:, ids]
    return torch.sum(torch.square(prev - a), dim=1)
