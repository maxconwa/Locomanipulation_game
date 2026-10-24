"""The legs' gait clock and reward terms ported from ALMI-Open's h1_2_lower_env.py; base_cfg.LowerRewardsCfg
uses them."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# --- gait clock (ALMI h1_2_lower_env.py, ~line 500) ---
# The stride period and the stand/walk gate are the env cfg's gait_period (s per full stride) and zero_cmd_eps (a
# command norm over vx, vy, wz below it counts as "standing"), so a CLI override reaches them.
STANCE_THRESHOLD = 0.55    # phase below this is stance


def standing_command(env: ManagerBasedRLEnv, cmd: torch.Tensor) -> torch.Tensor:
    """Per env, whether the command's norm over (vx, vy, wz) is under the env's zero_cmd_eps."""
    return torch.norm(cmd[:, :3], dim=1) < env.cfg.zero_cmd_eps


def leg_phase(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    standing_ = standing_command(env, cmd)
    period = env.cfg.gait_period

    t = env.episode_length_buf * env.step_dt
    phase = torch.where(
        standing_, torch.zeros_like(t, dtype=torch.float), (t % period) / period
    )
    offset = torch.where(standing_, 0.0, 0.5)

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
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    phase = leg_phase(env, command_name)

    res = torch.zeros(env.num_envs, device=env.device)
    for i, body_id in enumerate(sensor_cfg.body_ids):
        in_contact = forces[:, :, body_id, 2].max(dim=1)[0] > 1.0
        is_stance = phase[:, i] < STANCE_THRESHOLD
        res += (~(in_contact ^ is_stance)).float()
    return res




def feet_swing_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.08,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfg is not None:
        scanner: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
        foot_z = foot_z - torch.mean(scanner.data.ray_hits_w[..., 2], dim=1, keepdim=True)

    return torch.sum(torch.square(foot_z - target_height) * ~in_contact, dim=1)



def contact_no_vel(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > 1.0

    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(foot_vel * in_contact.unsqueeze(-1)), dim=(1, 2))


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_force: float = 700.0,
    peak_over_substeps: bool = False,
    window: int = 1,
) -> torch.Tensor:
    """Each foot's contact force above max_force, summed (N), at the policy step's last physics substep, as ALMI-Open
    reads it (its contact tensor is refreshed once, after the decimation loop). Not the peak over the substeps:
    at 500 Hz that caught every landing's spike, about 70x feetodom_knee_200k's penalty, and the legs stopped stepping
    (lstm_run). peak_over_substeps: the peak over the sensor's history, as every run up to lstm_run; with window > 1,
    the peak of the force averaged over that many consecutive substeps (a landing's impulse over a shorter dt reads
    as a higher force: at 500 Hz a window matches the 200 Hz substep feetodom_knee_200k's crouch was learned with)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if peak_over_substeps:
        force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
        if window > 1:
            force = force.unfold(1, window, 1).mean(dim=-1)  # (N, T - window + 1, B)
        force = force.max(dim=1)[0]
    else:
        force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return torch.sum((force - max_force).clip(min=0.0), dim=1)


def self_contacts(
    env: ManagerBasedRLEnv, sensor_names: list[str], threshold: float = 0.1
) -> torch.Tensor:
    count = torch.zeros(env.num_envs, device=env.device)
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        # (N, T, 1, M, 3) -> peak over history, per filter link -> (N, M)
        peak = sensor.data.force_matrix_w_history.norm(dim=-1).max(dim=1)[0][:, 0]
        count += torch.sum(peak > threshold, dim=1)
    return count


def body_pair_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float = 0.3,
    max_dist: float = 0.6,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.norm(pos[:, 0, :] - pos[:, 1, :], dim=1)

    d_min = torch.clamp(dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(dist - max_dist, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0




def stand_still(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing_ = standing_command(env, cmd)

    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(dev), dim=1) * standing_


def stance_base_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    standing_ = standing_command(env, cmd)
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * standing_




def joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(dev), dim=1)


def ankle_action_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)

    ankle_names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
    action_order = {n: i for i, n in enumerate(term._joint_names)}
    ids = [action_order[n] for n in ankle_names]

    a = env.action_manager.action[:, ids]
    prev = env.action_manager.prev_action[:, ids]
    return torch.sum(torch.square(prev - a), dim=1)
