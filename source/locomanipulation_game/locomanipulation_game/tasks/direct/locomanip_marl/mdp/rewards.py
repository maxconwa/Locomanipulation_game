"""Reward terms for the two-agent task.

Both agents' actions live in one ActionManager vector, and self-contact pairs involve both agents' links, so those
terms here are restricted to one agent.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from locomanipulation_game.tasks.manager_based.locomanipulation_game.mdp.rewards import (
    STANCE_THRESHOLD,
    joint_deviation_l2,
    leg_phase,
)

from .commands import ground_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_term_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """action_rate_l2 restricted to one action term's slice of the action vector."""
    manager = env.action_manager
    start = 0
    for name, dim in zip(manager.active_terms, manager.action_term_dim):
        if name == action_name:
            break
        start += dim
    a = manager.action[:, start : start + dim]
    prev = manager.prev_action[:, start : start + dim]
    return torch.sum(torch.square(a - prev), dim=1)


def action_beyond_clip(env, agent: str, action_name: str, clip: float) -> torch.Tensor:
    """How far the agent's raw actions are beyond the env's clip, plus what its action term's position bound cut off,
    summed over its joints (action units). Nothing else pulls a policy mean back once it drifts past the clip."""
    excess = (env.actions[agent].abs() - clip).clamp(min=0.0).sum(dim=1)
    return excess + env.action_manager.get_term(action_name).beyond_bounds.sum(dim=1)


class self_contacts_involving(ManagerTermBase):
    """Number of self-contact pairs above `threshold` that include at least one of `own_links`.

    Reads the per-link filtered sensors that TerrainSceneCfg builds: sensor i belongs to sensor_link_names[i] and
    its force-matrix column j is filter_link_names[i + 1 + j] (each pair appears once).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        own = set(cfg.params["own_links"])
        filter_links = cfg.params["filter_link_names"]
        self._sensors: list[ContactSensor] = []
        self._masks: list[torch.Tensor] = []
        for i, (name, link) in enumerate(zip(cfg.params["sensor_names"], cfg.params["sensor_link_names"])):
            mask = torch.tensor([link in own or other in own for other in filter_links[i + 1 :]], device=env.device)
            if mask.any():
                self._sensors.append(env.scene.sensors[name])
                self._masks.append(mask)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_names: list[str],
        sensor_link_names: list[str],
        filter_link_names: list[str],
        own_links: list[str],
        threshold: float = 0.1,
    ) -> torch.Tensor:
        count = torch.zeros(env.num_envs, device=env.device)
        for sensor, mask in zip(self._sensors, self._masks):
            # (N, T, 1, M, 3) -> peak over history, per filter link -> (N, M)
            peak = sensor.data.force_matrix_w_history.norm(dim=-1).max(dim=1)[0][:, 0]
            count += torch.sum((peak > threshold) & mask, dim=1)
        return count


class track_velocity_avg_exp(ManagerTermBase):
    """exp(-err^2 / std^2) of the velocity command against the pelvis velocity averaged over the last window_s.

    component "lin_xy": (vx, vy) in the pelvis frame; "ang_z": the yaw rate. A step sways and twists the pelvis
    within the gait cycle, so on the instantaneous velocity standing still under a small command pays about as
    much as following it. Averaged over one gait cycle, only the motion the command asks for counts.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._window = max(int(round(cfg.params["window_s"] / env.step_dt)), 1)
        dims = 2 if cfg.params["component"] == "lin_xy" else 1
        self._samples = torch.zeros(env.num_envs, self._window, dims, device=env.device)
        self._count = torch.zeros(env.num_envs, device=env.device)
        self._slot = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._samples[ids] = 0.0
        self._count[ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, std: float, component: str, window_s: float
    ) -> torch.Tensor:
        data = env.scene["robot"].data
        if component == "lin_xy":
            velocity, command = data.root_lin_vel_b[:, :2], env.command_manager.get_command(command_name)[:, :2]
        else:
            velocity, command = data.root_ang_vel_b[:, 2:3], env.command_manager.get_command(command_name)[:, 2:3]
        self._samples[:, self._slot] = velocity
        self._slot = (self._slot + 1) % self._window
        self._count = torch.clamp(self._count + 1.0, max=self._window)
        mean = self._samples.sum(dim=1) / self._count.unsqueeze(1)
        return torch.exp(-torch.sum(torch.square(command - mean), dim=1) / std**2)


def arm_target_pos_exp(env: ManagerBasedRLEnv, command_name: str, arm: int, std: float) -> torch.Tensor:
    """exp(-d^2 / std^2) on one arm's wrist position error."""
    pos_error, _ = env.command_manager.get_term(command_name).errors()
    return torch.exp(-pos_error[:, arm].square() / std**2)


def arm_target_quat_exp(env: ManagerBasedRLEnv, command_name: str, arm: int, std: float) -> torch.Tensor:
    """exp(-theta^2 / std^2) on one arm's wrist rotation error, std in radians."""
    _, rot_error = env.command_manager.get_term(command_name).errors()
    return torch.exp(-rot_error[:, arm].square() / std**2)


def arm_goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 on the step after both wrists were first held on the goal: a one-off bonus, so finishing pays more than
    hovering just outside the tolerance."""
    return env.command_manager.get_term(command_name).just_reached.float()


def yaw_rate_l2_during_arm_goal(env: ManagerBasedRLEnv, arm_command_name: str) -> torch.Tensor:
    """Squared pelvis yaw rate during an arm goal: turning moves the world-fixed targets in the pelvis frame."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate) * env.command_manager.get_term(arm_command_name).arm_mode


def joint_deviation_l2_modal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, arm_command_name: str, arm_goal_scale: float
) -> torch.Tensor:
    """joint_deviation_l2, times arm_goal_scale during arm goals."""
    arm_mode = env.command_manager.get_term(arm_command_name).arm_mode
    return joint_deviation_l2(env, asset_cfg) * torch.where(arm_mode, arm_goal_scale, 1.0)


def feet_swing_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rest_height: float,
    lift_height: float,
    command_name: str = "base_velocity",
    terrain_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """How far each foot in its swing phase is below a swing reference, summed over the feet (m).

    The gait clock (leg_phase, which the legs observe) puts a foot in swing for phase >= STANCE_THRESHOLD while a
    walk is commanded. The reference rises from rest_height to rest_height + lift_height and back,
    rest_height + lift_height * sin(pi * s) over the swing's progress s in [0, 1]. A foot pays for being below it
    whether it touches the ground or not, so dragging it through the swing costs. asset_cfg's bodies are the feet in
    leg_phase's order (left, right).
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfg is not None:
        foot_z = foot_z - ground_height(env.scene.sensors[terrain_sensor_cfg.name]).unsqueeze(1)
    phase = leg_phase(env, command_name)
    swing = phase >= STANCE_THRESHOLD
    progress = ((phase - STANCE_THRESHOLD) / (1.0 - STANCE_THRESHOLD)).clamp(0.0, 1.0)
    reference = rest_height + lift_height * torch.sin(torch.pi * progress)
    return torch.sum((reference - foot_z).clamp(min=0.0) * swing, dim=1)
