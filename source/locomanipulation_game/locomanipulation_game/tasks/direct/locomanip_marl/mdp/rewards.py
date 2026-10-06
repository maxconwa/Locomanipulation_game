"""Reward terms that need to know which agent they score.

Both agents' actions live in one ActionManager vector (legs first, then arms),
so Isaac Lab's action_rate_l2, which differences the whole vector, would charge
each agent for the other's actions. Self-contact has the same problem: every
self-contact pair would count against both agents.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.envs.mdp.rewards import ang_vel_xy_l2, lin_vel_z_l2

from locomanipulation_game.tasks.manager_based.locomanipulation_game.mdp.rewards import joint_deviation_l2, stand_still

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
    else:
        raise KeyError(f"No action term '{action_name}' in {manager.active_terms}")
    a = manager.action[:, start : start + dim]
    prev = manager.prev_action[:, start : start + dim]
    return torch.sum(torch.square(a - prev), dim=1)


def action_beyond_clip(env, agent: str, clip: float) -> torch.Tensor:
    """Sum over the agent's raw policy actions of how far each is beyond the clip the env applies.

    The env clamps actions to +-clip before the action manager sees them, so an
    output of 15 and one of 400 command the same joint target and no other term
    can tell them apart: nothing pulls a policy mean back once it drifts past
    the clip. Run 12 (agent_57600) sent knee actions of -300..-400 in the first
    steps of 10-15% of episodes at drop level 10, locking the knees straight.
    Linear, so those outliers are charged hard but not enough to swamp the
    critic. Reads the env's raw per-agent actions (LocoManipMarlEnv.actions).
    """
    return (env.actions[agent].abs() - clip).clamp(min=0.0).sum(dim=1)


class self_contacts_involving(ManagerTermBase):
    """Number of self-contact pairs above `threshold` that include at least one of `own_links`.

    Reads the per-link filtered sensors that TerrainSceneCfg builds: sensor i
    belongs to sensor_link_names[i] and its force-matrix column j is
    filter_link_names[i + 1 + j] (each sensor filters only against the links
    after its own, so every pair appears once).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        own = set(cfg.params["own_links"])
        sensor_names = cfg.params["sensor_names"]
        sensor_links = cfg.params["sensor_link_names"]
        filter_links = cfg.params["filter_link_names"]

        self._sensors: list[ContactSensor] = []
        self._masks: list[torch.Tensor] = []
        for i, (name, link) in enumerate(zip(sensor_names, sensor_links)):
            others = filter_links[i + 1 :]
            mask = torch.tensor([link in own or other in own for other in others], device=env.device)
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


# ---------------------------------------------------------------------------
# Arm targets (ArmTargetsCommand). The command computes the errors, so the
# arms' and the legs' copies of these terms read the same numbers.
# ---------------------------------------------------------------------------


def arm_target_pos_exp(env: ManagerBasedRLEnv, command_name: str, arm: int, std: float) -> torch.Tensor:
    """exp(-d^2 / std^2) on one arm's wrist position error."""
    pos_error, _ = env.command_manager.get_term(command_name).errors()
    return torch.exp(-pos_error[:, arm].square() / std**2)


def arm_target_quat_exp(env: ManagerBasedRLEnv, command_name: str, arm: int, std: float) -> torch.Tensor:
    """exp(-theta^2 / std^2) on one arm's wrist rotation error, std in radians."""
    _, rot_error = env.command_manager.get_term(command_name).errors()
    return torch.exp(-rot_error[:, arm].square() / std**2)


def arm_goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 on the step after both wrists reached the goal (and a new one was drawn).

    The smooth tracking kernels alone would pay the arms to hover just outside
    the tolerance: reaching swaps a near goal for a far one. This one-off bonus
    makes finishing a goal worth more than hovering. It lands one step late,
    since goals are checked after the reward is computed.
    """
    return env.command_manager.get_term(command_name).just_reached.float()


def base_height_l2_navigation(
    env: ManagerBasedRLEnv, target_height: float, sensor_cfg: SceneEntityCfg, arm_command_name: str
) -> torch.Tensor:
    """base_height_l2 during navigation only: the legs walk at standing height but may crouch for an arm goal.

    Same terrain-relative height as base_height_l2, but rays that missed the
    mesh are ignored instead of making the reward inf. During arm goals there
    is no height target at all: any crouch has to emerge from what makes the
    arms' targets reachable.
    """
    asset = env.scene["robot"]
    ground = ground_height(env.scene.sensors[sensor_cfg.name])
    error = torch.square(asset.data.root_pos_w[:, 2] - (target_height + ground))
    navigating = ~env.command_manager.get_term(arm_command_name).arm_mode
    # every ray missed: the robot is off the mesh, and terrain_out_of_bounds ends it
    return torch.nan_to_num(error, nan=0.0) * navigating


def stand_still_navigation(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, arm_command_name: str
) -> torch.Tensor:
    """stand_still, during navigation only.

    stand_still pulls the legs to their default angles whenever the velocity
    command is zero. Every arm goal zeroes it, so it was the legs' largest
    penalty (-0.27/s, run 5) and fought the posture changes the arms' reaching
    and the crouch levels need. During navigation it still holds the 5% of
    standing commands still.
    """
    navigating = ~env.command_manager.get_term(arm_command_name).arm_mode
    return stand_still(env, asset_cfg, command_name) * navigating


def yaw_rate_l2_during_arm_goal(env: ManagerBasedRLEnv, arm_command_name: str) -> torch.Tensor:
    """Squared pelvis yaw rate while an arm goal holds the velocity command at zero.

    Run 6 measured ~0.5 rad/s mean |yaw rate| during arm goals: the robot
    turned or wobbled while reaching (likely the arms' reaction torque), so the
    world-fixed targets kept moving in the pelvis frame and the arms stalled
    at ~11 cm. track_ang_vel_z already pays the legs for zero yaw; this term
    also charges the arms, whose motion causes it.
    """
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate) * env.command_manager.get_term(arm_command_name).arm_mode


# ---------------------------------------------------------------------------
# Gait shaping that resists a crouch, scaled down during arm goals only.
# Run 9's per-term breakdown (scripts/skrl/eval_crouch.py) measured these while
# the pelvis was going down: lin_vel_z -0.57/s, ang_vel_xy -0.96/s, and in a
# deep crouch hip_pos -0.57/s against -0.05 standing. Walking keeps the IBR
# weights; scale 1 during navigation, arm_goal_scale during an arm goal.
# ---------------------------------------------------------------------------


def _arm_goal_scale(env: ManagerBasedRLEnv, arm_command_name: str, arm_goal_scale: float) -> torch.Tensor:
    arm_mode = env.command_manager.get_term(arm_command_name).arm_mode
    return torch.where(arm_mode, arm_goal_scale, 1.0)


def lin_vel_z_l2_modal(
    env: ManagerBasedRLEnv, arm_command_name: str, arm_goal_scale: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """lin_vel_z_l2, scaled during arm goals: vertical pelvis velocity is what a crouch is."""
    return lin_vel_z_l2(env, asset_cfg) * _arm_goal_scale(env, arm_command_name, arm_goal_scale)


def ang_vel_xy_l2_modal(
    env: ManagerBasedRLEnv, arm_command_name: str, arm_goal_scale: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ang_vel_xy_l2, scaled during arm goals: a squat descent pitches the pelvis."""
    return ang_vel_xy_l2(env, asset_cfg) * _arm_goal_scale(env, arm_command_name, arm_goal_scale)


def joint_deviation_l2_modal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, arm_command_name: str, arm_goal_scale: float
) -> torch.Tensor:
    """joint_deviation_l2, scaled during arm goals (hip_pos: the hip roll/yaw a wide squat needs)."""
    return joint_deviation_l2(env, asset_cfg) * _arm_goal_scale(env, arm_command_name, arm_goal_scale)
