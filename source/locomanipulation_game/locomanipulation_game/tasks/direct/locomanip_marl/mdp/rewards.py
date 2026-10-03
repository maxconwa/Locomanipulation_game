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

from locomanipulation_game.tasks.manager_based.locomanipulation_game.mdp.rewards import stand_still

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


def base_height_l2_lowered(
    env: ManagerBasedRLEnv, target_height: float, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """base_height_l2 with the target lowered by the arm goal's height drop.

    Same terrain-relative height as base_height_l2, but rays that missed the
    mesh are ignored instead of making the reward inf.
    """
    asset = env.scene["robot"]
    drop = env.command_manager.get_term(command_name).height_drop
    ground = ground_height(env.scene.sensors[sensor_cfg.name])
    error = torch.square(asset.data.root_pos_w[:, 2] - (target_height - drop + ground))
    # every ray missed: the robot is off the mesh, and terrain_out_of_bounds ends it
    return torch.nan_to_num(error, nan=0.0)


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
