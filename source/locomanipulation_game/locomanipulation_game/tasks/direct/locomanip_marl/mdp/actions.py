"""Joint-position actions with bounded, optionally rate-limited targets.

Run 12's policy commanded targets far past the joint stops (knee actions of
-300 against a clip of 10). This term keeps the plain JointPositionAction
mapping (target = action * scale + offset) and then, once per policy step:

  1. bounds each target to [hard lower - target_margin, hard upper + target_margin]:
     the margin lets the PD controller push against a limit (holding a deep
     squat needs the knee target ~0.37 rad past the knee angle), but no further;
  2. if max_rate is set, limits how far each target may move from the last
     applied one: max_rate (rad/s) during arm goals and settle segments,
     max_rate_navigation while walking, per env from the arm command's mode.
     Flat run A had rate caps (arms 1.0-1.5 rad/s, knee and hip pitch 1.2 during
     arm goals) for slower, safer moves; the user reverted them for the restart.

The policy sees what was applied (applied_actions, in action units, through
mdp.applied_action), and beyond_bounds (also in action units) is what the
bound cut off, for the legs' action_beyond_clip penalty.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["RateLimitedJointPositionAction", "RateLimitedJointPositionActionCfg"]


class RateLimitedJointPositionAction(JointPositionAction):
    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        hard = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower = hard[..., 0] - cfg.target_margin
        self._upper = hard[..., 1] + cfg.target_margin
        self._rate = self._per_joint(cfg.max_rate) if cfg.max_rate else None
        self._rate_navigation = self._per_joint(cfg.max_rate_navigation) if cfg.max_rate_navigation else self._rate
        self._applied = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._beyond = torch.zeros_like(self._raw_actions)
        caps = "no rate caps"
        if self._rate is not None:
            caps = f"rate caps (rad/s, arm goals and settle) {dict(zip(self._joint_names, self._rate.tolist()))}"
            if cfg.max_rate_navigation:
                caps += f"; walking {dict(zip(self._joint_names, self._rate_navigation.tolist()))}"
        print(f"[INFO] {type(self).__name__}: targets within the hard limits +- {cfg.target_margin} rad; {caps}")

    def _per_joint(self, rates: dict[str, float]) -> torch.Tensor:
        index_list, _, value_list = string_utils.resolve_matching_names_values(rates, self._joint_names)
        values = torch.full((self._num_joints,), float("inf"), device=self.device)
        values[index_list] = torch.tensor(value_list, device=self.device)
        if torch.isinf(values).any():
            missing = [n for n, v in zip(self._joint_names, values.tolist()) if v == float("inf")]
            raise ValueError(f"{type(self).__name__}: no rate cap for joints {missing}")
        return values

    @property
    def applied_actions(self) -> torch.Tensor:
        """The applied targets in action units: what the policy's action amounted to after the bound and the rate."""
        return (self._applied - self._offset) / self._scale

    @property
    def beyond_bounds(self) -> torch.Tensor:
        """How far each raw action was past its joint's target bound, in action units (0 inside)."""
        return self._beyond

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target = self._raw_actions * self._scale + self._offset
        bounded = torch.maximum(torch.minimum(target, self._upper), self._lower)
        self._beyond = (target - bounded).abs() / abs(self._scale)
        if self._rate is None:
            self._applied = bounded
            self._processed_actions = self._applied
            return
        rate = self._rate.expand_as(bounded)
        if self.cfg.max_rate_navigation:
            arm = self._env.command_manager.get_term(self.cfg.mode_command_name)
            walking = ~(arm.arm_mode | arm.settling)
            rate = torch.where(walking.unsqueeze(1), self._rate_navigation, rate)
        step = rate * self._env.step_dt
        self._applied = torch.maximum(torch.minimum(bounded, self._applied + step), self._applied - step)
        self._processed_actions = self._applied

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # the reset events have written the new joint state: start the rate limit from it
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._beyond[ids] = 0.0


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RateLimitedJointPositionAction

    max_rate: dict[str, float] | None = None
    """rad/s per joint (regex keys, every joint covered): the cap during arm goals and settle segments,
    or always when max_rate_navigation is None. None: no rate limit, bounds only."""
    max_rate_navigation: dict[str, float] | None = None
    """rad/s per joint while walking (not in an arm goal or a settle)."""
    mode_command_name: str = "arm_targets"
    """The ArmTargetsCommand whose arm_mode / settling pick the cap."""
    target_margin: float = 0.4
    """rad past each hard joint limit a target may go."""
