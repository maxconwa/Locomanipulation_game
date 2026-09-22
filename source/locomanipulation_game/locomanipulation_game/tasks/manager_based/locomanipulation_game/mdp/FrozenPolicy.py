"""Action terms that run a policy without training it."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ObservationGroupCfg, ObservationManager
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import check_file_path, read_file

from locomanipulation_game.assets.h1_2 import REPO_ROOT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["FrozenPolicyAction", "FrozenPolicyActionCfg", "latest_export"]


def latest_export(experiment: str) -> str:
    """Newest exported policy in an experiment's log root; a missing path if none."""
    root = REPO_ROOT / "logs/rsl_rl" / experiment
    runs = sorted(p for p in root.glob("2*") if (p / "exported/policy.pt").is_file())
    return str(runs[-1] / "exported/policy.pt" if runs else root / "NO_EXPORT/policy.pt")

class FrozenPolicyAction(ActionTerm):
    """Runs an exported policy that consumes none of the agent's action vector.

    isaaclab_tasks' PreTrainedPolicyAction is the only implementation Isaac Lab
    ships and it is example code, not API: it hardcodes action_dim=3 and
    overwrites the low-level policy's velocity_commands with the agent's action,
    because in navigation the agent IS the commander. Copied rather than
    subclassed so an upstream refactor cannot change behaviour here.
    """

    cfg: FrozenPolicyActionCfg

    def __init__(self, cfg: FrozenPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = torch.jit.load(read_file(cfg.policy_path)).to(env.device).eval()

        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(
            self.num_envs, self._low_level_action_term.action_dim, device=self.device
        )

        # mdp.last_action reads env.action_manager.action -- the AGENT's vector.
        # The frozen policy needs its own previous output instead.
        def own_last_action(dummy_env=None):
            return self.low_level_actions

        cfg.low_level_observations.actions.func = own_last_action
        cfg.low_level_observations.actions.params = dict()

        # Its own manager: the agent's observation group is a different vector.
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self.low_level_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.low_level_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        ids = slice(None) if env_ids is None else env_ids
        self.low_level_actions[ids] = 0.0
        self._low_level_obs_manager.reset(env_ids)
        return {}

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            with torch.no_grad():
                # update_history=True is REQUIRED. The default False fills the
                # history buffers once and never again
                # (observation_manager.py:409-421).
                obs = self._low_level_obs_manager.compute_group("ll_policy", update_history=True)
                self.low_level_actions[:] = self.policy(obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class FrozenPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FrozenPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING
    low_level_decimation: int = 4
    low_level_actions: ActionTermCfg = MISSING
    low_level_observations: ObservationGroupCfg = MISSING
