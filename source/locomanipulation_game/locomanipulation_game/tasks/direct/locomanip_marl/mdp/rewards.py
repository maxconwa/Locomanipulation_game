"""Reward terms that need to know which agent they score.

Both agents' actions live in one ActionManager vector (legs first, then arms),
so Isaac Lab's action_rate_l2, which differences the whole vector, would charge
each agent for the other's actions. Self-contact has the same problem: every
self-contact pair would count against both agents.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.sensors import ContactSensor

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
