"""Event terms for the two-agent task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["passive_joint_damping"]


def passive_joint_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    damping_distribution_params: tuple[float, float],
):
    """Passive joint damping like MuJoCo's: PhysX's viscous joint friction, the asset's default plus a uniform draw
    per env and joint. Damping added to the PD drive instead is clipped with the drive at the motor's effort limit."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
    rows, cols = env_ids[:, None], joint_ids[None, :]
    damping = asset.data.default_joint_viscous_friction_coeff[rows, cols] + torch.empty(
        len(env_ids), len(joint_ids), device=asset.device
    ).uniform_(*damping_distribution_params)
    asset.write_joint_friction_coefficient_to_sim(
        asset.data.joint_friction_coeff[rows, cols].clone(),
        joint_viscous_friction_coeff=damping,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
