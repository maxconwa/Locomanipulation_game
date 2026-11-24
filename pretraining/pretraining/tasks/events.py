"""Start-up randomization of the tracking env: foot friction and encoder bias (the torso COM offset is Isaac Lab's
``randomize_rigid_body_com``).

Ported from mjlab (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0), ``src/mjlab/envs/mdp/dr/geom.py``
(``geom_friction`` with ``operation="abs"``, ``shared_random=True``) and ``src/mjlab/envs/mdp/dr/joint.py``
(``encoder_bias``). MuJoCo's per-geom friction becomes PhysX's per-shape material: one static and dynamic friction
per env on every collision shape of the foot bodies (the ground's multiply-combined friction of 1 makes it the
contact's, as the feet's geom priority does in h12_train). The encoder bias lives at ``env.encoder_bias``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

from pretraining.constants import JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def encoder_bias(env: ManagerBasedEnv) -> torch.Tensor:
    """``env.encoder_bias``, ``(num_envs, 27)`` rad in ``JOINT_NAMES`` order, created as zeros on first use."""
    if not hasattr(env, "encoder_bias"):
        env.encoder_bias = torch.zeros(env.num_envs, len(JOINT_NAMES), device=env.device)
    return env.encoder_bias


def sample_encoder_bias(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, bias_range: tuple[float, float]
) -> None:
    """Draw each joint's encoder bias, uniform in ``bias_range`` rad, for ``env_ids``."""
    bias = encoder_bias(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    bias[env_ids] = sample_uniform(bias_range[0], bias_range[1], (len(env_ids), bias.shape[1]), device=env.device)


def sample_encoder_offsets(env: ManagerBasedEnv, env_ids: torch.Tensor | None, half_widths: dict[str, float]) -> None:
    """Draw ``env.encoder_bias`` for ``env_ids`` as main's encoder offsets (``joint_noise.ENCODER_OFFSET_RAD``): each
    joint uniform in +- the half-width of the first pattern it matches in full, 0 for joints no pattern names."""
    bias = encoder_bias(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    span = torch.tensor([next((w for p, w in half_widths.items() if re.fullmatch(p, j)), 0.0) for j in JOINT_NAMES],
                        device=env.device)
    bias[env_ids] = (2.0 * torch.rand(len(env_ids), len(span), device=env.device) - 1.0) * span


def _shape_slices(asset: Articulation, body_ids: list[int]) -> list[int]:
    """Indices, in the articulation's shape order, of the collision shapes of ``body_ids``."""
    counts = [
        asset._physics_sim_view.create_rigid_body_view(path).max_shapes for path in asset.root_physx_view.link_paths[0]
    ]
    if sum(counts) != asset.root_physx_view.max_shapes:
        raise ValueError(f"shape count per body {counts} does not add up to {asset.root_physx_view.max_shapes}")
    starts = [sum(counts[:b]) for b in range(len(counts))]
    return [s for b in body_ids for s in range(starts[b], starts[b] + counts[b])]


def foot_friction(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, friction_range: tuple[float, float], asset_cfg: SceneEntityCfg
) -> None:
    """Set one friction coefficient per env, uniform in ``friction_range``, as the static and dynamic friction of
    every shape of ``asset_cfg``'s bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.num_envs) if env_ids is None else env_ids.cpu()
    shapes = torch.tensor(_shape_slices(asset, asset_cfg.body_ids))
    if len(shapes) == 0:
        raise ValueError(f"no collision shapes on bodies {asset_cfg.body_names}")
    friction = sample_uniform(friction_range[0], friction_range[1], (len(env_ids), 1), device="cpu")
    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids[:, None], shapes, 0] = friction
    materials[env_ids[:, None], shapes, 1] = friction
    asset.root_physx_view.set_material_properties(materials, env_ids)


def joint_ranges(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, ranges: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg
) -> None:
    """Set the hard position limits of the joints named in ``ranges`` (rad); the soft limits follow."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset.find_joints(list(ranges), preserve_order=True)[0]
    num = env.num_envs if env_ids is None else len(env_ids)
    limits = torch.tensor(list(ranges.values()), device=asset.device).expand(num, -1, -1)
    if env_ids is not None:
        env_ids = env_ids.to(asset.device)
    asset.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, env_ids=env_ids)
