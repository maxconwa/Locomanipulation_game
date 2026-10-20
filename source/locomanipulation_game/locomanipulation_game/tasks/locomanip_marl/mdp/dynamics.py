"""Per-env joint dynamics: passive damping, dry friction and armature around GOLEM's Robocasa MJCF values."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ..passive_actuator import PassivePDActuator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_passive_dynamics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    damping_range: tuple[float, float],
    friction_range: tuple[float, float],
    armature_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Startup event: per env and joint of every PassivePDActuator, passive damping (N m s/rad), dry friction (N m)
    and armature (kg m^2, written to PhysX) drawn uniformly from their ranges. Joints of other actuators (the
    grippers, implicit) keep theirs. Needs the explicit PD (env.explicit_pd) that builds the PassivePDActuators."""
    robot = env.scene[asset_cfg.name]
    actuators = [a for a in robot.actuators.values() if isinstance(a, PassivePDActuator)]
    if not actuators:
        raise RuntimeError("randomize_joint_passive_dynamics needs PassivePDActuators: run with env.explicit_pd=True")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    n = len(ids)

    def draw(rng: tuple[float, float], cols: int) -> torch.Tensor:
        return rng[0] + (rng[1] - rng[0]) * torch.rand(n, cols, device=env.device)

    armature = robot.data.joint_armature.clone()
    for act in actuators:
        cols = act.num_joints
        act.passive_damping[ids] = draw(damping_range, cols)
        act.passive_friction[ids] = draw(friction_range, cols)
        joint_ids = act.joint_indices
        if isinstance(joint_ids, slice):
            joint_ids = torch.arange(robot.num_joints, device=env.device)[joint_ids]
        joint_ids = torch.as_tensor(joint_ids, device=env.device)
        armature[ids[:, None], joint_ids[None, :]] = draw(armature_range, cols)
    robot.write_joint_armature_to_sim(armature[ids], env_ids=ids)
