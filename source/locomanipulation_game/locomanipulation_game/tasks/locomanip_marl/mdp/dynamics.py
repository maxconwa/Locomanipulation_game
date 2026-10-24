"""Per-env joint dynamics: passive damping, dry friction and armature around GOLEM's Robocasa MJCF values, and the
motors' gains and strength."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.actuators import IdealPDActuator
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


def randomize_actuator_scales(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gain_range: tuple[float, float],
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset event: per env and joint of every explicit PD actuator (IdealPDActuator, so not the implicit grippers),
    kp and kd x U(gain_range) together and the motor's torque x U(strength_range) (kp, kd and the effort limit
    together), from the actuator's nominal values. The torque bounds and GOLEM's e-stop checks keep reading the
    nominal gains (data.default_joint_stiffness), as the deploy knows only those."""
    robot = env.scene[asset_cfg.name]
    actuators = {n: a for n, a in robot.actuators.items() if isinstance(a, IdealPDActuator)}
    if not hasattr(env, "_actuator_nominal"):
        env._actuator_nominal = {n: (a.stiffness.clone(), a.damping.clone(), a.effort_limit.clone())
                                 for n, a in actuators.items()}
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    n = len(ids)
    for name, act in actuators.items():
        kp, kd, limit = env._actuator_nominal[name]
        cols = act.num_joints
        gain = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(n, cols, device=env.device)
        strength = strength_range[0] + (strength_range[1] - strength_range[0]) * torch.rand(n, cols, device=env.device)
        act.stiffness[ids] = kp[ids] * gain * strength
        act.damping[ids] = kd[ids] * gain * strength
        act.effort_limit[ids] = limit[ids] * strength
