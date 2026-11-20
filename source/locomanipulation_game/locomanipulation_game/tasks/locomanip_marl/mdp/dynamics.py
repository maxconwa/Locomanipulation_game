"""The motors' per-env gains and strength."""

from __future__ import annotations

import re

import torch
from typing import TYPE_CHECKING

from isaaclab.actuators import IdealPDActuator
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_actuator_scales(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gain_range: tuple[float, float],
    strength_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_gain_ranges: dict[str, tuple[float, float]] | None = None,
):
    """Reset event: per env and joint of every explicit PD actuator (IdealPDActuator, so not the implicit grippers),
    kp and kd x U(gain_range) together and the motor's torque x U(strength_range) (kp, kd and the effort limit
    together), from the actuator's nominal values. The torque bounds and GOLEM's e-stop checks keep reading the
    nominal gains (data.default_joint_stiffness), as the deploy knows only those. joint_gain_ranges (joint name
    pattern -> range) replaces gain_range for the joints it names."""
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
        low = torch.full((cols,), float(gain_range[0]), device=env.device)
        high = torch.full((cols,), float(gain_range[1]), device=env.device)
        for pattern, (lo, hi) in (joint_gain_ranges or {}).items():
            for i, joint in enumerate(act.joint_names):
                if re.fullmatch(pattern, joint):
                    low[i], high[i] = lo, hi
        gain = low + (high - low) * torch.rand(n, cols, device=env.device)
        strength = strength_range[0] + (strength_range[1] - strength_range[0]) * torch.rand(n, cols, device=env.device)
        act.stiffness[ids] = kp[ids] * gain * strength
        act.damping[ids] = kd[ids] * gain * strength
        act.effort_limit[ids] = limit[ids] * strength
