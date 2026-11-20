"""The torso IMU's terms the actors read: the env's model of /lowstate's imu_state (lowstate_imu.LowstateImu)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["lowstate_gyro", "lowstate_projected_gravity"]


def lowstate_gyro(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The torso IMU's gyroscope as reported (bias, noise, vibration) through the deploy's low-pass (imu_link, rad/s)."""
    return env.lowstate_imu.gyro


def lowstate_projected_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gravity's direction in imu_link from the torso IMU's reported quaternion (lagged, biased, noisy)."""
    return env.lowstate_imu.projected_gravity()
