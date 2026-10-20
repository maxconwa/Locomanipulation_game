"""The head IMU's terms: the low-passed gyro and the biased accelerometer the policies and the estimator read, and the
per-env mount and bias randomization.

The IMU is the MID-360's chip, scene["imu"] at livox_link/livox_imu_link (CL_Assets ed592de: the measured mount,
its lever arm and the chip's 0.68 deg tilt are in the asset). Its angular velocity goes through the deploy's gyro
low-pass (imu_filter.GyroLowpass, run by the env at every physics substep); its projected gravity does not.
"""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

G = 9.81  # m/s^2 per g


def imu_ang_vel_lowpass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """The IMU's angular velocity (sensor frame) through the env's gyro low-pass; the raw rate without one."""
    lowpass = getattr(env, "gyro_lowpass", None)
    if lowpass is None:
        return env.scene[asset_cfg.name].data.ang_vel_b
    return lowpass.output


def imu_lin_acc_biased(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """The IMU's specific force (sensor frame, m/s^2) plus the env's accelerometer bias (randomize_imu_mount)."""
    acc = env.scene[asset_cfg.name].data.lin_acc_b
    bias = getattr(env, "imu_acc_bias", None)
    return acc if bias is None else acc + bias


def randomize_imu_mount(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    acc_bias_g: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
):
    """Reset event: tilt each env's IMU from its nominal mount by U(+-roll, +-pitch, +-yaw) deg (in the sensor frame,
    rotating its attitude, gyro and accelerometer alike), and draw a horizontal (sensor x, y) accelerometer bias of
    U(+-acc_bias_g) g. The nominal mount is the IMU's offset as the sensor composed it from the asset at startup."""
    imu = env.scene[asset_cfg.name]
    if not hasattr(env, "imu_nominal_quat"):
        env.imu_nominal_quat = imu._offset_quat_b.clone()
        env.imu_acc_bias = torch.zeros(env.num_envs, 3, device=env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    n = len(ids)

    def draw(limit_deg: float) -> torch.Tensor:
        return (2.0 * torch.rand(n, device=env.device) - 1.0) * math.radians(limit_deg)

    tilt = quat_from_euler_xyz(draw(roll_deg), draw(pitch_deg), draw(yaw_deg))
    imu._offset_quat_b[ids] = quat_mul(env.imu_nominal_quat[ids], tilt)
    env.imu_acc_bias[ids] = 0.0
    env.imu_acc_bias[ids, :2] = (2.0 * torch.rand(n, 2, device=env.device) - 1.0) * acc_bias_g * G
