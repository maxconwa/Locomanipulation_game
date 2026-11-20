"""The H1-2's torso IMU as /lowstate's imu_state reports it, the IMU every task's actors read.

The IMU is imu_link, fixed to torso_link at (-0.04452, -0.01891, 0.27756) m, unrotated (GOLEM's URDF, CL_Assets):
scene["imu"]. /lowstate gives at 500 Hz the firmware filter's quaternion (imu_link to world), the gyroscope and the
accelerometer. The actors read:
- projected_gravity from the quaternion: the true orientation lag_s ago (drawn per reset), times a bias rotation drawn
  per reset (body frame: what it reads when the torso is level), times a small noise rotation each step;
- base_ang_vel from the gyroscope: the true rate plus a bias drawn per reset, noise, and a vibration that grows with
  the legs' target rate (the torso's shake when the legs bang-bang), through the deploy's 8 Hz low-pass
  (imu_filter.GyroLowpass), sampled at every physics substep as /lowstate's 500 Hz;
- not the accelerometer: it carried the vibration on the robot (2026-11-20: x std 44 m/s^2, peaks 164), and the
  quaternion already gives gravity.

Measured on the robot (lmg_legsenergy200k_s43_2026-11-20_1157, still on the hoist and in the pre-pose; the 11-17 and
11-18 IMU bags agree): the quaternion reads pitch -2.3 deg and roll -0.2 deg against the torso's own accelerometer,
pitch -3.2 to -3.4 deg and roll -0.05 to +0.2 deg against the head MID-360's; its noise 0.02-0.07 deg; the gyroscope's
bias about (-0.001, +0.0045, +0.005 to +0.015) rad/s and noise (0.012, 0.009, 0.010-0.023) rad/s; during the thrash
the torso gyro's std 0.4-0.8 rad/s. Pitch is positive nose down, roll positive tilting right (Isaac's x-y-z Euler).
GOLEM applies no trim to a policy trained on this model (export.yaml imu.source: lowstate).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

from .imu_filter import GyroLowpass


@configclass
class LowstateImuCfg:
    """The model's draws (the module docstring); every range is drawn uniformly per env at each reset."""

    quat_bias_deg: dict[str, tuple[float, float]] = {"roll": (-0.6, 0.4), "pitch": (-3.6, -2.0), "yaw": (-1.0, 1.0)}
    """What the quaternion reads with the torso level: centred on the measured pitch -2.8 (-2.3 to -3.3 by the two
    references) and roll -0.1, with a margin."""
    quat_noise_deg: float = 0.1
    """Std of a fresh roll and pitch noise rotation each policy step (measured 0.02-0.07)."""
    quat_lag_s: tuple[float, float] = (0.0, 0.02)
    """The firmware filter's lag under motion (unmeasured)."""
    gyro_bias: tuple[float, float, float] = (0.005, 0.005, 0.02)
    """rad/s, +- per axis (x, y, z)."""
    gyro_noise: float = 0.03
    """rad/s, std per sample: 1.5x the measured 0.009-0.023, a margin."""
    vibration_per_target_rate: float = 4.0
    """rad/s of gyro vibration std per rad of rms leg target change per policy step: 0.12-0.2 rad per step (the
    2026-11-20 bang-bang) gives 0.5-0.8 rad/s, the measured 0.4-0.8."""
    vibration_gain: tuple[float, float] = (0.5, 1.5)
    vibration_hz: tuple[float, float] = (18.0, 26.0)
    """The shake's frequency (the head's was 21-23 Hz)."""
    vibration_axes: tuple[float, float, float] = (0.6, 1.0, 0.4)
    """Its share per gyro axis (x, y, z): pitch the largest, as on the robot."""


class LowstateImu:
    """num_envs torso IMUs (LowstateImuCfg). sample() at every physics substep; the outputs at the policy step."""

    def __init__(self, cfg: LowstateImuCfg, gyro_lowpass_hz: float, physics_dt: float, num_envs: int, device):
        self.cfg, self.dt, self.n, self.device = cfg, float(physics_dt), num_envs, device
        self.depth = int(math.ceil(cfg.quat_lag_s[1] / physics_dt)) + 1
        self._quat_hist = torch.zeros(num_envs, self.depth, 4, device=device)
        self._quat_hist[..., 0] = 1.0
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        zeros = lambda *s: torch.zeros(num_envs, *s, device=device)  # noqa: E731
        self.lag = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.bias_quat = zeros(4)
        self.bias_quat[:, 0] = 1.0
        self.gyro_bias, self.vib_phase = zeros(3), zeros(3)
        self.vib_gain, self.vib_hz, self.time = zeros(), zeros(), zeros()
        self.target_rate = zeros()  # the env sets it at every policy step (rms leg target change, rad)
        self.gyro_lowpass = GyroLowpass(gyro_lowpass_hz, physics_dt, num_envs, device)
        self._axes = torch.tensor(cfg.vibration_axes, device=device)
        self.reset(None)

    def reset(self, env_ids):
        ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        m, cfg = len(ids), self.cfg
        u = lambda low, high, *s: low + (high - low) * torch.rand(m, *s, device=self.device)  # noqa: E731
        bias = cfg.quat_bias_deg
        self.bias_quat[ids] = quat_from_euler_xyz(*(u(*(math.radians(v) for v in bias[k])) for k in ("roll", "pitch", "yaw")))
        self.lag[ids] = torch.round(u(*cfg.quat_lag_s) / self.dt).long().clamp(0, self.depth - 1)
        limit = torch.tensor(cfg.gyro_bias, device=self.device)
        self.gyro_bias[ids] = (2.0 * torch.rand(m, 3, device=self.device) - 1.0) * limit
        self.vib_gain[ids] = u(*cfg.vibration_gain)
        self.vib_hz[ids] = u(*cfg.vibration_hz)
        self.vib_phase[ids] = u(0.0, 2.0 * math.pi, 3)
        self.time[ids] = 0.0
        self.target_rate[ids] = 0.0
        self._fresh[ids] = True
        self.gyro_lowpass.reset(ids)

    def sample(self, quat_w: torch.Tensor, ang_vel_b: torch.Tensor):
        """One physics substep: the true orientation into the lag history, the gyroscope through the low-pass."""
        fresh = self._fresh.view(-1, 1, 1)
        rolled = self._quat_hist.roll(1, dims=1)
        rolled[:, 0] = quat_w
        self._quat_hist = torch.where(fresh, quat_w.unsqueeze(1).expand_as(self._quat_hist), rolled)
        self._fresh[:] = False
        cfg = self.cfg
        std = cfg.vibration_per_target_rate * self.vib_gain * self.target_rate
        wave = torch.sin(2.0 * math.pi * self.vib_hz.unsqueeze(1) * self.time.unsqueeze(1) + self.vib_phase)
        vibration = math.sqrt(2.0) * std.unsqueeze(1) * self._axes * wave
        gyro = ang_vel_b + self.gyro_bias + cfg.gyro_noise * torch.randn_like(ang_vel_b) + vibration
        self.gyro_lowpass(gyro)
        self.time += self.dt

    @property
    def gyro(self) -> torch.Tensor:
        """The low-passed gyroscope (imu_link frame, rad/s)."""
        return self.gyro_lowpass.output

    def quaternion(self) -> torch.Tensor:
        """The reported quaternion (w, x, y, z; imu_link to world): lagged, biased, with fresh noise."""
        rows = torch.arange(self.n, device=self.device)
        q = quat_mul(self._quat_hist[rows, self.lag], self.bias_quat)
        noise = torch.randn(self.n, 2, device=self.device) * math.radians(self.cfg.quat_noise_deg)
        return quat_mul(q, quat_from_euler_xyz(noise[:, 0], noise[:, 1], torch.zeros_like(noise[:, 0])))

    def projected_gravity(self) -> torch.Tensor:
        """Gravity's direction in the reported imu_link frame."""
        down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.n, 3)
        return quat_apply_inverse(self.quaternion(), down)
