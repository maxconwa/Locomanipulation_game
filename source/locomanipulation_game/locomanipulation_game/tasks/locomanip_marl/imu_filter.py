"""The policy's gyro low-pass, batched: GOLEM lmg_deploy.livox_imu.GyroLowpass (556a9e7) in torch, one filter per env.

Per axis, a 2nd-order Butterworth at cutoff_hz: y'' + 2 zeta w y' + w^2 y = w^2 u, w = 2 pi cutoff_hz,
zeta = 1/sqrt(2), state x = (y, y'), discretized exactly with a zero-order hold on u over the sample's dt:
x <- Ad x + Bd u, Ad = exp(A dt), Bd = A^-1 (Ad - I) B. It starts, and restarts after reset(), at x = (u, 0) on the
next sample. The env feeds it the IMU's angular velocity at every physics substep, as the deploy feeds it /livox/imu
at 200 Hz; the same continuous filter at either rate.
"""

from __future__ import annotations

import math

import torch

ZETA = 1.0 / math.sqrt(2.0)


def discrete_matrices(cutoff_hz: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(Ad (2, 2), Bd (2,)) in float64: exp(A dt) in closed form (underdamped) and A^-1 (exp(A dt) - I) B."""
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, not {cutoff_hz}")
    w = 2.0 * math.pi * float(cutoff_hz)
    sigma, wd, w2 = ZETA * w, w * math.sqrt(1.0 - ZETA**2), w * w
    e, c, sn = math.exp(-sigma * dt), math.cos(wd * dt), math.sin(wd * dt)
    k = sigma / wd
    ad = e * torch.tensor([[c + k * sn, sn / wd], [-w2 * sn / wd, c - k * sn]], dtype=torch.float64)
    a = torch.tensor([[0.0, 1.0], [-w2, -2.0 * ZETA * w]], dtype=torch.float64)
    b = torch.tensor([0.0, w2], dtype=torch.float64)
    bd = torch.linalg.solve(a, (ad - torch.eye(2, dtype=torch.float64)) @ b)
    return ad, bd


class GyroLowpass:
    """num_envs filters of axes channels each, at a fixed sample dt."""

    def __init__(self, cutoff_hz: float, dt: float, num_envs: int, device, axes: int = 3):
        self.cutoff_hz, self.dt = float(cutoff_hz), float(dt)
        ad, bd = discrete_matrices(cutoff_hz, dt)
        self._ad, self._bd = ad.to(device), bd.to(device)
        self._y = torch.zeros(num_envs, axes, dtype=torch.float64, device=device)
        self._dy = torch.zeros_like(self._y)
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.output = torch.zeros(num_envs, axes, device=device)

    def reset(self, env_ids=None):
        """The given envs restart at x = (u, 0) on their next sample."""
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """Filtered value after one sample u (num_envs, axes) held over dt."""
        u64 = u.to(torch.float64)
        ad, bd = self._ad, self._bd
        y = ad[0, 0] * self._y + ad[0, 1] * self._dy + bd[0] * u64
        dy = ad[1, 0] * self._y + ad[1, 1] * self._dy + bd[1] * u64
        fresh = self._fresh.unsqueeze(-1)
        self._y = torch.where(fresh, u64, y)
        self._dy = torch.where(fresh, torch.zeros_like(dy), dy)
        self._fresh[:] = False
        self.output = self._y.to(u.dtype)
        return self.output
