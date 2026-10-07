"""Pelvis odometry: the pelvis's 6-DoF motion over one policy step, from what the robot measures.

The env moves the arm command by leg odometry (LocoManipMarlEnv._leg_odometry). Where no foot stayed planted it
falls back on this learned estimator: an MLP on the odometry observation group's history window and the action,
whose output is the step's mean linear and angular velocity in the pelvis frame at its start. The env trains it
online on the true motion and saves it beside the skrl checkpoints (<run>/estimator/), with the curriculum state.
"""

from __future__ import annotations

import glob
import os
import re
import torch
import torch.nn as nn
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_from_angle_axis, quat_inv, quat_mul


def pelvis_motion(pos_0, quat_0, pos_1, quat_1, dt: float) -> torch.Tensor:
    """True motion from pose 0 to pose 1, in frame 0, as (v_b, w_b) averaged over dt. Shape (N, 6)."""
    v_b = quat_apply_inverse(quat_0, pos_1 - pos_0) / dt
    w_b = axis_angle_from_quat(quat_mul(quat_inv(quat_0), quat_1)) / dt
    return torch.cat([v_b, w_b], dim=-1)


def motion_to_transform(motion: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(v_b, w_b) over dt -> (delta position, delta quaternion w-first) in the starting frame."""
    delta_pos = motion[:, :3] * dt
    rotvec = motion[:, 3:] * dt
    angle = torch.norm(rotvec, dim=-1)
    return delta_pos, quat_from_angle_axis(angle, rotvec)


class RunningNorm(nn.Module):
    """Per-feature running mean / std of the inputs (parallel Welford). Saved with the model."""

    def __init__(self, size: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean, batch_var, n = x.mean(dim=0), x.var(dim=0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class PelvisMotionEstimator(nn.Module):
    def __init__(self, num_inputs: int, hidden_dims: list[int]):
        super().__init__()
        self.norm = RunningNorm(num_inputs)
        layers, size = [], num_inputs
        for hidden in hidden_dims:
            layers += [nn.Linear(size, hidden), nn.ELU()]
            size = hidden
        layers.append(nn.Linear(size, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class EstimatorTrainer:
    """Ring buffer of (input, true motion) samples and the supervised fit."""

    def __init__(self, cfg: PelvisEstimatorCfg, num_inputs: int, num_envs: int, device: str):
        self.cfg = cfg
        self.model = PelvisMotionEstimator(num_inputs, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.inputs = torch.zeros(cfg.buffer_steps, num_envs, num_inputs, device=device)
        self.labels = torch.zeros(cfg.buffer_steps, num_envs, 6, device=device)
        self.valid = torch.zeros(cfg.buffer_steps, num_envs, dtype=torch.bool, device=device)
        self._slot = 0

    def add(self, inputs: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor):
        self.inputs[self._slot] = inputs
        self.labels[self._slot] = labels
        self.valid[self._slot] = valid
        self._slot = (self._slot + 1) % self.cfg.buffer_steps

    def train(self) -> dict[str, torch.Tensor]:
        x, y = self.inputs[self.valid], self.labels[self.valid]
        if len(x) < self.cfg.mini_batches:
            return {}
        self.model.norm.update(x)
        losses = []
        with torch.enable_grad():
            for _ in range(self.cfg.epochs):
                for idx in torch.randperm(len(x), device=x.device).chunk(self.cfg.mini_batches):
                    loss = (self.model(x[idx]) - y[idx]).square().mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.detach())
        with torch.no_grad():
            error = self.model(x) - y
        return {
            "loss": torch.stack(losses).mean(),
            "lin_vel_error": torch.norm(error[:, :3], dim=-1).mean(),  # m/s
            "ang_vel_error": torch.norm(error[:, 3:], dim=-1).mean(),  # rad/s
        }

    def save(self, path: str, env_state: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "env_state": env_state}, path)

    def load(self, path: str) -> dict:
        """Loads the model and optimizer; returns the saved env state."""
        state = torch.load(path, map_location=self.inputs.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return state["env_state"]


def estimator_checkpoint_for(agent_checkpoint: str) -> str | None:
    """The estimator saved with an skrl checkpoint: <run>/estimator/estimator_<step>.pt, else the newest one."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(agent_checkpoint)))
    step = re.search(r"_(\d+)\.pt$", os.path.basename(agent_checkpoint))
    if step:
        same_step = os.path.join(run_dir, "estimator", f"estimator_{step.group(1)}.pt")
        if os.path.isfile(same_step):
            return same_step
    latest = os.path.join(run_dir, "estimator", "estimator_latest.pt")
    if os.path.isfile(latest):
        return latest
    saved = sorted(glob.glob(os.path.join(run_dir, "estimator", "estimator_*.pt")), key=os.path.getmtime)
    return saved[-1] if saved else None


@configclass
class PelvisEstimatorCfg:
    foot_body_names: list[str] = MISSING
    """The feet, as robot bodies and contact-sensor bodies (leg odometry)."""
    contact_sensor_name: str = "contact_forces"
    stance_force: float = 50.0
    """A foot counts as planted over a step if its contact force stayed above this (N) at every physics substep."""
    train: bool = True
    checkpoint_path: str | None = None
    """Estimator file to start from (the scripts fill it in from an skrl checkpoint)."""
    hidden_dims: list[int] = [256, 128]
    learning_rate: float = 1.0e-3
    buffer_steps: int = 24
    train_every: int = 24
    epochs: int = 2
    mini_batches: int = 4
    save_every: int = 4800
    """Env steps between saves: the skrl checkpoint interval."""
    drift_gate: float = 0.05
    """While training, arm goals follow the estimate only while its mean drift over a whole goal is below this (m)."""
    drift_ema_per_goal: float = 0.001
    """EMA weight of each ended arm goal's drift."""
