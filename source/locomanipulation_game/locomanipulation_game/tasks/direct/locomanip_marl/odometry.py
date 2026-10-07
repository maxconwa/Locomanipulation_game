"""Pelvis odometry: estimate the pelvis's 6-DoF motion over one policy step from what the robot measures.

An estimate, not a prediction: it runs after the step, on the measurements
up to its end (the odometry group's last history_length frames of IMU,
joint states and leg torques, which include t and t+1) plus the action
applied in between. The history lets it filter sensor noise; the leg torques
carry the contact information that says which foot is planted. The output is the motion expressed in the pelvis frame at
t, as the average linear and angular velocity over the step (m/s, rad/s):
well-scaled regression targets, and dt * velocity is the transform.

Trained online by supervised learning inside the env: the sim knows the true
pelvis pose, so every step yields a labelled sample. The env keeps the last
buffer_steps steps and fits the estimator every train_every steps; skrl never
sees it. The env saves it beside the skrl checkpoints (estimator/ in the run
directory) and play.py loads it from there.

With odometry "legs" the arm commands follow the attitude and a planted
foot's kinematics instead (LocoManipMarlEnv._leg_odometry), and this
estimator only covers the steps where no foot stayed planted. Integrated over
a 4 s held goal its small per-step errors (run H: 0.03 m/s, 0.03 rad/s)
moved the command 10-15 cm.
"""

from __future__ import annotations

import glob
import os
import re
import torch
import torch.nn as nn

from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
)


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

    def train(self) -> dict[str, float]:
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

    def save(self, path: str, env_state: dict | None = None):
        """Model, optimizer and the env-side state (curriculum levels, gate) to resume or play with."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        if env_state is not None:
            state["env_state"] = env_state
        torch.save(state, path)

    def load(self, path: str) -> tuple[bool, dict | None]:
        """Returns (whether the model loaded, the saved env state or None).

        A model saved with other inputs (e.g. before the history window) is
        skipped with a warning, so its run's policies can still be resumed;
        the estimator then starts fresh behind the drift gate.
        """
        state = torch.load(path, map_location=self.inputs.device)
        try:
            self.model.load_state_dict(state["model"])
        except RuntimeError as error:
            print(f"[WARNING] Pelvis estimator in {path} doesn't fit this model, starting fresh: {str(error)[:200]}")
            return False, state.get("env_state")
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        return True, state.get("env_state")


def estimator_checkpoint_for(agent_checkpoint: str) -> str | None:
    """The estimator saved with an skrl checkpoint: <run>/estimator/estimator_<step>.pt, else _latest.pt."""
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
    use_estimate: bool = True
    """False: arm commands always follow the true pelvis motion (the ground-truth ablation)."""
    odometry: str = "learned"
    """Where the estimate comes from: "learned" (the MLP below) or "legs" (rotation from the attitude, translation
    from a planted foot's kinematics, the MLP's where no foot stayed planted; LocoManipMarlEnv._leg_odometry)."""
    foot_body_names: list[str] | None = None
    """odometry "legs": the feet, as robot bodies and contact-sensor bodies."""
    contact_sensor_name: str = "contact_forces"
    stance_force: float = 50.0
    """odometry "legs": a foot counts as planted over a step if its contact force stayed above this (N) at every
    physics substep. Standing, each foot of the H1-2 carries ~350 N."""
    train: bool = True
    checkpoint_path: str | None = None
    """Estimator to start from (train.py / play.py fill this from an skrl checkpoint)."""
    hidden_dims: list[int] = [256, 128]
    learning_rate: float = 1.0e-3
    buffer_steps: int = 24
    train_every: int = 24
    epochs: int = 2
    mini_batches: int = 4
    # Teacher forcing: each arm goal follows the estimate with probability
    # estimate_prob, 0 until warmup_steps, then rising linearly to 1 over
    # ramp_steps; and 0 whenever the drift gate is closed.
    warmup_steps: int = 4800
    ramp_steps: int = 19200
    save_every: int = 4800
    """Env steps between saves; 4800 matches the skrl checkpoint interval."""
    drift_gate: float = 0.05
    """Arm goals follow the estimate only while its mean drift over a whole goal is below this (m): the reach tolerance."""
    drift_ema_per_goal: float = 0.001
    """EMA weight of each ended arm goal's drift."""
