"""Pelvis odometry: the pelvis's 6-DoF motion over one policy step, from what the robot measures.

The env moves the arm command by this learned estimator (odometry "learned"): an MLP on the odometry observation
group's history window and the action, whose output is the step's mean linear and angular velocity in the pelvis
frame at its start. Or by leg odometry (odometry "legs", LocoManipMarlEnv._leg_odometry), with the estimator only
where no foot stayed planted. With vertical_from_feet the step's vertical motion comes from the pelvis's height above
the lowest ankle instead of the estimator, and with tilt_from_gravity its roll and pitch from gravity
(LocoManipMarlEnv._from_feet_and_gravity): both are absolute, so height and tilt never drift; only the horizontal
motion and yaw are integrated from the estimate.

The env trains it online on the true motion and saves it beside the skrl checkpoints (<run>/estimator/), with the
curriculum state and its input window. The fit charges each step's error and, window_loss_weight times, the mean
error over every window_steps consecutive steps in the world frame: integrated, a small steady bias is what drifts the
arm goals, and per step it hides under the noise.
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
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
)


def align_tilt(delta_quat: torch.Tensor, up_prev: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """delta_quat (previous pelvis frame <- new, w first) turned by the smallest rotation that makes it carry the
    world's vertical in the new frame, up, onto the one in the previous frame, up_prev: its roll and pitch then agree
    with gravity's, and its yaw is kept."""
    carried = quat_apply(delta_quat, up)
    axis = torch.cross(carried, up_prev, dim=-1)
    sin, cos = torch.norm(axis, dim=-1), (carried * up_prev).sum(-1)
    axis = axis / sin.clamp(min=1e-9).unsqueeze(-1)
    return quat_mul(quat_from_angle_axis(torch.atan2(sin, cos), axis), delta_quat)


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
    """Ring buffer of the last buffer_steps steps of every env, (input, true motion, pelvis orientation at the step's
    start), kept in time order for the windowed loss, and the supervised fit."""

    def __init__(self, cfg: PelvisEstimatorCfg, num_inputs: int, num_envs: int, device: str, history_frames: int):
        self.cfg = cfg
        self.history_frames = history_frames  # the odometry group's window, saved with the model
        self.model = PelvisMotionEstimator(num_inputs, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.inputs = torch.zeros(cfg.buffer_steps, num_envs, num_inputs, device=device)
        self.labels = torch.zeros(cfg.buffer_steps, num_envs, 6, device=device)
        self.quats = torch.zeros(cfg.buffer_steps, num_envs, 4, device=device)
        self.quats[..., 0] = 1.0
        self.valid = torch.zeros(cfg.buffer_steps, num_envs, dtype=torch.bool, device=device)
        self._slot = 0
        if cfg.window_steps > cfg.buffer_steps:
            raise ValueError(f"window_steps {cfg.window_steps} exceeds buffer_steps {cfg.buffer_steps}")

    def add(self, inputs: torch.Tensor, labels: torch.Tensor, quat: torch.Tensor, valid: torch.Tensor):
        """quat: the pelvis's world orientation at the step's start, the frame the labels are in."""
        self.inputs[self._slot] = inputs
        self.labels[self._slot] = labels
        self.quats[self._slot] = quat
        self.valid[self._slot] = valid
        self._slot = (self._slot + 1) % self.cfg.buffer_steps

    def train(self) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        steps, num_envs = self.valid.shape
        order = (torch.arange(steps, device=self.valid.device) + self._slot) % steps  # oldest first
        valid = self.valid[order]
        if valid.sum() < cfg.mini_batches:
            return {}
        # a window counts only if all its steps are valid: one episode, all in arm goals with train_on_arm_goals_only
        k = cfg.window_steps
        count = torch.cat([valid.new_zeros(1, num_envs, dtype=torch.float), valid.float().cumsum(0)])
        window_valid = (count[k:] - count[:-k]) == k  # (steps - k + 1, N), by first step
        chunks = torch.randperm(num_envs, device=valid.device).chunk(cfg.mini_batches)
        for envs in chunks:
            x = self.inputs[order[:, None], envs[None, :]]
            self.model.norm.update(x[valid[:, envs]])
        losses, step_errors, window_errors = [], [], []
        with torch.enable_grad():
            for epoch in range(cfg.epochs):
                for envs in chunks if epoch == 0 else torch.randperm(num_envs, device=valid.device).chunk(cfg.mini_batches):
                    mask, wmask = valid[:, envs], window_valid[:, envs]
                    if not mask.any():
                        continue
                    idx = (order[:, None], envs[None, :])
                    error = self.model(self.inputs[idx]) - self.labels[idx]  # (steps, n, 6), each step's own frame
                    loss = error[mask].square().mean()
                    if cfg.window_loss_weight > 0.0 and wmask.any():
                        mean_error = self._window_mean(error, self.quats[idx], mask, k)
                        loss = loss + cfg.window_loss_weight * mean_error[wmask].square().mean()
                        window_errors.append(mean_error[wmask].detach())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.detach())
                    step_errors.append(error[mask].detach())
        if not losses:
            return {}
        error = torch.cat(step_errors)
        stats = {
            "loss": torch.stack(losses).mean(),
            "lin_vel_error": torch.norm(error[:, :3], dim=-1).mean(),  # m/s
            "ang_vel_error": torch.norm(error[:, 3:], dim=-1).mean(),  # rad/s
        }
        if window_errors:
            window = torch.cat(window_errors)
            # the mean velocity error over a window: the drift rate a steady bias would give (m/s, rad/s)
            stats["window_lin_vel_error"] = torch.norm(window[:, :3], dim=-1).mean()
            stats["window_vertical_vel_error"] = window[:, 2].abs().mean()
            stats["window_ang_vel_error"] = torch.norm(window[:, 3:], dim=-1).mean()
        return stats

    @staticmethod
    def _window_mean(error: torch.Tensor, quats: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        """Mean of the per-step errors over every k consecutive steps, rotated into the world frame first so that
        they add up as the drift does. (steps, n, 6) -> (steps - k + 1, n, 6), by the window's first step."""
        q = quats.reshape(-1, 4)
        world = torch.cat(
            [quat_apply(q, error[..., :3].reshape(-1, 3)), quat_apply(q, error[..., 3:].reshape(-1, 3))], dim=-1
        ).view(error.shape)
        world = torch.where(mask.unsqueeze(-1), world, torch.zeros_like(world))
        total = torch.cat([world.new_zeros(1, *world.shape[1:]), world.cumsum(0)])
        return (total[k:] - total[:-k]) / k

    def save(self, path: str, env_state: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "env_state": env_state,
                    "history_frames": self.history_frames}, path)

    def load(self, path: str) -> dict:
        """Loads the model and optimizer; returns the saved env state."""
        state = torch.load(path, map_location=self.inputs.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return state["env_state"]


def saved_history_frames(path: str) -> int:
    """The odometry window an estimator file was trained on; files from before it was saved used 4 frames."""
    return int(torch.load(path, map_location="cpu", weights_only=False).get("history_frames", 4))


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
    odometry: str = "learned"
    """What moves the arm command: "learned" (the estimator), "legs" (attitude and a planted foot) or "truth" (the true
    pelvis motion and no estimator at all, as the deploy's FAST-LIO odometry stands in for it; the env still saves
    its curriculum state under estimator/)."""
    foot_body_names: list[str] | None = None
    """The feet, as robot bodies (and, for odometry "legs", contact-sensor bodies): odometry "legs" and
    vertical_from_feet need them."""
    vertical_from_feet: bool = False
    """Take each step's vertical pelvis motion from the feet instead of the estimator: the change in the pelvis's height
    above the lowest ankle along gravity, from forward kinematics and the attitude. That height is absolute, so it
    never drifts; the estimator keeps the horizontal motion and the rotation."""
    tilt_from_gravity: bool = False
    """Turn each step's estimated rotation so that its roll and pitch match the change in gravity's direction in the
    pelvis frame (on the robot, the IMU's): tilt stays absolute and only yaw is integrated. A tilt error swings the
    arm goals up or down by their distance from the pelvis."""
    contact_sensor_name: str = "contact_forces"
    stance_force: float = 50.0
    """A foot counts as planted over a step if its contact force stayed above this (N) at every physics substep."""
    train: bool = True
    checkpoint_path: str | None = None
    """Estimator file to start from (the scripts fill it in from an skrl checkpoint)."""
    hidden_dims: list[int] = [256, 128]
    learning_rate: float = 1.0e-3
    buffer_steps: int = 100
    """Steps of every env kept for the fit, in time order (the windowed loss needs them consecutive)."""
    train_every: int = 50
    epochs: int = 1
    """Passes per fit. Each step stays in the buffer for buffer_steps / train_every = 2 fits."""
    mini_batches: int = 4
    window_steps: int = 50
    """The windowed loss's window (policy steps; 50 = 1 s)."""
    window_loss_weight: float = 10.0
    """Weight of the squared mean error over windows of window_steps steps, beside the per-step squared error (1).
    A steady bias counts in full in both, noise only 1/window_steps as much in the window term. 0 turns it off."""
    save_every: int = 4800
    """Env steps between saves: the skrl checkpoint interval."""
    drift_gate: float = 0.05
    """While training, arm goals follow the estimate only while its mean drift over a whole goal is below this (m)."""
    true_odometry_while_training: bool = False
    """While training, move every arm goal by the true pelvis motion (the drift gate stays shut); the estimator still
    trains on it, and outside training the estimate moves the command as before."""
    train_on_arm_goals_only: bool = False
    """Fit the estimator only on steps during arm goals, the only motion it moves the command by (standing,
    crouching and reaching), not walking, whose larger and faster motion it is never used for."""
    drift_ema_per_goal: float = 0.001
    """EMA weight of each ended arm goal's drift."""
