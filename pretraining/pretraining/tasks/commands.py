"""The motion command: the reference clip frame each env tracks, its reset to sampled frames, and the targets.

Ported from mjlab (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0),
``src/mjlab/tasks/tracking/mdp/commands.py`` (``MotionCommand``, ``MotionCommandCfg``) onto Isaac Lab's
``CommandTerm``. Same cfg fields, attributes, sampling modes and reset. Differences: the motion is
``pretraining.motions.MotionLoader`` (tracked-body states computed by forward kinematics at load, so
``set_time_steps`` only moves the frame); the robot's joints are read and written in ``JOINT_NAMES`` order (the
articulation also has the gripper hinges); the root state is written as the root link's (mjlab's free-joint
velocity is the link's, Isaac Lab's ``write_root_state_to_sim`` would take it as the COM's); mjlab's ``sim.forward()``
after a reset is ``refresh_robot_state`` (Isaac Lab's step does not forward the reset envs before the command
update); the viz and GUI code is dropped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, sample_uniform, yaw_quat

from pretraining.constants import JOINT_NAMES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.motions import MotionLoader
from pretraining.tasks.metrics import ERROR_METRICS, tracking_errors

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

# The articulation's body buffers that a kinematic refresh makes stale (those its state writers invalidate).
_BODY_BUFFERS = (
    "_body_com_vel_w", "_body_link_vel_w", "_body_com_pose_b", "_body_com_pose_w", "_body_link_pose_w",
    "_body_state_w", "_body_link_state_w", "_body_com_state_w",
)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.entity_name]
        self.joint_ids = torch.tensor(
            self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(list(self.cfg.body_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        fk = ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), self.cfg.body_names, device=self.device)
        motion_file = self.cfg.motion_file
        if isinstance(motion_file, (list, tuple)):
            motion_file = [Path(f) for f in motion_file]
        self.motion = MotionLoader(motion_file, fk, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (0.2 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        # Per-bin distribution for sampling_mode="external"; the runner pushes this in
        # via set_sampling_distribution(). Uniform until the first update.
        self._external_probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        for name in ERROR_METRICS:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self._pending_forward = False

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_ids]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics.update(tracking_errors(self))

    def _sample_bins(self, env_ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Draw reset frames from a normalized per-bin distribution."""
        sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)
        # Integer, multiply-first bin->frame map: exact for any bin count, and bounded.
        total = self.motion.time_step_total - 2
        b = sampled_bins.clamp_(max=self.bin_count - 1)
        lo = (b * total) // self.bin_count
        span = (((b + 1) * total) // self.bin_count - lo).clamp_(min=1)
        jitter = (sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * span).long()
        entropy = -(probs * (probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        return (lo + jitter).clamp_(max=total)

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample. Clip the per-bin failure count at adaptive_clip*mean so no single
        # bin can dominate the distribution (prevents runaway concentration/forgetting).
        failed = self.bin_failed_count
        if self.cfg.adaptive_clip > 0:
            failed = torch.clamp(failed, max=self.cfg.adaptive_clip * failed.mean())
        sampling_probabilities = failed + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        return self._sample_bins(env_ids, sampling_probabilities)

    def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
        self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
        self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.
        return torch.randint(0, self.motion.time_step_total - 1, (len(env_ids),), device=self.device)

    def set_sampling_distribution(self, probs: torch.Tensor) -> None:
        """Set the per-bin reset distribution used by ``sampling_mode='external'``.

        ``probs`` must have length ``bin_count``; it is moved to device, clamped
        non-negative and renormalized. Invalid inputs are ignored so the previous
        (or uniform) distribution stays in effect.
        """
        probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
        total = probs.sum()
        if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
            return
        self._external_probs = probs / total

    def _external_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample reset frames from the externally-provided distribution."""
        return self._sample_bins(env_ids, self._external_probs)

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Clip joint positions and write root + joint state to sim."""
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_ids, env_ids=env_ids)

        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_link_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.sampling_mode == "start":
            frames = torch.zeros_like(env_ids)
        elif self.cfg.sampling_mode == "uniform":
            frames = self._uniform_sampling(env_ids)
        elif self.cfg.sampling_mode == "external":
            frames = self._external_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive"
            frames = self._adaptive_sampling(env_ids)

        on_last = torch.isin(frames, self.motion.motion_ends - 1)
        frames = torch.where(on_last, frames - 1, frames)
        self.set_time_steps(env_ids, frames)

        root_pos = self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.root_quat_w[frames]
        root_lin_vel = self.motion.root_lin_vel_w[frames]
        root_ang_vel = self.motion.root_ang_vel_w[frames]

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rand_samples[:, :3]
        root_ang_vel = root_ang_vel + rand_samples[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]

        joint_pos += sample_uniform(
            self.cfg.joint_position_range[0], self.cfg.joint_position_range[1], joint_pos.shape, device=self.device
        )

        self._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )
        self._pending_forward = True

    def set_time_steps(self, env_ids: torch.Tensor | slice, frames: torch.Tensor) -> None:
        """Move ``env_ids`` to ``frames`` (the tracked bodies' states there are read from the motion)."""
        self.time_steps[env_ids] = frames

    def refresh_robot_state(self) -> None:
        """Recompute the articulation's link poses and velocities from the joint and root state written to it.

        mjlab's ``sim.forward()``: Isaac Lab's state writers update the root and joint buffers only, and PhysX moves
        the links at the next step or kinematic update."""
        self._env.sim.physics_sim_view.update_articulations_kinematic()
        for name in _BODY_BUFFERS:
            getattr(self.robot.data, name).timestamp = -1.0

    def update_relative_body_poses(self) -> None:
        """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

        Called after ``reset_to_frame`` so that termination checks that
        compare relative body positions see the correct state.
        """
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _update_command(self):
        self.time_steps += 1
        wrap_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if wrap_ids.numel() > 0:
            self._resample_command(wrap_ids)

        # A resample (at reset, wraparound or timer expiry) wrote the root and joint state; refresh the links so
        # update_relative_body_poses and the observations read the post-teleport robot.
        if self._pending_forward:
            self._pending_forward = False
            self.refresh_robot_state()

        self.update_relative_body_poses()

        # Fold failure counts into the EMA once per step.
        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Reset to exact reference state at a specific frame.

        Like ``_resample_command`` but deterministic: no random
        perturbations to pose, velocity, or joint positions.
        """
        frames = torch.full_like(env_ids, frame)
        self.set_time_steps(env_ids, frames)
        self._write_reference_state_to_sim(
            env_ids,
            self.motion.root_pos_w[frames] + self._env.scene.env_origins[env_ids],
            self.motion.root_quat_w[frames],
            self.motion.root_lin_vel_w[frames],
            self.motion.root_ang_vel_w[frames],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand

    motion_file: str | list[str] = MISSING
    """A motion directory of ``.npz`` clips, or a list of clip files."""
    anchor_body_name: str = MISSING
    body_names: tuple[str, ...] = MISSING
    entity_name: str = "robot"
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_clip: float = 100.0
    sampling_mode: Literal["adaptive", "uniform", "start", "external"] = "adaptive"
