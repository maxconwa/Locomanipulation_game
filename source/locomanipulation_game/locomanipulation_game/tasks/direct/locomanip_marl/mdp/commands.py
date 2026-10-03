"""Arm pose targets that the arm is known to reach."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["ReachablePoseCommand", "ReachablePoseCommandCfg"]


class ReachablePoseCommand(CommandTerm):
    """Pose command for one body, drawn from a table of poses the robot reached.

    UniformPoseCommand samples a position box and Euler ranges, so many of its
    targets are out of the arm's reach or inside the torso. This term builds its
    targets from the real kinematics instead. At construction it repeatedly:

      1. sets `joint_names` to uniform random angles inside their soft limits
         (every other joint at default, root at its default pose),
      2. takes one physics step,
      3. records `body_name`'s pose in the root frame,
      4. drops the sample if any of `collision_body_names` is in contact.

    Targets are drawn from that table, so each one was reached once, with no
    collision, by this robot. The root of the H1-2 articulation is the pelvis,
    so the root frame is the pelvis frame.

    The command is (x, y, z, qw, qx, qy, qz) in the root frame, w-first like
    UniformPoseCommand, with qw >= 0. The reward terms in the manager-based
    mdp (ee_position_error, ee_orientation_error) read it in this layout.
    """

    cfg: ReachablePoseCommandCfg

    def __init__(self, cfg: ReachablePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.pose_table, self.default_pose = self._build_table(env)

    def __str__(self) -> str:
        msg = "ReachablePoseCommand:\n"
        msg += f"\tBody: {self.cfg.body_name}\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTable size: {len(self.pose_table)}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command in the root frame, (x, y, z, qw, qx, qy, qz). Shape is (num_envs, 7)."""
        return self.pose_command_b

    """
    Table construction.
    """

    def _build_table(self, env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robot
        joint_ids, _ = robot.find_joints(self.cfg.joint_names, preserve_order=True)
        sensor: ContactSensor = env.scene.sensors[self.cfg.contact_sensor_name]
        check_ids, _ = sensor.find_bodies(self.cfg.collision_body_names, preserve_order=True)

        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        mid = limits.mean(dim=-1)
        half = 0.5 * (limits[:, 1] - limits[:, 0]) * self.cfg.joint_range_scale
        low, high = mid - half, mid + half

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        default_q = robot.data.default_joint_pos.clone()
        zero_qd = torch.zeros_like(robot.data.default_joint_vel)

        # batch 0 is the default pose: rel_default_envs targets, and a check
        # that the contact test passes where we know there is no collision
        default_pose, default_ok = self._reach(env, default_q, zero_qd, root_state, sensor, check_ids)
        if not default_ok.all():
            raise RuntimeError(
                f"{self.cfg.body_name}: the default pose reads as a collision in "
                f"{int((~default_ok).sum())} envs; the contact test is broken."
            )

        poses, total, tried = [], 0, 0
        for _ in range(self.cfg.max_build_batches):
            q = default_q.clone()
            q[:, joint_ids] = low + (high - low) * torch.rand(self.num_envs, len(joint_ids), device=self.device)
            pose_b, ok = self._reach(env, q, zero_qd, root_state, sensor, check_ids)
            poses.append(pose_b[ok])
            total += int(ok.sum())
            tried += self.num_envs
            if total >= self.cfg.table_size:
                break

        # leave the robot where the env's reset expects it
        self._reach(env, default_q, zero_qd, root_state, sensor, check_ids)

        table = torch.cat(poses)[: self.cfg.table_size]
        print(
            f"[INFO] ReachablePoseCommand({self.cfg.body_name}): {len(table)} targets,"
            f" {100.0 * total / tried:.1f}% of {tried} random arm poses collision-free."
            f" Default pose (root frame): {[round(v, 3) for v in default_pose[0].tolist()]}"
        )
        return table, default_pose[0].clone()

    def _reach(self, env, q, qd, root_state, sensor, check_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Puts every env in joint state q, steps once, returns (body pose in root frame, collision-free)."""
        robot = self.robot
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        robot.write_joint_state_to_sim(q, qd)
        robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_pos_w[:, self.body_idx],
            robot.data.body_quat_w[:, self.body_idx],
        )
        peak_force = sensor.data.net_forces_w[:, check_ids].norm(dim=-1).max(dim=1)[0]
        ok = peak_force < self.cfg.collision_force_threshold
        return torch.cat([pos_b, quat_unique(quat_b)], dim=-1), ok

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device)
        idx = torch.randint(0, len(self.pose_table), (len(env_ids),), device=self.device)
        self.pose_command_b[env_ids] = self.pose_table[idx]
        if self.cfg.rel_default_envs > 0.0:
            at_default = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.rel_default_envs]
            self.pose_command_b[at_default] = self.default_pose

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot can be de-initialized while the callback is still subscribed
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


def _frame_marker(prim_path: str) -> VisualizationMarkersCfg:
    marker = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    # replace() is shallow: copy the markers dict before rescaling, or every
    # FRAME_MARKER_CFG user in the process gets the new scale
    marker.markers = dict(marker.markers)
    marker.markers["frame"] = marker.markers["frame"].replace(scale=(0.1, 0.1, 0.1))
    return marker


@configclass
class ReachablePoseCommandCfg(CommandTermCfg):
    class_type: type = ReachablePoseCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    """The body whose pose is commanded."""
    joint_names: list[str] = MISSING
    """Joints randomized to build the table: the arm that moves body_name."""
    collision_body_names: list[str] = MISSING
    """Bodies that must have no contact for a table sample to count."""
    contact_sensor_name: str = "contact_forces"
    collision_force_threshold: float = 1.0

    table_size: int = 50_000
    max_build_batches: int = 2000
    joint_range_scale: float = 1.0
    """Fraction of each joint's soft range to sample, about its centre."""
    rel_default_envs: float = 0.1
    """Fraction of resamples that target the default (arms-down) pose."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING
    current_pose_visualizer_cfg: VisualizationMarkersCfg = MISSING

    def __post_init__(self):
        side = self.body_name.split("_")[0] if isinstance(self.body_name, str) else "ee"
        if isinstance(self.goal_pose_visualizer_cfg, type(MISSING)):
            self.goal_pose_visualizer_cfg = _frame_marker(f"/Visuals/Command/{side}_goal_pose")
        if isinstance(self.current_pose_visualizer_cfg, type(MISSING)):
            self.current_pose_visualizer_cfg = _frame_marker(f"/Visuals/Command/{side}_body_pose")
