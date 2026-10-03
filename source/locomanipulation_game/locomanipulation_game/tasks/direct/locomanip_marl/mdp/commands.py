"""Commands for the two-agent task: navigation OR an arm goal, never both.

ArmTargetsCommand decides, at every command event, which kind of goal an env
gets next:

  * navigation: the velocity command (ModalVelocityCommand) is live and the
    arms hold their rest pose, fixed in the pelvis frame;
  * arm goal: the velocity command is zero and each wrist has a target pose
    fixed in the WORLD, so it stays put while the pelvis sways, crouches or is
    pushed.

The arms never see the world target. They see a pelvis-frame command, set
exactly at the event and afterwards moved by the pelvis motion the env's
odometry estimator reports (apply_pelvis_motion), the way the robot would do
it without motion capture. Rewards, reach detection and the curriculum all
use the true world target.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_unique,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "ArmTargetsCommand",
    "ArmTargetsCommandCfg",
    "ModalVelocityCommand",
    "ModalVelocityCommandCfg",
    "ground_height",
]


def ground_height(scanner: RayCaster) -> torch.Tensor:
    """Mean height of the scan's ray hits, ignoring rays that missed the mesh (inf).

    base_height_l2 averages all rays, so one miss at the terrain edge makes it
    inf. Returns nan for an env where every ray missed.
    """
    hits = scanner.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits)
    total = torch.where(finite, hits, torch.zeros_like(hits)).sum(dim=1)
    count = finite.sum(dim=1)
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, float("nan")))


def _apply(pos_a, quat_a, pose_b):
    """T_a o pose_b for (N, 3) / (N, 4) frames and (N, A, 7) poses."""
    n, a = pose_b.shape[:2]
    pos, rot = combine_frame_transforms(
        pos_a.unsqueeze(1).expand(n, a, 3).reshape(-1, 3),
        quat_a.unsqueeze(1).expand(n, a, 4).reshape(-1, 4),
        pose_b[..., :3].reshape(-1, 3),
        pose_b[..., 3:].reshape(-1, 4),
    )
    return torch.cat([pos, rot], dim=-1).view(n, a, 7)


def _relative(pos_a, quat_a, pose_w):
    """T_a^-1 o pose_w for (N, 3) / (N, 4) frames and (N, A, 7) poses, qw >= 0."""
    n, a = pose_w.shape[:2]
    pos, rot = subtract_frame_transforms(
        pos_a.unsqueeze(1).expand(n, a, 3).reshape(-1, 3),
        quat_a.unsqueeze(1).expand(n, a, 4).reshape(-1, 4),
        pose_w[..., :3].reshape(-1, 3),
        pose_w[..., 3:].reshape(-1, 4),
    )
    return torch.cat([pos, quat_unique(rot)], dim=-1).view(n, a, 7)


class ModalVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that is zero while the env has an arm goal."""

    cfg: ModalVelocityCommandCfg

    def _update_command(self):
        super()._update_command()
        arm_term: ArmTargetsCommand = self._env.command_manager.get_term(self.cfg.arm_command_name)
        self.vel_command_b[arm_term.arm_mode] = 0.0


@configclass
class ModalVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ModalVelocityCommand
    arm_command_name: str = MISSING


class ArmTargetsCommand(CommandTerm):
    """Per env, either a navigation goal or an arm goal; for an arm goal, one wrist pose per arm.

    **Events.** Reset, a reached arm goal (both wrists inside the tolerances
    for reach_hold_s), or the goal's timer running out. At each event the env
    draws its next goal: an arm goal with probability arm_goal_prob, else
    navigation for nav_time_range seconds (the velocity command is resampled).

    **Arm goal targets.** Drawn from a per-arm table, then fixed in the world.
    At construction, for each arm: set its joints to uniform random angles
    inside their soft limits (everything else at default, root at its default
    pose), take one physics step, record the wrist pose in the pelvis frame,
    and drop the sample if any of the arm's links is in contact. Every target
    was reached once, collision-free, standing. At the event the table pose is
    placed in the *standing frame* (pelvis x, y and yaw; standing pelvis height
    above the scanned ground; no roll or pitch) and that world pose is kept.

    **Curriculum.** Each env has a level. Levels 0..spread_levels draw from the
    easiest fraction (level + 1) / (spread_levels + 1) of each table, sorted by
    the larger of the distance and the rotation from the default pose (each
    normalised by its maximum), so goals start in front of the robot and widen
    to the whole standing workspace. The next drop_levels levels use the whole
    table and lower both targets by a shared random drop up to
    max_height_drop * k / drop_levels, which the legs must crouch for.
    update_levels() moves the level at episode end.

    **What the policies see.** believed_b: the targets in the pelvis frame.
    Exact at the event (on the robot: the operator's pelvis-frame command);
    then each step apply_pelvis_motion() moves it by the pelvis motion the
    env reports, estimated or true per goal (use_estimate, drawn at the event
    with probability estimate_prob, which the env ramps up).

    The command (get_command) is believed_b flattened: (num_envs, num_arms * 7),
    per arm (x, y, z, qw, qx, qy, qz) in the pelvis frame, w-first.
    """

    cfg: ArmTargetsCommandCfg

    def __init__(self, cfg: ArmTargetsCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.scanner: RayCaster = env.scene.sensors[cfg.height_scanner_name]
        self.num_arms = len(cfg.body_names)
        self.body_ids = [self.robot.find_bodies(name)[0][0] for name in cfg.body_names]
        self.max_level = cfg.spread_levels + cfg.drop_levels
        shape = (self.num_envs, self.num_arms, 7)

        # -- goal state
        self.arm_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.anchor_w = torch.zeros(shape, device=self.device)  # arm goal: world-fixed targets
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.height_drop = torch.zeros(self.num_envs, device=self.device)
        self.use_estimate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.estimate_prob = 0.0  # set by the env
        self.level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._at_reset = False
        # -- metrics (CommandTerm.reset logs their mean over the reset envs, then zeroes them)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["command_drift"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goals_missed"] = torch.zeros(self.num_envs, device=self.device)

        self._tables: list[torch.Tensor] = []
        self._level_counts: list[torch.Tensor] = []
        default_poses = []
        for arm in range(self.num_arms):
            table, default_pose = self._build_table(env, arm)
            sorted_table, counts = self._sort_by_difficulty(table, default_pose)
            self._tables.append(sorted_table)
            self._level_counts.append(counts)
            default_poses.append(default_pose)
        # (num_arms, 7): the arms-forward default, also the navigation rest pose
        self.rest_pose_b = torch.stack(default_poses)

    def __str__(self) -> str:
        msg = "ArmTargetsCommand:\n"
        msg += f"\tBodies: {self.cfg.body_names}\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tArm goal probability: {self.cfg.arm_goal_prob}\n"
        msg += f"\tArm goal timeout range: {self.cfg.resampling_time_range}\n"
        msg += f"\tNavigation time range: {self.cfg.nav_time_range}\n"
        msg += f"\tLevels: {self.cfg.spread_levels + 1} spread + {self.cfg.drop_levels} drop\n"
        for arm, counts in enumerate(self._level_counts):
            msg += f"\tTargets per spread level ({self.cfg.body_names[arm]}): {counts.tolist()}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The pelvis-frame command the policies see. Shape is (num_envs, num_arms * 7), quaternion w-first."""
        return self.believed_b.view(self.num_envs, -1)

    """
    Frames and errors. Recomputed from the current state on every call.
    """

    def standing_frame_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(origin, yaw-only quaternion) of the standing frame in world."""
        origin = self.robot.data.root_pos_w.clone()
        ground = ground_height(self.scanner)
        # an env whose scan missed entirely: fall back to the pelvis height
        origin[:, 2] = torch.where(torch.isnan(ground), origin[:, 2], ground + self.cfg.standing_height)
        return origin, yaw_quat(self.robot.data.root_quat_w)

    def targets_w(self) -> torch.Tensor:
        """True targets in world, (num_envs, num_arms, 7): the anchor for an arm goal, the rest pose otherwise."""
        rest = self.rest_pose_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        rest_w = _apply(self.robot.data.root_pos_w, self.robot.data.root_quat_w, rest)
        return torch.where(self.arm_mode.view(-1, 1, 1), self.anchor_w, rest_w)

    def true_targets_in_root(self) -> torch.Tensor:
        """True targets in the pelvis frame, (num_envs, num_arms, 7), qw >= 0. Privileged: the critic's."""
        return _relative(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.targets_w())

    def errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(position error in m, rotation error in rad) per arm against the true targets. Each (num_envs, num_arms)."""
        targets_w = self.targets_w()
        body_pos = self.robot.data.body_pos_w[:, self.body_ids]
        body_quat = self.robot.data.body_quat_w[:, self.body_ids]
        pos_error = torch.norm(body_pos - targets_w[..., :3], dim=-1)
        rot_error = quat_error_magnitude(body_quat.reshape(-1, 4), targets_w[..., 3:].reshape(-1, 4))
        return pos_error, rot_error.view(self.num_envs, self.num_arms)

    def apply_pelvis_motion(self, delta_pos: torch.Tensor, delta_quat: torch.Tensor, env_mask: torch.Tensor):
        """Re-express the arm-goal commands after the pelvis moved by (delta_pos, delta_quat).

        The motion is given in the previous pelvis frame. A world-fixed target
        obeys T_prev o c_prev = T_new o c_new, so c_new = delta^-1 o c_prev.
        Navigation commands are fixed in the pelvis frame and don't move.
        """
        mask = env_mask & self.arm_mode
        moved = _relative(delta_pos, delta_quat, self.believed_b)
        self.believed_b[mask] = moved[mask]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._at_reset = True
        try:
            return super().reset(env_ids)
        finally:
            self._at_reset = False

    """
    Curriculum.
    """

    def update_levels(self, env_ids: Sequence[int], fell: torch.Tensor):
        """At episode end: up if most arm goals were reached without a fall, down on a fall or mostly misses.

        A fall only counts against the level if it happened during an arm goal.
        """
        fell = fell & self.arm_mode[env_ids]
        reached = self.metrics["goals_reached"][env_ids]
        missed = self.metrics["goals_missed"][env_ids]
        finished = reached + missed
        rate = reached / finished.clamp(min=1.0)
        up = ~fell & (reached >= self.cfg.promote_min_goals) & (rate >= self.cfg.promote_rate)
        down = fell | ((finished >= 1.0) & (rate < self.cfg.demote_rate))
        self.level[env_ids] = (self.level[env_ids] + up.long() - down.long()).clamp(0, self.max_level)

    """
    Table construction.
    """

    def _build_table(self, env: ManagerBasedEnv, arm: int) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.robot
        body_idx = self.body_ids[arm]
        joint_ids, _ = robot.find_joints(self.cfg.joint_names[arm], preserve_order=True)
        sensor: ContactSensor = env.scene.sensors[self.cfg.contact_sensor_name]
        check_ids, _ = sensor.find_bodies(self.cfg.collision_body_names[arm], preserve_order=True)

        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        low, high = limits[:, 0], limits[:, 1]

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        default_q = robot.data.default_joint_pos.clone()
        zero_qd = torch.zeros_like(robot.data.default_joint_vel)

        # the default pose: curriculum centre, rest pose, rel_default_envs
        # target, and a check that the contact test passes where nothing touches
        default_pose, default_ok = self._reach(env, body_idx, default_q, zero_qd, root_state, sensor, check_ids)
        if not default_ok.all():
            raise RuntimeError(
                f"{self.cfg.body_names[arm]}: the default pose reads as a collision in "
                f"{int((~default_ok).sum())} envs; the contact test is broken."
            )

        poses, total, tried = [], 0, 0
        for _ in range(self.cfg.max_build_batches):
            q = default_q.clone()
            q[:, joint_ids] = low + (high - low) * torch.rand(self.num_envs, len(joint_ids), device=self.device)
            pose_b, ok = self._reach(env, body_idx, q, zero_qd, root_state, sensor, check_ids)
            poses.append(pose_b[ok])
            total += int(ok.sum())
            tried += self.num_envs
            if total >= self.cfg.table_size:
                break

        # leave the robot where the env's reset expects it
        self._reach(env, body_idx, default_q, zero_qd, root_state, sensor, check_ids)

        table = torch.cat(poses)[: self.cfg.table_size]
        print(
            f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}): {len(table)} targets,"
            f" {100.0 * total / tried:.1f}% of {tried} random arm poses collision-free."
            f" Default pose (pelvis frame): {[round(v, 3) for v in default_pose[0].tolist()]}"
        )
        return table, default_pose[0].clone()

    def _reach(self, env, body_idx, q, qd, root_state, sensor, check_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Puts every env in joint state q, steps once, returns (body pose in pelvis frame, collision-free)."""
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
            robot.data.body_pos_w[:, body_idx],
            robot.data.body_quat_w[:, body_idx],
        )
        peak_force = sensor.data.net_forces_w[:, check_ids].norm(dim=-1).max(dim=1)[0]
        ok = peak_force < self.cfg.collision_force_threshold
        return torch.cat([pos_b, quat_unique(quat_b)], dim=-1), ok

    def _sort_by_difficulty(self, table: torch.Tensor, default_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Table sorted easiest first, and how many entries each spread level may draw from."""
        distance = torch.norm(table[:, :3] - default_pose[:3], dim=-1)
        angle = quat_error_magnitude(table[:, 3:], default_pose[3:].expand(len(table), 4))
        difficulty = torch.maximum(distance / distance.max(), angle / angle.max())
        order = torch.argsort(difficulty)
        fractions = torch.arange(1, self.cfg.spread_levels + 2, device=self.device) / (self.cfg.spread_levels + 1)
        counts = torch.searchsorted(difficulty[order].contiguous(), fractions, right=True)
        counts = counts.clamp(min=min(self.cfg.min_targets_per_level, len(table)))
        counts[-1] = len(table)
        return table[order], counts

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pos_error, rot_error = self.errors()
        self.metrics["position_error"] = pos_error.mean(dim=1)
        self.metrics["orientation_error"] = rot_error.mean(dim=1)
        believed_w = _apply(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.believed_b)
        drift = torch.norm(believed_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(dim=1)
        self.metrics["command_drift"] = drift * self.arm_mode

        within = (pos_error < self.cfg.reach_pos_tol).all(dim=1) & (rot_error < self.cfg.reach_rot_tol).all(dim=1)
        step_dt = self._env.step_dt
        self.hold_time = torch.where(within & self.arm_mode, self.hold_time + step_dt, torch.zeros_like(self.hold_time))
        self.just_reached = self.hold_time >= self.cfg.reach_hold_s
        timed_out = (self.time_left - step_dt <= 0.0) & ~self.just_reached & self.arm_mode
        self.metrics["goals_reached"] += self.just_reached.float()
        self.metrics["goals_missed"] += timed_out.float()
        # CommandTerm.compute counts time_left down next and resamples every env at <= 0
        self.time_left[self.just_reached] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        arm_goal = torch.rand(n, device=self.device) < self.cfg.arm_goal_prob
        self.arm_mode[env_ids] = arm_goal
        self.hold_time[env_ids] = 0.0
        self.height_drop[env_ids] = 0.0
        self.use_estimate[env_ids] = torch.rand(n, device=self.device) < self.estimate_prob

        # -- navigation: rest pose in the pelvis frame, a fresh velocity command
        nav_ids = env_ids[~arm_goal]
        if len(nav_ids) > 0:
            self.believed_b[nav_ids] = self.rest_pose_b
            self.time_left[nav_ids] = torch.empty(len(nav_ids), device=self.device).uniform_(*self.cfg.nav_time_range)
            self._env.command_manager.get_term(self.cfg.velocity_command_name)._resample(nav_ids)

        # -- arm goal: table targets, anchored in the world now
        arm_ids = env_ids[arm_goal]
        if len(arm_ids) == 0:
            return
        m = len(arm_ids)
        level = self.level[arm_ids]
        spread_level = level.clamp(max=self.cfg.spread_levels)
        at_default = torch.rand(m, device=self.device) < self.cfg.rel_default_envs
        targets_s = torch.empty(m, self.num_arms, 7, device=self.device)
        for arm in range(self.num_arms):
            count = self._level_counts[arm][spread_level]
            idx = (torch.rand(m, device=self.device) * count).long()
            pose = self._tables[arm][idx]
            pose[at_default] = self.rest_pose_b[arm]
            targets_s[:, arm] = pose
        drop_level = (level - self.cfg.spread_levels).clamp(min=0).float()
        max_drop = self.cfg.max_height_drop * drop_level / max(self.cfg.drop_levels, 1)
        self.height_drop[arm_ids] = torch.rand(m, device=self.device) * max_drop
        targets_s[..., 2] -= self.height_drop[arm_ids].unsqueeze(1)

        origin, quat = self.standing_frame_w()
        if self._at_reset:
            # the height scan still shows where the robot was before the reset
            # teleported it; the reset places it on its env origin's ground
            origin[arm_ids, 2] = self._env.scene.env_origins[arm_ids, 2] + self.cfg.standing_height
        self.anchor_w[arm_ids] = _apply(origin[arm_ids], quat[arm_ids], targets_s)
        self.believed_b[arm_ids] = _relative(
            self.robot.data.root_pos_w[arm_ids], self.robot.data.root_quat_w[arm_ids], self.anchor_w[arm_ids]
        )
        # stand still for it: zero now rather than at the velocity term's next update
        self._env.command_manager.get_term(self.cfg.velocity_command_name).vel_command_b[arm_ids] = 0.0

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.believed_pose_visualizer = VisualizationMarkers(self.cfg.believed_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.believed_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.believed_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot can be de-initialized while the callback is still subscribed
        if not self.robot.is_initialized:
            return
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        targets_w = self.targets_w().reshape(-1, 7)
        self.goal_pose_visualizer.visualize(targets_w[:, :3], targets_w[:, 3:])
        believed_w = _apply(root_pos, root_quat, self.believed_b).reshape(-1, 7)
        self.believed_pose_visualizer.visualize(believed_w[:, :3], believed_w[:, 3:])
        body_pose_w = self.robot.data.body_link_pose_w[:, self.body_ids].reshape(-1, 7)
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:])


def _frame_marker(prim_path: str, scale: float = 0.1) -> VisualizationMarkersCfg:
    marker = FRAME_MARKER_CFG.replace(prim_path=prim_path)
    # replace() is shallow: copy the markers dict before rescaling, or every
    # FRAME_MARKER_CFG user in the process gets the new scale
    marker.markers = dict(marker.markers)
    marker.markers["frame"] = marker.markers["frame"].replace(scale=(scale, scale, scale))
    return marker


@configclass
class ArmTargetsCommandCfg(CommandTermCfg):
    class_type: type = ArmTargetsCommand

    asset_name: str = MISSING
    velocity_command_name: str = MISSING
    """The ModalVelocityCommand this term zeroes during arm goals and resamples for navigation."""
    body_names: list[str] = MISSING
    """One body per arm whose pose is commanded."""
    joint_names: list[list[str]] = MISSING
    """Per arm, the joints randomized to build its table."""
    collision_body_names: list[list[str]] = MISSING
    """Per arm, bodies that must have no contact for a table sample to count."""
    contact_sensor_name: str = "contact_forces"
    collision_force_threshold: float = 1.0
    height_scanner_name: str = "height_scanner"
    standing_height: float = MISSING
    """Pelvis height above the ground when standing: the standing frame's origin."""

    table_size: int = 100_000
    max_build_batches: int = 4000
    rel_default_envs: float = 0.1
    """Fraction of arm goals that are the default (arms-forward) pose, lowered by the drop like any other."""

    # -- goals. resampling_time_range is an arm goal's timeout.
    arm_goal_prob: float = 0.5
    nav_time_range: tuple[float, float] = (4.0, 8.0)
    reach_pos_tol: float = 0.05
    reach_rot_tol: float = 0.35
    reach_hold_s: float = 0.2

    # -- curriculum
    spread_levels: int = 10
    """Levels 0..spread_levels widen the region; the last one is the whole standing table."""
    min_targets_per_level: int = 256
    drop_levels: int = 5
    max_height_drop: float = 0.25
    promote_rate: float = 0.8
    promote_min_goals: int = 3
    demote_rate: float = 0.4

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_goal_pose", 0.1)
    believed_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_believed_pose", 0.06)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_body_pose", 0.1)
