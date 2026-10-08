"""Commands for the two-agent task: walking OR an arm goal, never both.

At every command event ArmTargetsCommand gives an env either walking (the velocity command is live, each wrist holds
its start pose in the pelvis frame) or, once the env enables goals (its walking gate), an arm goal (zero velocity, a
wrist pose per arm fixed in the world, drawn at random in the workspace in front of the robot). The policies see the
arm command in the pelvis frame, exact at the event and then moved by the pelvis motion the env's odometry reports
(apply_pelvis_motion); rewards, reach detection and the curriculum use the true world target.
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
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
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
    """Mean height of the scan's ray hits, ignoring rays that missed (inf); nan for an env where every ray missed."""
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


def _add_weighted_mean(extras: dict, key: str, total: torch.Tensor, count: torch.Tensor):
    """extras[key] = sum(total) / sum(count) over the logged envs; left out when none had a counted step."""
    steps = count.sum().item()
    if steps > 0:
        extras[key] = total.sum().item() / steps


class ModalVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that is zero during arm goals.

    A resampled moving command is replaced, with probability pure_turn_prob, by a turn in place (|yaw rate| in
    pure_turn_speed, either sign), and with probability pure_lateral_prob by a sideways walk (|vy| in
    pure_lateral_speed): uniform sampling almost never draws either. Logs the tracking error while walking and the
    yaw rate during arm goals.
    """

    cfg: ModalVelocityCommandCfg

    def __init__(self, cfg: ModalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        zeros = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._nav_steps, self._nav_err_xy, self._nav_err_yaw = zeros(), zeros(), zeros()
        self._arm_steps, self._arm_yaw_rate = zeros(), zeros()

    def _arm_mode(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).arm_mode

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device)
        ids = ids[~self.is_standing_env[ids]]
        u = torch.rand(len(ids), device=self.device)
        turn = ids[u < self.cfg.pure_turn_prob]
        lateral = ids[(u >= self.cfg.pure_turn_prob) & (u < self.cfg.pure_turn_prob + self.cfg.pure_lateral_prob)]
        for sub, axis, speed in ((turn, 2, self.cfg.pure_turn_speed), (lateral, 1, self.cfg.pure_lateral_speed)):
            if len(sub) == 0:
                continue
            sign = torch.where(torch.rand(len(sub), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[sub] = 0.0
            self.vel_command_b[sub, axis] = sign * torch.empty(len(sub), device=self.device).uniform_(*speed)

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self._arm_mode()] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        arm = self._arm_mode().float()
        nav = 1.0 - arm
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        self._nav_steps += nav
        self._nav_err_xy += torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) * nav
        self._nav_err_yaw += torch.abs(self.vel_command_b[:, 2] - yaw_rate) * nav
        self._arm_steps += arm
        self._arm_yaw_rate += torch.abs(yaw_rate) * arm

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        extras = {}
        _add_weighted_mean(extras, "nav_error_vel_xy", self._nav_err_xy[ids], self._nav_steps[ids])
        _add_weighted_mean(extras, "nav_error_vel_yaw", self._nav_err_yaw[ids], self._nav_steps[ids])
        _add_weighted_mean(extras, "arm_goal_yaw_rate", self._arm_yaw_rate[ids], self._arm_steps[ids])
        for buffer in (self._nav_steps, self._nav_err_xy, self._nav_err_yaw, self._arm_steps, self._arm_yaw_rate):
            buffer[ids] = 0.0
        extras.update(super().reset(env_ids))
        return extras


@configclass
class ModalVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ModalVelocityCommand
    arm_command_name: str = MISSING
    pure_turn_prob: float = 0.0
    pure_turn_speed: tuple[float, float] = (0.2, 0.5)       # rad/s
    pure_lateral_prob: float = 0.0
    pure_lateral_speed: tuple[float, float] = (0.15, 0.3)   # m/s


class ArmTargetsCommand(CommandTerm):
    """Per env, either walking (each wrist holds its start pose) or an arm goal (one wrist pose per arm in the world).

    **Events.** Reset or the timer running out. The next segment is an arm goal with probability arm_goal_prob once
    goals_enabled (the env sets it when its walking gate opens), lasting resampling_time_range; otherwise walking
    for nav_time_range, with a fresh velocity command. While walking each wrist holds its start pose, the wrist pose
    at the default joint angles in the pelvis frame. During an arm goal the velocity command is zero
    (ModalVelocityCommand).

    **Goals.** Drawn in the heading frame at the event (the pelvis's x, y and yaw, on the scanned ground) and kept
    fixed in the world. Per arm: x in x_range ahead of the pelvis, y within half_width of the start pose's y, z
    above the ground between min_height and max_height, all uniform; the orientation is the start pose's turned by
    a uniform angle up to orientation_cone about a uniform random axis.

    **Curriculum.** One level per env, 0..levels. min_height falls linearly from min_height_range[0] to
    min_height_range[1] and half_width grows from half_width_range[0] to half_width_range[1]. The level moves when
    the env's last level_window goals are judged: up if at least promote_rate were reached, down below
    demote_rate. A fall during a goal costs a level.

    **Reach.** A goal is reached once, when both wrists are first held inside the tolerances for reach_hold_s (the
    bonus is paid then); the tracking terms keep paying for staying on target.

    **What the policies see.** believed_b, the targets in the pelvis frame: exact at the event, then moved each
    step by the pelvis motion the env reports, estimated or true per goal (use_estimate, drawn with the env's
    estimate_prob). shadow_b is moved by the estimate on every goal: its drift measures the estimator.
    """

    cfg: ArmTargetsCommandCfg

    def __init__(self, cfg: ArmTargetsCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.scanner: RayCaster = env.scene.sensors[cfg.height_scanner_name]
        self.num_arms = len(cfg.body_names)
        self.body_ids = [self.robot.find_bodies(name)[0][0] for name in cfg.body_names]
        shape = (self.num_envs, self.num_arms, 7)
        zeros = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        flags = lambda: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # noqa: E731

        # -- goal state
        self.goals_enabled = False  # set by the env
        self.arm_mode = flags()
        self.anchor_w = torch.zeros(shape, device=self.device)  # the goal, fixed in the world
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.shadow_b = torch.zeros(shape, device=self.device)  # moved by the estimate, always
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.shadow_b[..., 3] = 1.0
        self.goal_height = zeros()  # the lower target's height above the ground, at the event
        self.use_estimate = flags()
        self.estimate_prob = 0.0  # set by the env
        self.hold_time = zeros()
        self.just_reached = flags()
        self.goal_reached = flags()  # this goal, once
        self._at_reset = False
        self._errors_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._errors_step = -1

        # -- curriculum: the reach outcomes of each env's last level_window goals
        self.level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        self.outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # each goal's closest approach (mean of both wrists' position error, m)
        self.goal_best_error = torch.full((self.num_envs,), float("inf"), device=self.device)

        # -- metrics. CommandTerm.reset logs their mean over the reset envs, then zeroes them.
        for name in ("command_drift", "estimator_drift", "goals_reached", "goals_missed"):
            self.metrics[name] = zeros()
        # per-episode sums, logged as weighted means by reset()
        self._goal_steps, self._goal_pos_err, self._goal_rot_err = zeros(), zeros(), zeros()
        self._hold_steps, self._hold_pos_err = zeros(), zeros()
        self._goal_n, self._goal_h, self._pelvis_h = zeros(), zeros(), zeros()
        self._best_err_sum, self._best_err_n = zeros(), zeros()
        # shadow drift summed over goals that ended since the env last read it (the env's drift gate)
        self.ended_goal_drift_sum = torch.zeros((), device=self.device)
        self.ended_goal_count = torch.zeros((), device=self.device)

        # (num_arms, 7): the wrists' start pose in the pelvis frame
        self.rest_pose_b = self._start_pose(env)
        print(f"[INFO] ArmTargetsCommand: start pose (pelvis frame) {[[round(v, 3) for v in p] for p in self.rest_pose_b.tolist()]}")

    def __str__(self) -> str:
        cfg = self.cfg
        msg = "ArmTargetsCommand:\n"
        msg += f"\tBodies: {cfg.body_names}\n"
        msg += f"\tArm goal probability once enabled: {cfg.arm_goal_prob}, walking {cfg.nav_time_range} s\n"
        msg += f"\tGoals: x {cfg.x_range} m, height {cfg.min_height_range} .. {cfg.max_height} m above the ground,"
        msg += f" y within {cfg.half_width_range} m of the start pose, orientation within {cfg.orientation_cone} rad\n"
        msg += f"\tLevels: {cfg.levels + 1}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The pelvis-frame command the policies see, (num_envs, num_arms * 7), quaternion w-first."""
        return self.believed_b.view(self.num_envs, -1)

    def _start_pose(self, env: ManagerBasedEnv) -> torch.Tensor:
        """The wrists' pose in the pelvis frame with every env at rest at the default joint angles: one physics step,
        then the robot is left in the state the env's reset expects."""
        robot = self.robot
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        q = robot.data.default_joint_pos.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        poses = []
        for body_id in self.body_ids:
            pos, quat = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id]
            )
            poses.append(torch.cat([pos[0], quat_unique(quat[0:1])[0]]))
        return torch.stack(poses)

    """
    Frames and errors.
    """

    def heading_frame_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(origin, yaw-only quaternion) of the heading frame in world: under the pelvis, on the scanned ground."""
        origin = self.robot.data.root_pos_w.clone()
        ground = ground_height(self.scanner)
        # an env whose scan missed entirely: the env origin's height
        origin[:, 2] = torch.where(torch.isnan(ground), self._env.scene.env_origins[:, 2], ground)
        return origin, yaw_quat(self.robot.data.root_quat_w)

    def targets_w(self) -> torch.Tensor:
        """True targets in world, (num_envs, num_arms, 7): the goal, or the start pose while walking."""
        rest = self.rest_pose_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        rest_w = _apply(self.robot.data.root_pos_w, self.robot.data.root_quat_w, rest)
        return torch.where(self.arm_mode.view(-1, 1, 1), self.anchor_w, rest_w)

    def true_targets_in_root(self) -> torch.Tensor:
        """True targets in the pelvis frame, (num_envs, num_arms, 7), qw >= 0. Privileged: the critic's."""
        return _relative(self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.targets_w())

    def errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(position error in m, rotation error in rad) per arm against the true targets, each (num_envs, num_arms).

        Cached per physics step; invalidate_errors() drops the cache when targets or the robot are written outside
        a physics step (goal events, resets).
        """
        step = self._env._sim_step_counter
        if self._errors_cache is None or self._errors_step != step:
            targets_w = self.targets_w()
            body_pos = self.robot.data.body_pos_w[:, self.body_ids]
            body_quat = self.robot.data.body_quat_w[:, self.body_ids]
            pos_error = torch.norm(body_pos - targets_w[..., :3], dim=-1)
            rot_error = quat_error_magnitude(body_quat.reshape(-1, 4), targets_w[..., 3:].reshape(-1, 4))
            self._errors_cache = (pos_error, rot_error.view(self.num_envs, self.num_arms))
            self._errors_step = step
        return self._errors_cache

    def invalidate_errors(self):
        self._errors_cache = None

    def apply_pelvis_motion(
        self, estimated: tuple[torch.Tensor, torch.Tensor], true: tuple[torch.Tensor, torch.Tensor], env_mask: torch.Tensor
    ):
        """Re-express the goals after the pelvis moved by (delta_pos, delta_quat), previous pelvis frame.

        A world-fixed target obeys T_prev o c_prev = T_new o c_new, so c_new = delta^-1 o c_prev. believed_b moves
        by the estimate where use_estimate, else by the true motion; shadow_b always by the estimate.
        """
        mask = env_mask & self.arm_mode
        by_estimate = _relative(estimated[0], estimated[1], self.believed_b)
        by_truth = _relative(true[0], true[1], self.believed_b)
        moved = torch.where(self.use_estimate.view(-1, 1, 1), by_estimate, by_truth)
        self.believed_b[mask] = moved[mask]
        shadow = _relative(estimated[0], estimated[1], self.shadow_b)
        self.shadow_b[mask] = shadow[mask]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        extras = {}
        _add_weighted_mean(extras, "goal_position_error", self._goal_pos_err[ids], self._goal_steps[ids])
        _add_weighted_mean(extras, "goal_orientation_error", self._goal_rot_err[ids], self._goal_steps[ids])
        _add_weighted_mean(extras, "hold_position_error", self._hold_pos_err[ids], self._hold_steps[ids])
        _add_weighted_mean(extras, "goal_height", self._goal_h[ids], self._goal_n[ids])
        _add_weighted_mean(extras, "pelvis_height", self._pelvis_h[ids], self._goal_n[ids])
        _add_weighted_mean(extras, "goal_best_error", self._best_err_sum[ids], self._best_err_n[ids])
        for buffer in (self._goal_steps, self._goal_pos_err, self._goal_rot_err, self._hold_steps, self._hold_pos_err,
                       self._goal_n, self._goal_h, self._pelvis_h, self._best_err_sum, self._best_err_n):
            buffer[ids] = 0.0
        self.invalidate_errors()
        self._at_reset = True
        try:
            extras.update(super().reset(env_ids))
        finally:
            self._at_reset = False
        return extras

    """
    Curriculum.
    """

    def min_height(self, level: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.min_height_range
        return lo + (hi - lo) * level.float() / max(self.cfg.levels, 1)

    def half_width(self, level: torch.Tensor) -> torch.Tensor:
        lo, hi = self.cfg.half_width_range
        return lo + (hi - lo) * level.float() / max(self.cfg.levels, 1)

    def update_levels(self, env_ids: Sequence[int], fell: torch.Tensor):
        """At episode end: a fall during a goal costs a level and restarts the env's window."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        down[env_ids] = fell & self.arm_mode[env_ids]
        self.level -= (down & (self.level > 0)).long()
        self.outcome_count = torch.where(down, 0, self.outcome_count)

    def _record_outcomes(self, reached: torch.Tensor, missed: torch.Tensor):
        """Push this step's ended goals into each env's window; move the level on a full window, then restart it."""
        cfg, window = self.cfg, self.cfg.level_window
        ended = reached | missed
        slot = (self.outcome_count % window).unsqueeze(1)
        value = torch.where(ended, reached.float(), self.outcomes.gather(1, slot).squeeze(1))
        self.outcomes.scatter_(1, slot, value.unsqueeze(1))
        self.outcome_count += ended.long()
        judged = ended & (self.outcome_count >= window)
        rate = self.outcomes.mean(dim=1)
        up = judged & (rate >= cfg.promote_rate) & (self.level < cfg.levels)
        down = judged & (rate < cfg.demote_rate) & (self.level > 0)
        self.level += up.long() - down.long()
        self.outcome_count = torch.where(up | down, 0, self.outcome_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Curriculum state, saved with the estimator so a resumed or played run keeps its levels."""
        return {"level": self.level.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        saved = state["level"].to(self.device)
        pick = slice(None) if len(saved) == self.num_envs else torch.randint(0, len(saved), (self.num_envs,), device=self.device)
        self.level[:] = saved[pick].clamp(0, self.cfg.levels)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pos_error, rot_error = self.errors()
        arm = self.arm_mode.float()
        self._goal_steps += arm
        self._goal_pos_err += pos_error.mean(dim=1) * arm
        self._goal_rot_err += rot_error.mean(dim=1) * arm
        self._hold_steps += 1.0 - arm
        self._hold_pos_err += pos_error.mean(dim=1) * (1.0 - arm)
        ground = ground_height(self.scanner)
        counted = arm * torch.isfinite(ground).float()
        pelvis_height = torch.nan_to_num(self.robot.data.root_pos_w[:, 2] - ground, nan=0.0)
        self._goal_n += counted
        self._goal_h += self.goal_height * counted
        self._pelvis_h += pelvis_height * counted

        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        believed_w = _apply(root_pos, root_quat, self.believed_b)
        self.metrics["command_drift"] = torch.norm(believed_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(1) * arm
        shadow_w = _apply(root_pos, root_quat, self.shadow_b)
        shadow_drift = torch.norm(shadow_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(dim=1)
        self.metrics["estimator_drift"] = shadow_drift * arm

        within = (pos_error < self.cfg.reach_pos_tol).all(dim=1) & (rot_error < self.cfg.reach_rot_tol).all(dim=1)
        step_dt = self._env.step_dt
        self.hold_time = torch.where(within & self.arm_mode, self.hold_time + step_dt, torch.zeros_like(self.hold_time))
        # reached once, when first held; judged when the goal's timer runs out
        self.just_reached = (self.hold_time >= self.cfg.reach_hold_s) & ~self.goal_reached & self.arm_mode
        self.goal_reached |= self.just_reached
        ended = (self.time_left - step_dt <= 0.0) & self.arm_mode
        timed_out = ended & ~self.goal_reached
        self.ended_goal_drift_sum += (shadow_drift * ended).sum()
        self.ended_goal_count += ended.sum()
        self.metrics["goals_reached"] += self.just_reached.float()
        self.metrics["goals_missed"] += timed_out.float()
        best = torch.minimum(self.goal_best_error, pos_error.mean(dim=1))
        self.goal_best_error = torch.where(self.arm_mode, best, self.goal_best_error)
        self._best_err_sum += torch.where(ended, self.goal_best_error.clamp(max=10.0), 0.0)
        self._best_err_n += ended.float()
        # before CommandTerm.compute resamples the ended goals, so the next goal is drawn at the new level
        self._record_outcomes(ended & self.goal_reached, timed_out)

    def _resample_command(self, env_ids: Sequence[int]):
        self.invalidate_errors()
        env_ids = torch.as_tensor(env_ids, device=self.device)
        arm_goal = self.goals_enabled & (torch.rand(len(env_ids), device=self.device) < self.cfg.arm_goal_prob)
        self.arm_mode[env_ids] = arm_goal
        self.goal_best_error[env_ids] = float("inf")
        self.goal_reached[env_ids] = False
        self.hold_time[env_ids] = 0.0
        self.goal_height[env_ids] = 0.0
        self.use_estimate[env_ids] = torch.rand(len(env_ids), device=self.device) < self.estimate_prob

        # -- walking: the start pose in the pelvis frame, a fresh velocity command
        nav_ids = env_ids[~arm_goal]
        if len(nav_ids) > 0:
            self.believed_b[nav_ids] = self.rest_pose_b
            self.shadow_b[nav_ids] = self.rest_pose_b
            self.time_left[nav_ids] = torch.empty(len(nav_ids), device=self.device).uniform_(*self.cfg.nav_time_range)
            self._env.command_manager.get_term(self.cfg.velocity_command_name)._resample(nav_ids)

        # -- arm goal: drawn in the heading frame, anchored in the world now
        env_ids = env_ids[arm_goal]
        n = len(env_ids)
        if n == 0:
            return
        cfg, level = self.cfg, self.level[env_ids]
        uniform = lambda lo, hi: lo + (hi - lo) * torch.rand(n, self.num_arms, device=self.device)  # noqa: E731
        x = uniform(*cfg.x_range)
        y = self.rest_pose_b[:, 1] + uniform(-1.0, 1.0) * self.half_width(level).unsqueeze(1)
        z = uniform(0.0, 1.0) * (cfg.max_height - self.min_height(level).unsqueeze(1)) + self.min_height(level).unsqueeze(1)
        axis = torch.nn.functional.normalize(torch.randn(n * self.num_arms, 3, device=self.device), dim=-1)
        angle = cfg.orientation_cone * torch.rand(n * self.num_arms, device=self.device)
        turn = quat_from_angle_axis(angle, axis)
        rest_quat = self.rest_pose_b[:, 3:].unsqueeze(0).expand(n, -1, -1).reshape(-1, 4)
        targets_h = torch.cat([torch.stack([x, y, z], dim=-1), quat_mul(rest_quat, turn).view(n, self.num_arms, 4)], dim=-1)
        self.goal_height[env_ids] = z.min(dim=1)[0]

        origin, quat = self.heading_frame_w()
        origin, quat = origin[env_ids], quat[env_ids]
        if self._at_reset:
            # the height scan still shows where the robot was before the reset teleported it
            origin[:, 2] = self._env.scene.env_origins[env_ids, 2]
        self.anchor_w[env_ids] = _apply(origin, quat, targets_h)
        self.believed_b[env_ids] = _relative(
            self.robot.data.root_pos_w[env_ids], self.robot.data.root_quat_w[env_ids], self.anchor_w[env_ids]
        )
        self.shadow_b[env_ids] = self.believed_b[env_ids]
        self._env.command_manager.get_term(self.cfg.velocity_command_name).vel_command_b[env_ids] = 0.0

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.believed_pose_visualizer = VisualizationMarkers(self.cfg.believed_pose_visualizer_cfg)
            self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
        if hasattr(self, "goal_pose_visualizer"):
            for marker in (self.goal_pose_visualizer, self.believed_pose_visualizer, self.current_pose_visualizer):
                marker.set_visibility(debug_vis)

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
    # replace() is shallow: copy the markers dict before rescaling, or every FRAME_MARKER_CFG user gets the new scale
    marker.markers = dict(marker.markers)
    marker.markers["frame"] = marker.markers["frame"].replace(scale=(scale, scale, scale))
    return marker


@configclass
class ArmTargetsCommandCfg(CommandTermCfg):
    """resampling_time_range is an arm goal's duration."""

    class_type: type = ArmTargetsCommand

    asset_name: str = MISSING
    velocity_command_name: str = MISSING
    """The ModalVelocityCommand this term zeroes during arm goals and resamples for walking."""
    body_names: list[str] = MISSING
    """One body per arm whose pose is commanded."""
    height_scanner_name: str = "height_scanner"
    arm_goal_prob: float = 0.5
    """Once goals are enabled, the share of command events that start an arm goal."""
    nav_time_range: tuple[float, float] = (4.0, 8.0)
    """s: a walking segment's duration."""

    # -- goals, in the heading frame at the event
    x_range: tuple[float, float] = (0.25, 0.5)
    """m ahead of the pelvis."""
    max_height: float = 1.4
    """m above the ground."""
    orientation_cone: float = 0.5
    """rad: the largest turn of a goal's orientation away from the start pose's."""
    reach_pos_tol: float = 0.05
    reach_rot_tol: float = 0.35
    reach_hold_s: float = 1.0

    # -- curriculum
    levels: int = 10
    min_height_range: tuple[float, float] = (0.9, 0.2)
    """m above the ground: the lowest goal height at level 0 and at the top level."""
    half_width_range: tuple[float, float] = (0.05, 0.3)
    """m: how far a goal's y may be from the start pose's, at level 0 and at the top level."""
    level_window: int = 5
    promote_rate: float = 0.8
    demote_rate: float = 0.4

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_goal_pose", 0.1)
    believed_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_believed_pose", 0.06)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_body_pose", 0.1)
