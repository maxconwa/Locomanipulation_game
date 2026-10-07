"""Commands for the two-agent task: navigation OR an arm goal, never both.

At every command event ArmTargetsCommand gives an env either navigation (the velocity command is live, the arms
hold their rest pose in the pelvis frame) or an arm goal (zero velocity, each wrist a target pose fixed in the
world). The policies see the arm command in the pelvis frame, exact at the event and then moved by the pelvis
motion the env's odometry reports (apply_pelvis_motion); rewards, reach detection and the curriculum use the true
world target.
"""

from __future__ import annotations

import os
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
    euler_xyz_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
    wrap_to_pi,
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


def _balance_cells(table: torch.Tensor, cell: float, size: int, min_rows: int, generator: torch.Generator) -> torch.Tensor:
    """table resampled so every occupied cell (a cube of side cell, by position) holds size // cells rows.

    Random joint angles crowd the wrist onto the outstretched shell of the workspace. Each cell is drawn without
    replacement where it has enough rows, with random repeats where it has fewer; cells with fewer than min_rows
    rows (slivers at the workspace edge) are dropped.
    """
    keys = torch.floor(table[:, :3] / cell).long()
    _, cell_of, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    kept = counts >= min_rows
    quota = max(size // max(int(kept.sum()), 1), 1)
    # rows grouped by cell, in random order within it; rank = position inside its cell
    noise = torch.rand(len(table), generator=generator, device=table.device, dtype=torch.float64)
    order = torch.argsort(cell_of.double() + noise)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(table), device=table.device) - starts[cell_of[order]]
    rows = order[(rank < quota) & kept[cell_of[order]]]
    short = torch.nonzero(kept & (counts < quota)).flatten()
    if len(short) > 0:
        cells = torch.repeat_interleave(short, quota - counts[short])
        pick = (torch.rand(len(cells), generator=generator, device=table.device) * counts[cells]).long()
        rows = torch.cat([rows, order[starts[cells] + pick]])
    return table[rows]


def _add_weighted_mean(extras: dict, key: str, total: torch.Tensor, count: torch.Tensor):
    """extras[key] = sum(total) / sum(count) over the logged envs; left out when none had a step in that mode."""
    steps = count.sum().item()
    if steps > 0:
        extras[key] = total.sum().item() / steps


class ModalVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that is zero during arm goals and settles.

    A resampled moving command is replaced, with probability pure_turn_prob, by a turn in place (|yaw rate| in
    pure_turn_speed, either sign), and with probability pure_lateral_prob by a sideways walk (|vy| in
    pure_lateral_speed). Logs the tracking error over navigation steps and the yaw rate during arm goals.
    """

    cfg: ModalVelocityCommandCfg

    def __init__(self, cfg: ModalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        zeros = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._nav_steps, self._nav_err_xy, self._nav_err_yaw = zeros(), zeros(), zeros()
        self._arm_steps, self._arm_yaw_rate = zeros(), zeros()

    def _arm_mode(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).arm_mode

    def _settling(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).settling

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
        self.vel_command_b[self._arm_mode() | self._settling()] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        arm = self._arm_mode().float()
        nav = (1.0 - arm) * (~self._settling()).float()
        cmd_xy = self.vel_command_b[:, :2]
        vel_xy = self.robot.data.root_lin_vel_b[:, :2]
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        self._nav_steps += nav
        self._nav_err_xy += torch.norm(cmd_xy - vel_xy, dim=-1) * nav
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
    """Per env, either navigation or an arm goal (one wrist pose per arm), with a settle between modes.

    **Events.** Reset or the goal's timer running out. The next goal is an arm goal with probability arm_goal_prob,
    else navigation for nav_time_range seconds (the velocity command is resampled). A mode switch first settles
    for settle_to_arm_s / settle_to_nav_s: velocity zero, arms at the rest pose, arm_mode False, so the
    navigation-only legs terms pull the pelvis back up after a crouch.

    **Targets.** Half of the arm goals are standing goals from the standing table, half low goals from a squat
    table. The standing table: random collision-free arm poses with the robot standing, in the pelvis frame. A
    squat table per drop level j: random collision-free arm poses with the legs in a feet-flat squat j /
    drop_levels of the way down to the deepest one, below the standing table's low_target_quantile height, in the
    standing frame. Both of a low goal's targets come from one depth, drawn uniformly up to the env's drop level.
    Tables drop targets behind min_target_x and are balanced over balance_cell cubes. They are built once per run
    and saved to <log_dir>/<table_file>. At the event a target is placed in the standing frame (pelvis x, y and yaw,
    standing_height above the scanned ground) and that world pose is kept for the goal.

    **Curriculum.** Each table is sorted by difficulty (distance and rotation from the default pose); spread level
    k draws from the easiest first_level_fraction ** (1 - k / spread_levels) of it. Spread moves on standing
    goals' reach rate over the env's last level_window of them (up at >= promote_rate, down below demote_rate),
    drop on low goals' mean closest approach (up at <= drop_promote_error, down above drop_demote_error). A fall
    during an arm goal costs a level of the goal's axis.

    **Reach.** A goal lasts its timer. It is reached once, when both wrists are first held inside the tolerances
    for reach_hold_s (the bonus is paid then); the tracking terms keep paying for staying on target.

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
        self.arm_mode = flags()
        self.anchor_w = torch.zeros(shape, device=self.device)  # arm goal: world-fixed targets
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.shadow_b = torch.zeros(shape, device=self.device)  # moved by the estimate, always
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.shadow_b[..., 3] = 1.0
        self.height_drop = zeros()  # a low goal's squat depth (pelvis drop, m)
        self.lowest_target_height = zeros()  # lower wrist target above the ground, at the event
        self.needs_crouch = flags()  # a target below anything the standing table reaches
        self.use_estimate = flags()
        self.estimate_prob = 0.0  # set by the env
        self.hold_time = zeros()
        self.just_reached = flags()
        self.goal_reached = flags()  # this goal, once
        self.goal_low = flags()  # from a squat table
        self.settling = flags()
        self.pending_arm = flags()  # the mode drawn when the settle began, applied when it ends
        self._at_reset = False
        self._errors_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._errors_step = -1

        # -- curriculum: standing goals' reach outcomes move spread, low goals' closest approach moves drop
        self.spread_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.drop_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        self.outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.error_outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        self.low_outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # each arm goal's closest approach (mean of both wrists' position error, m)
        self.goal_best_error = torch.full((self.num_envs,), float("inf"), device=self.device)

        # -- metrics. CommandTerm.reset logs their mean over the reset envs, then zeroes them.
        for name in ("command_drift", "estimator_drift", "goals_reached", "goals_missed", "crouch_goals_reached",
                     "crouch_goals_missed", "settle_steps"):
            self.metrics[name] = zeros()
        # per-episode sums for the mode-split metrics, logged as weighted means by reset()
        self._goal_steps, self._goal_pos_err, self._goal_rot_err = zeros(), zeros(), zeros()
        self._rest_steps, self._rest_pos_err = zeros(), zeros()
        self._crouch_n, self._crouch_t, self._crouch_p = zeros(), zeros(), zeros()
        self._best_err_sum, self._best_err_n = zeros(), zeros()
        # shadow drift summed over arm goals that ended since the env last read it (the env's drift gate)
        self.ended_goal_drift_sum = torch.zeros((), device=self.device)
        self.ended_goal_count = torch.zeros((), device=self.device)

        data = self._load_or_build_tables(env)
        self._tables: list[torch.Tensor] = []
        self._level_counts: list[torch.Tensor] = []
        for table, default_pose in zip(data["tables"], data["default_poses"]):
            sorted_table, counts = self._sort_by_difficulty(table, default_pose)
            self._tables.append(sorted_table)
            self._level_counts.append(counts)
        # (num_arms,): the lowest wrist height each arm reaches standing, pelvis frame
        self._standing_min_z = torch.stack([table[:, 2].min() for table in self._tables])
        # (num_arms, 7): the arms-forward default, also the navigation rest pose
        self.rest_pose_b = torch.stack(data["default_poses"])
        self._setup_squat_tables(data["squat"])

    def __str__(self) -> str:
        msg = "ArmTargetsCommand:\n"
        msg += f"\tBodies: {self.cfg.body_names}\n"
        msg += f"\tArm goal probability: {self.cfg.arm_goal_prob}, low goals {self.cfg.low_goal_prob:.0%}\n"
        msg += f"\tLevels: {self.cfg.spread_levels + 1} spread x {self.cfg.drop_levels + 1} drop\n"
        msg += f"\tSquat depths (pelvis drop, m): {[round(v, 3) for v in self._squat_drops.tolist()]}\n"
        for arm, counts in enumerate(self._level_counts):
            msg += f"\tStanding targets per spread level ({self.cfg.body_names[arm]}): {counts.tolist()}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The pelvis-frame command the policies see, (num_envs, num_arms * 7), quaternion w-first."""
        return self.believed_b.view(self.num_envs, -1)

    """
    Frames and errors.
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
        """Re-express the arm-goal commands after the pelvis moved by (delta_pos, delta_quat), previous pelvis frame.

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
        _add_weighted_mean(extras, "rest_position_error", self._rest_pos_err[ids], self._rest_steps[ids])
        _add_weighted_mean(extras, "target_drop", self._crouch_t[ids], self._crouch_n[ids])
        _add_weighted_mean(extras, "pelvis_drop", self._crouch_p[ids], self._crouch_n[ids])
        _add_weighted_mean(extras, "goal_best_error", self._best_err_sum[ids], self._best_err_n[ids])
        for buffer in (self._goal_steps, self._goal_pos_err, self._goal_rot_err, self._rest_steps, self._rest_pos_err,
                       self._crouch_n, self._crouch_t, self._crouch_p, self._best_err_sum, self._best_err_n):
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

    @property
    def level(self) -> torch.Tensor:
        """spread_level + drop_level: one number for logs."""
        return self.spread_level + self.drop_level

    def update_levels(self, env_ids: Sequence[int], fell: torch.Tensor):
        """At episode end: a fall during a low goal costs a drop level, during a standing goal a spread level."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        down[env_ids] = fell & self.arm_mode[env_ids]
        low = down & self.goal_low
        self.drop_level -= (low & (self.drop_level > 0)).long()
        self.spread_level -= (down & ~low & (self.spread_level > 0)).long()
        self.low_outcome_count = torch.where(low, 0, self.low_outcome_count)
        self.outcome_count = torch.where(down & ~low, 0, self.outcome_count)

    def _record_outcomes(self, reached: torch.Tensor, missed: torch.Tensor):
        """Push this step's ended arm goals into each axis's window; move an axis on its full window.

        Each axis has its own window of the env's last level_window goals of its kind, restarted when it moves.
        """
        cfg, window = self.cfg, self.cfg.level_window
        ended = reached | missed
        standing_ended, low_ended = ended & ~self.goal_low, ended & self.goal_low
        slot = (self.outcome_count % window).unsqueeze(1)
        value = torch.where(standing_ended, reached.float(), self.outcomes.gather(1, slot).squeeze(1))
        self.outcomes.scatter_(1, slot, value.unsqueeze(1))
        self.outcome_count += standing_ended.long()
        slot = (self.low_outcome_count % window).unsqueeze(1)
        value = torch.where(low_ended, self.goal_best_error.clamp(max=10.0), self.error_outcomes.gather(1, slot).squeeze(1))
        self.error_outcomes.scatter_(1, slot, value.unsqueeze(1))
        self.low_outcome_count += low_ended.long()

        judged = standing_ended & (self.outcome_count >= window)
        rate = self.outcomes.mean(dim=1)
        up_spread = judged & (rate >= cfg.promote_rate) & (self.spread_level < cfg.spread_levels)
        down_spread = judged & (rate < cfg.demote_rate) & (self.spread_level > 0)
        judged = low_ended & (self.low_outcome_count >= window)
        error = self.error_outcomes.mean(dim=1)
        up_drop = judged & (error <= cfg.drop_promote_error) & (self.drop_level < cfg.drop_levels)
        down_drop = judged & (error > cfg.drop_demote_error) & (self.drop_level > 0)
        self.spread_level += up_spread.long() - down_spread.long()
        self.drop_level += up_drop.long() - down_drop.long()
        self.outcome_count = torch.where(up_spread | down_spread, 0, self.outcome_count)
        self.low_outcome_count = torch.where(up_drop | down_drop, 0, self.low_outcome_count)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Curriculum state, saved with the estimator so a resumed or played run keeps its levels."""
        return {
            "spread_level": self.spread_level.clone(),
            "drop_level": self.drop_level.clone(),
            "error_outcomes": self.error_outcomes.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        saved = {k: v.to(self.device) for k, v in state.items()}
        n = len(saved["spread_level"])
        pick = slice(None) if n == self.num_envs else torch.randint(0, n, (self.num_envs,), device=self.device)
        self.spread_level[:] = saved["spread_level"][pick].clamp(0, self.cfg.spread_levels)
        self.drop_level[:] = saved["drop_level"][pick].clamp(0, self.cfg.drop_levels)
        self.error_outcomes[:] = saved["error_outcomes"][pick]

    """
    Table construction.
    """

    def _table_settings(self) -> dict:
        """The cfg fields a saved table depends on; a file saved with others is rebuilt, not loaded."""
        cfg = self.cfg
        return {
            "body_names": list(cfg.body_names), "min_target_x": cfg.min_target_x, "balance_cell": cfg.balance_cell,
            "balance_min_rows": cfg.balance_min_rows, "table_size": cfg.table_size, "build_size": cfg.build_size,
            "squat_tables": True, "drop_levels": cfg.drop_levels, "low_target_quantile": cfg.low_target_quantile,
            "squat_table_size": cfg.squat_table_size, "squat_build_size": cfg.squat_build_size,
            "squat_limit_margin": cfg.squat_limit_margin, "squat_pelvis_pitch": cfg.squat_pelvis_pitch,
        }

    def _load_or_build_tables(self, env: ManagerBasedEnv) -> dict:
        """{"tables", "default_poses", "squat"}: loaded from <log_dir>/<table_file> if built with the same settings."""
        log_dir = getattr(env.cfg, "log_dir", None)
        path = os.path.join(log_dir, self.cfg.table_file) if log_dir else None
        settings = self._table_settings()
        if path and os.path.isfile(path):
            saved = torch.load(path, map_location=self.device)
            if saved.get("settings") == settings:
                print(f"[INFO] ArmTargetsCommand: loaded the target tables from {path}")
                return saved
            print(f"[INFO] ArmTargetsCommand: {path} was built with other settings; rebuilding.")
        generator = torch.Generator(device=self.device).manual_seed(0)
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        default_q = self.robot.data.default_joint_pos.clone()
        tables, default_poses = [], []
        for arm in range(self.num_arms):
            table, default_pose = self._build_table(env, arm, default_q, root_state, None, self.cfg.build_size)
            table = self._filter_and_balance(table, self.cfg.table_size, generator)
            print(
                f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}): {len(table)} standing targets after"
                f" filtering and balancing. Default pose (pelvis frame): {[round(v, 3) for v in default_pose.tolist()]}"
            )
            tables.append(table)
            default_poses.append(default_pose)
        data = {"settings": settings, "body_names": list(self.cfg.body_names), "tables": tables,
                "default_poses": default_poses}
        data["squat"] = self._build_squat_tables(env, tables, root_state, generator)
        # leave the robot where the env's reset expects it
        self._step_state(env, default_q, root_state)
        if path:
            os.makedirs(log_dir, exist_ok=True)
            torch.save(data, path)
        return data

    def _filter_and_balance(self, table: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
        table = table[table[:, 0] >= self.cfg.min_target_x]
        return _balance_cells(table, self.cfg.balance_cell, size, self.cfg.balance_min_rows, generator)

    def _build_table(
        self, env: ManagerBasedEnv, arm: int, base_q: torch.Tensor, root_state: torch.Tensor,
        frame: tuple[torch.Tensor, torch.Tensor] | None, size: int, keep=None, label: str = "standing",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """size collision-free wrist poses of one arm at random joint angles, the rest of the robot at base_q.

        Poses are in frame (the pelvis frame when None); keep(poses) -> bool mask filters them further. Returns
        (table, the pose at the arm's default angles). Standing, that pose must read collision-free: the check that
        the contact test works.
        """
        robot = self.robot
        body_idx = self.body_ids[arm]
        joint_ids, _ = robot.find_joints(self.cfg.joint_names[arm], preserve_order=True)
        sensor: ContactSensor = env.scene.sensors[self.cfg.contact_sensor_name]
        check_ids, _ = sensor.find_bodies(self.cfg.collision_body_names[arm], preserve_order=True)
        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        low, high = limits[:, 0], limits[:, 1]

        default_pose, default_ok = self._reach(env, body_idx, base_q, root_state, frame, sensor, check_ids)
        if frame is None and not default_ok.all():
            raise RuntimeError(
                f"{self.cfg.body_names[arm]}: the default pose reads as a collision in "
                f"{int((~default_ok).sum())} envs; the contact test is broken."
            )

        poses, total, free, tried = [], 0, 0, 0
        for _ in range(self.cfg.max_build_batches):
            q = base_q.clone()
            q[:, joint_ids] = low + (high - low) * torch.rand(self.num_envs, len(joint_ids), device=self.device)
            pose, ok = self._reach(env, body_idx, q, root_state, frame, sensor, check_ids)
            free += int(ok.sum())
            if keep is not None:
                ok &= keep(pose)
            poses.append(pose[ok])
            total += int(ok.sum())
            tried += self.num_envs
            if total >= size:
                break
        table = torch.cat(poses)[:size]
        print(
            f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}, {label}): {len(table)} targets kept,"
            f" {100.0 * free / tried:.1f}% of {tried} random arm poses collision-free"
            + ("." if keep is None else f", {100.0 * total / tried:.1f}% also kept.")
        )
        return table, default_pose[0].clone()

    def _step_state(self, env: ManagerBasedEnv, q: torch.Tensor, root_state: torch.Tensor):
        """Every env at rest in joint state q with its root at root_state, then one physics step."""
        robot = self.robot
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

    def _reach(self, env, body_idx, q, root_state, frame, sensor, check_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Every env in joint state q, one step: (body pose in frame, or the pelvis frame, and collision-free)."""
        self._step_state(env, q, root_state)
        robot = self.robot
        frame_pos, frame_quat = (robot.data.root_pos_w, robot.data.root_quat_w) if frame is None else frame
        pos_b, quat_b = subtract_frame_transforms(
            frame_pos, frame_quat, robot.data.body_pos_w[:, body_idx], robot.data.body_quat_w[:, body_idx]
        )
        peak_force = sensor.data.net_forces_w[:, check_ids].norm(dim=-1).max(dim=1)[0]
        ok = peak_force < self.cfg.collision_force_threshold
        return torch.cat([pos_b, quat_unique(quat_b)], dim=-1), ok

    def _squat_posture(
        self, env: ManagerBasedEnv, s: torch.Tensor, base_root: torch.Tensor, foot_home: torch.Tensor,
        foot_pitch_home: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A feet-flat squat per env at depth parameter s in [0, 1]: (joint state, root state, worst sole error in m).

        The hip, knee and ankle pitch joints move linearly from their defaults (s = 0) to the deepest squat (s = 1).
        The root is pitched so each sole stays flat and moved so each sole stays where it stands.
        """
        robot = self.robot
        default = robot.data.default_joint_pos
        q = default.clone()
        for ids, deep in zip(self._squat_joint_ids, self._squat_deep):
            q[:, ids] = default[:, ids] + s.unsqueeze(1) * (deep - default[:, ids])
        # flat soles: the pelvis pitch cancels what the leg's pitch joints add to their standing sum
        pitch = -sum(q[:, ids[0]] - default[:, ids[0]] for ids in self._squat_joint_ids)
        zeros = torch.zeros_like(pitch)
        root = base_root.clone()
        for _ in range(3):
            root[:, 3:7] = quat_from_euler_xyz(zeros, pitch, zeros)
            self._step_state(env, q, root)
            feet = robot.data.body_pos_w[:, self._foot_ids]
            _, foot_pitch, _ = euler_xyz_from_quat(robot.data.body_quat_w[:, self._foot_ids[0]])
            error = (foot_home - feet).norm(dim=-1).max(dim=1)[0]
            pitch = pitch - wrap_to_pi(foot_pitch - foot_pitch_home)
            root[:, :3] += (foot_home - feet).mean(dim=1)
        return q, root, error

    def _build_squat_tables(
        self, env: ManagerBasedEnv, standing: list[torch.Tensor], base_root: torch.Tensor, generator: torch.Generator
    ) -> dict:
        """Per drop level j (0..drop_levels), per arm: targets reachable collision-free in a squat j / drop_levels deep.

        Depths are evenly spaced in pelvis drop from standing to the deepest squat (knee and ankle pitch
        squat_limit_margin inside their soft limits, the pelvis leaning squat_pelvis_pitch). A low target sits below
        the standing table's low_target_quantile height and is recorded in the standing frame. Arm contact with
        anything, the legs, torso and ground included, rejects a sample.
        """
        robot, cfg = self.robot, self.cfg
        self._squat_joint_ids = [robot.find_joints(name)[0] for name in cfg.squat_joint_names]
        self._foot_ids = [robot.find_bodies(name)[0][0] for name in cfg.foot_body_names]
        hip, knee, ankle = self._squat_joint_ids
        soft = robot.data.soft_joint_pos_limits[0]
        default = robot.data.default_joint_pos[0]
        knee_deep = soft[knee, 1] - cfg.squat_limit_margin
        ankle_deep = soft[ankle, 0] + cfg.squat_limit_margin
        # flat soles keep hip + knee + ankle pitch at their standing sum minus the pelvis pitch
        hip_deep = default[hip] + default[knee] + default[ankle] - cfg.squat_pelvis_pitch - knee_deep - ankle_deep
        self._squat_deep = [hip_deep, knee_deep, ankle_deep]

        default_q = robot.data.default_joint_pos.clone()
        self._step_state(env, default_q, base_root)
        foot_home = robot.data.body_pos_w[:, self._foot_ids].clone()
        _, foot_pitch_home, _ = euler_xyz_from_quat(robot.data.body_quat_w[:, self._foot_ids[0]])
        frame = (base_root[:, :3], base_root[:, 3:7])  # the standing pelvis: the standing frame at an event

        # pelvis drop along the squat, one depth parameter per env, then the parameter of each even depth
        s = torch.linspace(0.0, 1.0, self.num_envs, device=self.device)
        _, root, error = self._squat_posture(env, s, base_root, foot_home, foot_pitch_home)
        drop = base_root[:, 2] - root[:, 2]
        if error.max() > 0.005 or not bool((drop[1:] >= drop[:-1] - 1e-4).all()):
            raise RuntimeError(f"Squat placement failed: sole error {error.max():.4f} m, drop not monotone in depth.")
        depths = cfg.drop_levels + 1
        wanted = drop[-1] * torch.arange(depths, device=self.device) / max(cfg.drop_levels, 1)
        hi = torch.searchsorted(drop.contiguous(), wanted).clamp(1, len(drop) - 1)
        lo = hi - 1
        w = ((wanted - drop[lo]) / (drop[hi] - drop[lo]).clamp(min=1e-9)).clamp(0.0, 1.0)
        s_depth = s[lo] + w * (s[hi] - s[lo])

        low_z = [torch.quantile(table[:, 2], cfg.low_target_quantile) for table in standing]
        tables = [[] for _ in range(self.num_arms)]
        rests = [[] for _ in range(self.num_arms)]
        drops, postures = [], []
        for j in range(depths):
            q, root, error = self._squat_posture(env, s_depth[j].expand(self.num_envs), base_root, foot_home, foot_pitch_home)
            if error.max() > 0.005:
                raise RuntimeError(f"Squat depth {j}: sole error {error.max():.4f} m.")
            drops.append(float(base_root[0, 2] - root[0, 2]))
            _, pelvis_pitch, _ = euler_xyz_from_quat(root[:1, 3:7])
            postures.append({
                "pelvis_drop": drops[-1],
                "pelvis_back": float(base_root[0, 0] - root[0, 0]),
                "pelvis_pitch": float(wrap_to_pi(pelvis_pitch)[0]),
                "hip_pitch": float(q[0, hip[0]]), "knee": float(q[0, knee[0]]), "ankle_pitch": float(q[0, ankle[0]]),
            })
            print(f"[INFO] ArmTargetsCommand: squat depth {j}: {postures[-1]}")
            for arm in range(self.num_arms):

                def keep(pose, z_max=low_z[arm]):
                    return (pose[:, 2] < z_max) & (pose[:, 0] >= cfg.min_target_x)

                table, rest = self._build_table(
                    env, arm, q, root, frame, cfg.squat_build_size, keep=keep, label=f"squat depth {j}"
                )
                table = self._filter_and_balance(table, cfg.squat_table_size, generator)
                if len(table) == 0:
                    raise RuntimeError(f"{cfg.body_names[arm]}: no low targets at squat depth {j}.")
                tables[arm].append(table)
                rests[arm].append(rest)
        return {"drops": drops, "tables": tables, "rest": rests, "postures": postures}

    def _setup_squat_tables(self, squat: dict):
        """Each depth's table sorted by difficulty from that depth's rest pose, padded into one tensor per arm."""
        self._squat_drops = torch.tensor(squat["drops"], device=self.device)
        self.squat_postures = squat["postures"]
        self._squat_tables: list[torch.Tensor] = []
        self._squat_counts: list[torch.Tensor] = []  # (depths, spread_levels + 1) per arm
        for arm in range(self.num_arms):
            ordered = [self._sort_by_difficulty(t, r) for t, r in zip(squat["tables"][arm], squat["rest"][arm])]
            padded = torch.zeros(len(ordered), max(len(t) for t, _ in ordered), 7, device=self.device)
            for j, (table, _) in enumerate(ordered):
                padded[j, : len(table)] = table
            self._squat_tables.append(padded)
            self._squat_counts.append(torch.stack([counts for _, counts in ordered]))

    def _sort_by_difficulty(self, table: torch.Tensor, default_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Table sorted easiest first, and how many entries each spread level may draw from (geometric)."""
        distance = torch.norm(table[:, :3] - default_pose[:3], dim=-1)
        angle = quat_error_magnitude(table[:, 3:], default_pose[3:].expand(len(table), 4))
        difficulty = torch.maximum(distance / distance.max(), angle / angle.max())
        order = torch.argsort(difficulty)
        levels = torch.arange(self.cfg.spread_levels + 1, device=self.device, dtype=torch.float32)
        fractions = self.cfg.first_level_fraction ** (1.0 - levels / max(self.cfg.spread_levels, 1))
        counts = (fractions * len(table)).long().clamp(min=1, max=len(table))
        counts[-1] = len(table)
        return table[order], counts

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pos_error, rot_error = self.errors()
        self.metrics["settle_steps"] += self.settling.float()
        arm = self.arm_mode.float()
        self._goal_steps += arm
        self._goal_pos_err += pos_error.mean(dim=1) * arm
        self._goal_rot_err += rot_error.mean(dim=1) * arm
        self._rest_steps += 1.0 - arm
        self._rest_pos_err += pos_error.mean(dim=1) * (1.0 - arm)
        ground = ground_height(self.scanner)
        counted = arm * torch.isfinite(ground).float()
        pelvis_drop = torch.nan_to_num(self.cfg.standing_height - (self.robot.data.root_pos_w[:, 2] - ground), nan=0.0)
        self._crouch_n += counted
        self._crouch_t += self.height_drop * counted
        self._crouch_p += pelvis_drop * counted

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
        self.metrics["crouch_goals_reached"] += (self.just_reached & self.needs_crouch).float()
        self.metrics["crouch_goals_missed"] += (timed_out & self.needs_crouch).float()
        best = torch.minimum(self.goal_best_error, pos_error.mean(dim=1))
        self.goal_best_error = torch.where(self.arm_mode, best, self.goal_best_error)
        self._best_err_sum += torch.where(ended, self.goal_best_error.clamp(max=10.0), 0.0)
        self._best_err_n += ended.float()
        # before CommandTerm.compute resamples the ended goals, so the next goal is drawn at the new level
        self._record_outcomes(ended & self.goal_reached, timed_out)

    def _resample_command(self, env_ids: Sequence[int]):
        self.invalidate_errors()
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        was_arm, was_settling = self.arm_mode[env_ids].clone(), self.settling[env_ids].clone()
        arm_goal = torch.rand(n, device=self.device) < self.cfg.arm_goal_prob
        # a settle ends in the mode drawn when it began; a mode switch first settles
        arm_goal = torch.where(was_settling, self.pending_arm[env_ids], arm_goal)
        settle_s = torch.where(arm_goal, self.cfg.settle_to_arm_s, self.cfg.settle_to_nav_s)
        settle = ~was_settling & (arm_goal != was_arm) & (not self._at_reset)
        self.settling[env_ids] = settle
        self.pending_arm[env_ids] = arm_goal
        arm_goal = arm_goal & ~settle

        self.arm_mode[env_ids] = arm_goal
        self.goal_best_error[env_ids] = float("inf")
        self.goal_reached[env_ids] = False
        self.goal_low[env_ids] = False
        self.hold_time[env_ids] = 0.0
        self.height_drop[env_ids] = 0.0
        self.lowest_target_height[env_ids] = 0.0
        self.needs_crouch[env_ids] = False
        self.use_estimate[env_ids] = torch.rand(n, device=self.device) < self.estimate_prob

        # -- settle: stop, arms at the rest pose
        settle_ids = env_ids[settle]
        if len(settle_ids) > 0:
            self.believed_b[settle_ids] = self.rest_pose_b
            self.shadow_b[settle_ids] = self.rest_pose_b
            self.time_left[settle_ids] = settle_s[settle]
            velocity.vel_command_b[settle_ids] = 0.0

        # -- navigation: rest pose in the pelvis frame, a fresh velocity command
        nav_ids = env_ids[~arm_goal & ~settle]
        if len(nav_ids) > 0:
            self.believed_b[nav_ids] = self.rest_pose_b
            self.shadow_b[nav_ids] = self.rest_pose_b
            self.time_left[nav_ids] = torch.empty(len(nav_ids), device=self.device).uniform_(*self.cfg.nav_time_range)
            velocity._resample(nav_ids)

        # -- arm goal: a standing or a low goal, anchored in the world now
        arm_ids = env_ids[arm_goal]
        if len(arm_ids) == 0:
            return
        m = len(arm_ids)
        spread_level = self.spread_level[arm_ids]
        at_default = torch.rand(m, device=self.device) < self.cfg.rel_default_envs
        low = torch.rand(m, device=self.device) < self.cfg.low_goal_prob
        depth = (torch.rand(m, device=self.device) * (self.drop_level[arm_ids] + 1)).long().clamp(max=self.cfg.drop_levels)
        at_default &= ~low
        self.height_drop[arm_ids] = torch.where(low, self._squat_drops[depth], 0.0)
        self.goal_low[arm_ids] = low
        targets_s = torch.empty(m, self.num_arms, 7, device=self.device)
        for arm in range(self.num_arms):
            count = self._level_counts[arm][spread_level]
            idx = (torch.rand(m, device=self.device) * count).long()
            pose = self._tables[arm][idx]
            if low.any():
                d, level = depth[low], spread_level[low]
                pick = (torch.rand(len(d), device=self.device) * self._squat_counts[arm][d, level]).long()
                pose[low] = self._squat_tables[arm][d, pick]
            pose[at_default] = self.rest_pose_b[arm]
            targets_s[:, arm] = pose
        # the standing frame's origin is standing_height above the ground
        self.lowest_target_height[arm_ids] = targets_s[..., 2].min(dim=1)[0] + self.cfg.standing_height
        self.needs_crouch[arm_ids] = (targets_s[..., 2] < self._standing_min_z).any(dim=1)

        origin, quat = self.standing_frame_w()
        if self._at_reset:
            # the height scan still shows where the robot was before the reset teleported it
            origin[arm_ids, 2] = self._env.scene.env_origins[arm_ids, 2] + self.cfg.standing_height
        self.anchor_w[arm_ids] = _apply(origin[arm_ids], quat[arm_ids], targets_s)
        self.believed_b[arm_ids] = _relative(
            self.robot.data.root_pos_w[arm_ids], self.robot.data.root_quat_w[arm_ids], self.anchor_w[arm_ids]
        )
        self.shadow_b[arm_ids] = self.believed_b[arm_ids]
        velocity.vel_command_b[arm_ids] = 0.0

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
    """The ModalVelocityCommand this term zeroes during arm goals and resamples for navigation."""
    body_names: list[str] = MISSING
    """One body per arm whose pose is commanded."""
    joint_names: list[list[str]] = MISSING
    """Per arm, the joints randomized to build its tables."""
    collision_body_names: list[list[str]] = MISSING
    """Per arm, bodies that must have no contact for a table sample to count."""
    foot_body_names: list[str] = MISSING
    """Bodies held where they stand while squatting (the soles stay flat on the ground)."""
    contact_sensor_name: str = "contact_forces"
    collision_force_threshold: float = 1.0
    height_scanner_name: str = "height_scanner"
    standing_height: float = MISSING
    """Pelvis height above the ground when standing: the standing frame's origin."""

    # -- goals
    arm_goal_prob: float = 0.5
    low_goal_prob: float = 0.5
    """The share of arm goals drawn from the squat tables."""
    rel_default_envs: float = 0.1
    """The share of standing goals that are the default (arms-forward) pose."""
    nav_time_range: tuple[float, float] = (4.0, 8.0)
    settle_to_arm_s: float = 0.75
    settle_to_nav_s: float = 1.5
    reach_pos_tol: float = 0.05
    reach_rot_tol: float = 0.35
    reach_hold_s: float = 1.0

    # -- curriculum
    spread_levels: int = 10
    first_level_fraction: float = 0.005
    """Share of each table (easiest first) that spread level 0 draws from; levels grow geometrically to 1."""
    drop_levels: int = 10
    level_window: int = 5
    promote_rate: float = 0.8
    demote_rate: float = 0.4
    drop_promote_error: float = 0.12
    drop_demote_error: float = 0.18

    # -- tables
    table_file: str = "arm_target_tables_v2.pt"
    """Saved under the env's log_dir after the first build; loaded instead of rebuilding when present."""
    table_size: int = 100_000
    build_size: int = 300_000
    """Collision-free standing targets collected per arm before filtering and balancing."""
    max_build_batches: int = 4000
    min_target_x: float = 0.1
    """m: targets less than this far forward of the pelvis are dropped."""
    balance_cell: float = 0.1
    balance_min_rows: int = 4
    low_target_quantile: float = 0.1
    """Low targets sit below this quantile of the standing table's heights."""
    squat_table_size: int = 50_000
    squat_build_size: int = 60_000
    squat_joint_names: tuple[str, str, str] = (".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint")
    """The pitch joints the squat bends: hip, knee, ankle, each matching both legs (left first)."""
    squat_limit_margin: float = 0.05
    """rad: the deepest squat stays this far inside the knee and ankle-pitch soft limits."""
    squat_pelvis_pitch: float = 0.15
    """rad: forward lean of the pelvis at the deepest squat."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_goal_pose", 0.1)
    believed_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_believed_pose", 0.06)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_body_pose", 0.1)
