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


def _balance_cells(table: torch.Tensor, cell: float, size: int, min_rows: int, generator: torch.Generator) -> torch.Tensor:
    """table resampled so every occupied cell (a cube of side cell, by position) holds the same number of rows.

    Random joint angles crowd the wrist onto the outstretched shell of the
    workspace (run G's table: 29% of its targets in a band beside the
    shoulder, 2.3% in front of the chest). Each cell keeps size // cells rows:
    drawn without replacement where it has that many, all of them plus random
    repeats where it has fewer. Cells with fewer than min_rows rows, slivers
    at the edge of the workspace, are dropped.
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
    """extras[key] = sum(total) / sum(count) over the envs being logged.

    Left out when none of them had a step in that mode: a 0 there would drag
    the logged average down.
    """
    steps = count.sum().item()
    if steps > 0:
        extras[key] = total.sum().item() / steps


class ModalVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that is zero while the env has an arm goal.

    Besides the parent's metrics (which average over every step, arm goals
    included) it logs, per episode:

      * nav_error_vel_xy / nav_error_vel_yaw: mean tracking error over
        navigation steps only (m/s, rad/s);
      * arm_goal_yaw_rate: mean |yaw rate| while an arm goal holds the
        command at zero: how much reaching turns the robot;
      * commanded_path / tracked_path (m): distance the navigation commands
        asked for, and the part of it the base covered along the commanded
        direction (capped at the commanded speed). The terrain curriculum
        (terrain_levels_tracking) reads these before the reset zeroes them.

    The command is also zero while ArmTargetsCommand has the env settling
    between modes, and settle steps count in neither mode's metrics.
    seg_commanded / seg_tracked are the same two paths summed over the current
    navigation segment only; ArmTargetsCommand reads them for its walking gate
    and zeroes them at each goal event.

    A resampled moving command is replaced, with probability pure_turn_prob, by
    a turn in place (zero linear velocity, |yaw rate| in pure_turn_speed, either
    sign), and with probability pure_lateral_prob by a sideways walk (zero
    forward and yaw, |vy| in pure_lateral_speed). Standing commands are kept.
    """

    cfg: ModalVelocityCommandCfg

    def __init__(self, cfg: ModalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        zeros = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self.metrics["commanded_path"] = zeros()
        self.metrics["tracked_path"] = zeros()
        self._nav_steps, self._nav_err_xy, self._nav_err_yaw = zeros(), zeros(), zeros()
        self._arm_steps, self._arm_yaw_rate = zeros(), zeros()
        self.seg_commanded, self.seg_tracked = zeros(), zeros()

    def _arm_mode(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).arm_mode

    def _settling(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).settling

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if self.cfg.pure_turn_prob <= 0.0 and self.cfg.pure_lateral_prob <= 0.0:
            return
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
        dt = self._env.step_dt
        arm = self._arm_mode().float()
        nav = (1.0 - arm) * (~self._settling()).float()
        cmd_xy = self.vel_command_b[:, :2]
        vel_xy = self.robot.data.root_lin_vel_b[:, :2]
        yaw_rate = self.robot.data.root_ang_vel_b[:, 2]
        speed = torch.norm(cmd_xy, dim=-1)
        along = (vel_xy * cmd_xy).sum(dim=-1) / speed.clamp(min=1e-6)
        commanded = speed * dt * nav
        tracked = torch.minimum(along.clamp(min=0.0), speed) * dt * nav
        self.metrics["commanded_path"] += commanded
        self.metrics["tracked_path"] += tracked
        self.seg_commanded += commanded
        self.seg_tracked += tracked
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
    """Per env, either a navigation goal or an arm goal; for an arm goal, one wrist pose per arm.

    **Events.** Reset, a reached arm goal (both wrists inside the tolerances
    for reach_hold_s), or the goal's timer running out. At each event the env
    draws its next goal: an arm goal with probability arm_goal_prob, else
    navigation for nav_time_range seconds (the velocity command is resampled).

    **Arm goal targets.** Drawn from a per-arm table, then fixed in the world.
    The table: for each arm, set its joints to uniform random angles inside
    their soft limits (everything else at default, root at its default pose),
    take one physics step, record the wrist pose in the pelvis frame, and drop
    the sample if any of the arm's links is in contact. Every target was
    reached once, collision-free, standing. It is built once per run and saved
    to <log_dir>/<table_file>; a later env with the same log_dir (play.py)
    loads it. At the event the table pose is placed in the *standing frame*
    (pelvis x, y and yaw; standing pelvis height above the scanned ground; no
    roll or pitch) and that world pose is kept.

    **Curriculum: two axes per env.** Each table is sorted by difficulty: the
    larger of the distance and the rotation from the default pose, each
    normalised by its maximum. The *spread* level k (0..spread_levels) draws
    from the easiest first_level_fraction ** (1 - k / spread_levels) of it, a
    geometric spacing, so goals start in front of the robot and widen to the
    whole standing workspace. The *drop* level j (0..drop_levels) lowers both
    targets by a shared random drop in [0, max_height_drop * j / drop_levels]:
    targets below where the arms reach standing. Nothing tells the legs to
    crouch; lowering the pelvis is just what makes those targets reachable.

    A lowered goal (drop > low_target_min_drop) draws each arm's target from
    the lower low_target_quantile of the level's region by height, beside or
    in front of the pelvis (x >= low_target_min_x), so the arms are already
    near the bottom of their reach and the drop is left to the legs (run 10:
    with targets drawn from the whole region the arms covered about two thirds
    of every drop, crouch slope 0.33; run 11, the lower half: the lowest
    target still 0.63 m above the ground, slope 0.5). With the bottom 10% and
    a 0.5 m drop the deepest targets sit about 0.4 m above the ground, which
    takes a full squat (feet flat, the pelvis at most 0.40 m down at the
    knee and ankle-pitch soft limits, with the hips behind the ankles) and a
    forward lean.

    needs_crouch marks a goal with a target below the lowest wrist height of
    the standing table (lowest_target_height, above the ground at the event):
    no arm pose reaches it standing. The crouch_goals_* metrics count those.

    **Squat tables (squat_tables).** Lowering standing targets has two
    faults: the drop draws most goals at high drop levels, so the high and
    wide workspace stops being trained (run G at drop 10: 96% of goals
    lowered, 3% of targets above 1.2 m), and a lowered target was never
    checked against a crouching body. With squat_tables the drop only
    extends the bottom of the range. An arm goal is a low goal with
    probability low_goal_prob, else a standing goal from the standing table,
    whose top never moves. Low goals come from tables built in feet-flat
    squats, one per drop level j, j / drop_levels of the way down in pelvis
    drop to the deepest squat (knee and ankle pitch squat_limit_margin
    inside their soft limits, the pelvis leaning squat_pelvis_pitch): random
    arm poses with the legs squatting, rejected on any arm contact (legs,
    torso, ground), kept below the standing table's low_target_quantile
    height, recorded in the standing frame. Both wrists of a low goal come
    from the same depth, so one pelvis height reaches both; an env draws a
    depth from 0 up to its drop level, uniformly (depth 0: the bottom of the
    standing reach). height_drop is then the pelvis drop the goal was built
    at. min_target_x drops targets behind the body and balance_cell spreads
    every table evenly over the space it covers, in both kinds of table.
    judge_axes_separately moves spread on standing goals only and drop on
    low goals only.

    **Goal timing.** resample_on_reach (default) replaces a reached goal at
    once. Off, a goal lasts its whole timer: it is reached once, when first
    held inside the tolerances for reach_hold_s (the bonus is paid then), and
    the tracking terms keep paying for holding still on target.

    Both move on the env's last level_window arm goals, across episodes: a
    promotion at >= promote_rate reached raises one axis, alternating and drop
    first, so low targets appear early while the spread is still easy; a
    demotion below demote_rate undoes the most recent promotion. The window
    restarts at each move. A fall during an arm goal demotes at the end of the
    episode (update_levels, from the arm_target_levels curriculum term).

    **Walking gate (walk_gate).** Off by default (arm goals with probability
    arm_goal_prob, as above). On: each env starts in walk_stage 0, navigation
    only with the arms at the rest pose, and is promoted to stage 1, where
    each event is an arm goal with probability alternate_arm_goal_prob, once
    its last gate_window judged navigation segments were tracked: a segment
    with at least gate_min_path m commanded is judged at its end, a success
    if it covered gate_track_ratio of that path along the commanded direction
    (ModalVelocityCommand.seg_*) without a fall. Promotion at >=
    gate_promote_rate of the window; demotion back to stage 0 below
    gate_demote_rate, or at once on a fall while walking or settling. The
    reach levels are kept through demotions.

    **Settle.** When an event switches mode (navigation -> arm goal or back)
    and settle_to_arm_s / settle_to_nav_s is > 0, the env first gets a settle
    segment of that length: velocity zero, arms at the rest pose, arm_mode
    False (so the navigation-only legs terms, base_height and stand_still,
    pull the pelvis back up after a crouch). The drawn mode starts when it
    ends; an arm goal's targets are anchored only then.

    **What the policies see.** believed_b: the targets in the pelvis frame.
    Exact at the event (on the robot: the operator's pelvis-frame command);
    then each step apply_pelvis_motion() moves it by the pelvis motion the
    env reports, estimated or true per goal (use_estimate, drawn at the event
    with probability estimate_prob, which the env ramps up and gates). shadow_b
    is the same command moved by the estimate on every arm goal, whatever
    use_estimate says: its drift (metric estimator_drift) measures the
    estimator even while the policies are still given the true motion.

    The command (get_command) is believed_b flattened: (num_envs, num_arms * 7),
    per arm (x, y, z, qw, qx, qy, qz) in the pelvis frame, w-first.

    **Metrics**, per episode, split by mode: goal_position_error /
    goal_orientation_error (mean over arm-goal steps), rest_position_error
    (mean over navigation steps), command_drift / estimator_drift, and the
    goals reached and missed, of them those that needed a crouch
    (crouch_goals_*). Crouch: target_drop and pelvis_drop (mean over arm-goal
    steps, m below standing), target_height (the lower target above the
    ground) and crouch_slope, the regression slope of pelvis drop on target
    drop over the logged episodes (what scripts/skrl/eval_crouch.py measures,
    live during training).
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

        # -- goal state
        self.arm_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.anchor_w = torch.zeros(shape, device=self.device)  # arm goal: world-fixed targets
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.shadow_b = torch.zeros(shape, device=self.device)  # moved by the estimate, always
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.shadow_b[..., 3] = 1.0
        self.height_drop = zeros()
        self.lowest_target_height = zeros()  # lower wrist target above the ground, at the event
        self.needs_crouch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.use_estimate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.estimate_prob = 0.0  # set by the env
        self.hold_time = zeros()
        self.just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # this goal, once
        self.goal_low = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # from a squat table
        self._at_reset = False
        self._errors_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._errors_step = -1

        # -- curriculum: two levels, and the env's recent arm-goal outcomes (1 reached, 0 missed)
        self.spread_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.drop_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._promote_drop_next = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_promoted_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        self.outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # each arm goal's closest approach (mean of both wrists' position error, m) and the window of them
        self.goal_best_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.error_outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        # judge_axes_separately: outcomes/outcome_count hold standing goals, error_outcomes/low_outcome_count low ones
        self.low_outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._best_err_sum, self._best_err_n = zeros(), zeros()

        # -- walking gate: stage 0 walks only, stage 1 alternates; recent judged navigation segments
        self.walk_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.nav_outcomes = torch.zeros(self.num_envs, cfg.gate_window, device=self.device)
        self.nav_outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # -- settle between modes: the mode drawn when the settle began, applied when it ends
        self.settling = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_arm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- metrics. CommandTerm.reset logs their mean over the reset envs, then zeroes them.
        self.metrics["command_drift"] = zeros()
        self.metrics["estimator_drift"] = zeros()
        self.metrics["goals_reached"] = zeros()
        self.metrics["goals_missed"] = zeros()
        self.metrics["crouch_goals_reached"] = zeros()
        self.metrics["crouch_goals_missed"] = zeros()
        self.metrics["nav_segments_judged"] = zeros()
        self.metrics["nav_segments_tracked"] = zeros()
        self.metrics["settle_steps"] = zeros()
        # per-episode sums for the mode-split metrics, logged as weighted means by reset()
        self._goal_steps, self._goal_pos_err, self._goal_rot_err = zeros(), zeros(), zeros()
        self._rest_steps, self._rest_pos_err = zeros(), zeros()
        # shadow drift summed over arm goals that ended since the env last read it
        self.ended_goal_drift_sum = torch.zeros((), device=self.device)
        self.ended_goal_count = torch.zeros((), device=self.device)

        if cfg.judge_axes_separately and (not cfg.squat_tables or cfg.drop_promote_error is None):
            raise ValueError("judge_axes_separately needs squat_tables and drop_promote_error.")
        if cfg.squat_tables and not cfg.foot_body_names:
            raise ValueError("squat_tables needs foot_body_names.")
        data = self._load_or_build_tables(env)
        raw_tables, default_poses = data["tables"], data["default_poses"]
        self._tables: list[torch.Tensor] = []
        self._level_counts: list[torch.Tensor] = []
        for table, default_pose in zip(raw_tables, default_poses):
            sorted_table, counts = self._sort_by_difficulty(table, default_pose)
            self._tables.append(sorted_table)
            self._level_counts.append(counts)
        # per arm, per spread level: indices of that level's region in its lower height quantile, not
        # behind the pelvis (a squat moves the hips back); the quantile alone if that leaves none
        self._low_index: list[torch.Tensor] = []
        self._low_counts: list[torch.Tensor] = []
        for table, counts in zip(self._tables, self._level_counts):
            rows = []
            for count in counts.tolist():
                z = table[:count, 2]
                low = z <= torch.quantile(z, self.cfg.low_target_quantile)
                in_front = low & (table[:count, 0] >= self.cfg.low_target_min_x)
                rows.append(torch.nonzero(in_front if in_front.any() else low).flatten())
            padded = torch.zeros(len(rows), max(len(r) for r in rows), dtype=torch.long, device=self.device)
            for level, row in enumerate(rows):
                padded[level, : len(row)] = row
            self._low_index.append(padded)
            self._low_counts.append(torch.tensor([len(r) for r in rows], device=self.device))
        # (num_arms,): the lowest wrist height each arm reaches standing, pelvis frame
        self._standing_min_z = torch.stack([table[:, 2].min() for table in self._tables])
        # crouch: per-episode sums over arm-goal steps of target drop (t), pelvis drop (p) and
        # lowest target height (h)
        self._crouch_n, self._crouch_t, self._crouch_p, self._crouch_tt, self._crouch_tp, self._crouch_h = (
            zeros(), zeros(), zeros(), zeros(), zeros(), zeros()
        )
        # (num_arms, 7): the arms-forward default, also the navigation rest pose
        self.rest_pose_b = torch.stack(default_poses)
        if cfg.squat_tables:
            self._setup_squat_tables(data["squat"])

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
        if self.cfg.squat_tables:
            msg += f"\tLow goals: {self.cfg.low_goal_prob:.0%}, squat depths (pelvis drop, m): "
            msg += f"{[round(v, 3) for v in self._squat_drops.tolist()]}\n"
            for arm, counts in enumerate(self._squat_counts):
                msg += f"\tLow targets per depth, all spread ({self.cfg.body_names[arm]}): {counts[:, -1].tolist()}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The pelvis-frame command the policies see. Shape is (num_envs, num_arms * 7), quaternion w-first."""
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
        """(position error in m, rotation error in rad) per arm against the true targets. Each (num_envs, num_arms).

        Cached per physics step: the arms' and the legs' tracking terms all ask
        for it. invalidate_errors() drops the cache when targets or robot state
        are written outside a physics step (goal events, env resets).
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
        self,
        estimated: tuple[torch.Tensor, torch.Tensor],
        true: tuple[torch.Tensor, torch.Tensor],
        env_mask: torch.Tensor,
    ):
        """Re-express the arm-goal commands after the pelvis moved; each motion is (delta_pos, delta_quat).

        The motion is given in the previous pelvis frame. A world-fixed target
        obeys T_prev o c_prev = T_new o c_new, so c_new = delta^-1 o c_prev.
        believed_b moves by the estimate where use_estimate, else by the true
        motion; shadow_b always by the estimate. Navigation commands are fixed
        in the pelvis frame and don't move.
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
        _add_weighted_mean(extras, "target_height", self._crouch_h[ids], self._crouch_n[ids])
        _add_weighted_mean(extras, "goal_best_error", self._best_err_sum[ids], self._best_err_n[ids])
        n = self._crouch_n[ids].sum()
        if n.item() > 1:
            t, p = self._crouch_t[ids].sum(), self._crouch_p[ids].sum()
            var = self._crouch_tt[ids].sum() - t * t / n
            if var.item() > 1e-6 * n.item():
                extras["crouch_slope"] = ((self._crouch_tp[ids].sum() - t * p / n) / var).item()
        for buffer in (self._goal_steps, self._goal_pos_err, self._goal_rot_err, self._rest_steps, self._rest_pos_err,
                       self._crouch_n, self._crouch_t, self._crouch_p, self._crouch_tt, self._crouch_tp,
                       self._crouch_h, self._best_err_sum, self._best_err_n):
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
        """At episode end: a fall during an arm goal is a demotion. Restarts the env's outcome window."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        down[env_ids] = fell & self.arm_mode[env_ids]
        if self.cfg.judge_axes_separately:
            # a fall during a low goal costs a drop level, during a standing goal a spread level
            low = down & self.goal_low
            self.drop_level -= (low & (self.drop_level > 0)).long()
            self.spread_level -= (down & ~low & (self.spread_level > 0)).long()
            self.low_outcome_count = torch.where(low, 0, self.low_outcome_count)
            down = down & ~low  # the spread window restarts only after a standing goal's fall, below
        elif self.cfg.drop_promote_error is None:
            self._move_levels(torch.zeros_like(down), down)
        else:
            # a fall while reaching: one drop level back if there is one, else one spread level
            lower_drop = down & (self.drop_level > 0)
            lower_spread = down & ~lower_drop & (self.spread_level > 0)
            self.drop_level -= lower_drop.long()
            self.spread_level -= lower_spread.long()
        self.outcome_count = torch.where(down, 0, self.outcome_count)
        if self.cfg.walk_gate:
            # the episode's last navigation segment: judged now (the reset's resample skips it); a fall
            # while walking or settling fails it whatever its path, and demotes an alternating env
            walking = ~self.arm_mode[env_ids]
            self._judge_nav_segments(env_ids[walking], fell[walking])

    def _record_outcomes(self, reached: torch.Tensor, missed: torch.Tensor):
        """Push this step's ended arm goals into each env's window; move levels on full windows."""
        ended = reached | missed
        if self.cfg.judge_axes_separately:
            self._record_split_outcomes(ended & ~self.goal_low, reached, ended & self.goal_low)
            return
        slot = self.outcome_count % self.cfg.level_window
        current = self.outcomes.gather(1, slot.unsqueeze(1)).squeeze(1)
        value = torch.where(ended, reached.float(), current)
        self.outcomes.scatter_(1, slot.unsqueeze(1), value.unsqueeze(1))
        current_err = self.error_outcomes.gather(1, slot.unsqueeze(1)).squeeze(1)
        err_value = torch.where(ended, self.goal_best_error.clamp(max=10.0), current_err)
        self.error_outcomes.scatter_(1, slot.unsqueeze(1), err_value.unsqueeze(1))
        self.outcome_count += ended.long()

        judged = ended & (self.outcome_count >= self.cfg.level_window)
        rate = self.outcomes.mean(dim=1)
        if self.cfg.drop_promote_error is None:
            up = judged & (rate >= self.cfg.promote_rate)
            down = judged & (rate < self.cfg.demote_rate)
            self._move_levels(up, down)
            self.outcome_count = torch.where(up | down, 0, self.outcome_count)
            return
        # independent axes: spread on the reach rate, drop on how close the wrists got
        error = self.error_outcomes.mean(dim=1)
        demote_error = self.cfg.drop_demote_error if self.cfg.drop_demote_error is not None else 1.5 * self.cfg.drop_promote_error
        up_drop = judged & (error <= self.cfg.drop_promote_error) & (self.drop_level < self.cfg.drop_levels)
        down_drop = judged & (error > demote_error) & (self.drop_level > 0)
        up_spread = judged & (rate >= self.cfg.promote_rate) & (self.spread_level < self.cfg.spread_levels)
        down_spread = judged & (rate < self.cfg.demote_rate) & (self.spread_level > 0)
        self.drop_level += up_drop.long() - down_drop.long()
        self.spread_level += up_spread.long() - down_spread.long()
        moved = up_drop | down_drop | up_spread | down_spread
        self.outcome_count = torch.where(moved, 0, self.outcome_count)

    def _record_split_outcomes(self, standing_ended: torch.Tensor, reached: torch.Tensor, low_ended: torch.Tensor):
        """judge_axes_separately: standing goals' reach outcomes move spread, low goals' closest approach moves drop.

        Each axis has its own window of the env's last level_window goals of
        its kind, restarted when that axis moves.
        """
        cfg, window = self.cfg, self.cfg.level_window
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
        demote_error = cfg.drop_demote_error if cfg.drop_demote_error is not None else 1.5 * cfg.drop_promote_error
        up_drop = judged & (error <= cfg.drop_promote_error) & (self.drop_level < cfg.drop_levels)
        down_drop = judged & (error > demote_error) & (self.drop_level > 0)
        self.spread_level += up_spread.long() - down_spread.long()
        self.drop_level += up_drop.long() - down_drop.long()
        self.outcome_count = torch.where(up_spread | down_spread, 0, self.outcome_count)
        self.low_outcome_count = torch.where(up_drop | down_drop, 0, self.low_outcome_count)

    def _move_levels(self, up: torch.Tensor, down: torch.Tensor):
        """Promotions alternate drop / spread (drop first); a demotion undoes the most recent promotion."""
        can_drop = self.drop_level < self.cfg.drop_levels
        can_spread = self.spread_level < self.cfg.spread_levels
        raise_drop = up & can_drop & (self._promote_drop_next | ~can_spread)
        raise_spread = up & can_spread & ~raise_drop
        self.drop_level += raise_drop.long()
        self.spread_level += raise_spread.long()
        self._promote_drop_next = torch.where(raise_drop, False, torch.where(raise_spread, True, self._promote_drop_next))
        self._last_promoted_drop = torch.where(raise_drop, True, torch.where(raise_spread, False, self._last_promoted_drop))

        lower_drop = down & (self.drop_level > 0) & (self._last_promoted_drop | (self.spread_level == 0))
        lower_spread = down & (self.spread_level > 0) & ~lower_drop
        self.drop_level -= lower_drop.long()
        self.spread_level -= lower_spread.long()
        # the next promotion re-raises what was just lowered; the one before it becomes the last promoted
        self._promote_drop_next = torch.where(lower_drop, True, torch.where(lower_spread, False, self._promote_drop_next))
        self._last_promoted_drop = torch.where(
            lower_drop, False, torch.where(lower_spread, True, self._last_promoted_drop)
        )

    def _judge_nav_segments(self, env_ids: torch.Tensor, fell: torch.Tensor | None = None):
        """Score the navigation segments that just ended in env_ids, move walk_stage, zero the segment paths."""
        if len(env_ids) == 0:
            return
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        fell = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device) if fell is None else fell
        commanded, tracked = velocity.seg_commanded[env_ids], velocity.seg_tracked[env_ids]
        # a settle segment has no commanded path: judged only when it ends in a fall
        judged = (commanded >= self.cfg.gate_min_path) | fell
        success = ~fell & (tracked >= self.cfg.gate_track_ratio * commanded)
        ids, ok = env_ids[judged], success[judged].float()
        self.metrics["nav_segments_judged"][ids] += 1.0
        self.metrics["nav_segments_tracked"][ids] += ok
        slot = self.nav_outcome_count[ids] % self.cfg.gate_window
        self.nav_outcomes[ids, slot] = ok
        self.nav_outcome_count[ids] += 1
        full = self.nav_outcome_count[ids] >= self.cfg.gate_window
        rate = self.nav_outcomes[ids].mean(dim=1)
        stage = self.walk_stage[ids]
        up = full & (stage == 0) & (rate >= self.cfg.gate_promote_rate)
        down = (stage == 1) & ((full & (rate < self.cfg.gate_demote_rate)) | fell[judged])
        self.walk_stage[ids[up]] = 1
        self.walk_stage[ids[down]] = 0
        self.nav_outcome_count[ids[up | down]] = 0  # the window restarts at each move
        velocity.seg_commanded[env_ids] = 0.0
        velocity.seg_tracked[env_ids] = 0.0

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Env-side curriculum state, saved with the estimator so a resumed or played run keeps its levels."""
        return {
            "spread_level": self.spread_level.clone(),
            "drop_level": self.drop_level.clone(),
            "promote_drop_next": self._promote_drop_next.clone(),
            "last_promoted_drop": self._last_promoted_drop.clone(),
            "error_outcomes": self.error_outcomes.clone(),
            "walk_stage": self.walk_stage.clone(),
            "nav_outcomes": self.nav_outcomes.clone(),
            "nav_outcome_count": self.nav_outcome_count.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        if "level" in state:
            # single-level runs (before the two axes): levels above spread_levels were drops
            level = state["level"].to(self.device)
            saved = {
                "spread_level": level.clamp(max=self.cfg.spread_levels),
                "drop_level": (level - self.cfg.spread_levels).clamp(min=0),
            }
        else:
            saved = {k: v.to(self.device) for k, v in state.items()}
        n = len(saved["spread_level"])
        pick = slice(None) if n == self.num_envs else torch.randint(0, n, (self.num_envs,), device=self.device)
        self.spread_level[:] = saved["spread_level"][pick].clamp(0, self.cfg.spread_levels)
        self.drop_level[:] = saved["drop_level"][pick].clamp(0, self.cfg.drop_levels)
        if "promote_drop_next" in saved:
            self._promote_drop_next[:] = saved["promote_drop_next"][pick]
            self._last_promoted_drop[:] = saved["last_promoted_drop"][pick]
        if "error_outcomes" in saved and saved["error_outcomes"].shape[1] == self.cfg.level_window:
            self.error_outcomes[:] = saved["error_outcomes"][pick]
        if "walk_stage" in saved and saved["nav_outcomes"].shape[1] == self.cfg.gate_window:
            self.walk_stage[:] = saved["walk_stage"][pick]
            self.nav_outcomes[:] = saved["nav_outcomes"][pick]
            self.nav_outcome_count[:] = saved["nav_outcome_count"][pick]

    """
    Table construction.
    """

    def _table_settings(self) -> dict:
        """The cfg fields a saved table depends on; a file saved with others is rebuilt, not loaded."""
        cfg = self.cfg
        settings = {"body_names": list(cfg.body_names)}
        if cfg.min_target_x is not None or cfg.balance_cell is not None or cfg.squat_tables:
            settings.update(
                min_target_x=cfg.min_target_x, balance_cell=cfg.balance_cell, balance_min_rows=cfg.balance_min_rows,
                table_size=cfg.table_size, build_size=cfg.build_size, squat_tables=cfg.squat_tables,
            )
        if cfg.squat_tables:
            settings.update(
                drop_levels=cfg.drop_levels, low_target_quantile=cfg.low_target_quantile,
                squat_table_size=cfg.squat_table_size, squat_build_size=cfg.squat_build_size,
                squat_limit_margin=cfg.squat_limit_margin, squat_pelvis_pitch=cfg.squat_pelvis_pitch,
            )
        return settings

    def _load_or_build_tables(self, env: ManagerBasedEnv) -> dict:
        """{"tables", "default_poses"} per arm, plus "squat" with squat_tables.

        Loaded from <log_dir>/<table_file> when it was saved with the same
        settings, else built and saved there.
        """
        log_dir = getattr(env.cfg, "log_dir", None)
        path = os.path.join(log_dir, self.cfg.table_file) if log_dir else None
        settings = self._table_settings()
        if path and os.path.isfile(path):
            saved = torch.load(path, map_location=self.device)
            # files from before the settings were saved hold plain tables
            if saved.get("settings", {"body_names": saved.get("body_names")}) == settings:
                print(f"[INFO] ArmTargetsCommand: loaded the target tables from {path}")
                return saved
            print(f"[INFO] ArmTargetsCommand: {path} was built with other settings; rebuilding.")
        generator = torch.Generator(device=self.device).manual_seed(0)
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += env.scene.env_origins
        default_q = self.robot.data.default_joint_pos.clone()
        tables, default_poses = [], []
        for arm in range(self.num_arms):
            size = self.cfg.build_size or self.cfg.table_size
            table, default_pose = self._build_table(env, arm, default_q, root_state, None, size)
            table = self._filter_and_balance(table, self.cfg.table_size, generator)
            print(
                f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}): {len(table)} standing targets after"
                f" filtering and balancing. Default pose (pelvis frame): {[round(v, 3) for v in default_pose.tolist()]}"
            )
            tables.append(table)
            default_poses.append(default_pose)
        data = {"settings": settings, "body_names": list(self.cfg.body_names), "tables": tables,
                "default_poses": default_poses}
        if self.cfg.squat_tables:
            data["squat"] = self._build_squat_tables(env, tables, root_state, generator)
        # leave the robot where the env's reset expects it
        self._step_state(env, default_q, root_state)
        if path:
            os.makedirs(log_dir, exist_ok=True)
            torch.save(data, path)
        return data

    def _filter_and_balance(self, table: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
        """Drops targets behind min_target_x, then balances (balance_cell) or truncates to size rows."""
        if self.cfg.min_target_x is not None:
            table = table[table[:, 0] >= self.cfg.min_target_x]
        if self.cfg.balance_cell is None:
            return table[:size]
        return _balance_cells(table, self.cfg.balance_cell, size, self.cfg.balance_min_rows, generator)

    def _build_table(
        self, env: ManagerBasedEnv, arm: int, base_q: torch.Tensor, root_state: torch.Tensor,
        frame: tuple[torch.Tensor, torch.Tensor] | None, size: int, keep=None, label: str = "standing",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """size collision-free wrist poses of one arm at random joint angles, the rest of the robot at base_q.

        Poses are in frame (pelvis frame when None). keep(poses) -> bool mask
        further filters the samples. Returns (table, the pose at the arm's
        default angles). Standing, that pose must read collision-free: the
        check that the contact test works.
        """
        robot = self.robot
        body_idx = self.body_ids[arm]
        joint_ids, _ = robot.find_joints(self.cfg.joint_names[arm], preserve_order=True)
        sensor: ContactSensor = env.scene.sensors[self.cfg.contact_sensor_name]
        check_ids, _ = sensor.find_bodies(self.cfg.collision_body_names[arm], preserve_order=True)

        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        low, high = limits[:, 0], limits[:, 1]

        # the default pose: curriculum centre, rest pose, rel_default_envs
        # target, and a check that the contact test passes where nothing touches
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
        """Puts every env in joint state q, steps once, returns (body pose in frame or the pelvis frame, collision-free)."""
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

        The hip, knee and ankle pitch joints move linearly from their defaults
        (s = 0) to the deepest squat (s = 1, _squat_deep). The root is pitched so
        each sole stays flat and moved so each sole stays where it stands: the
        pelvis lowers, goes back and leans forward, as it must on flat ground.
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

        The depths are evenly spaced in pelvis drop, from standing (j = 0) to
        the deepest squat. A low target sits below the standing table's
        low_target_quantile height (standing frame z) and, with min_target_x,
        that far forward of the standing pelvis; it is recorded in the standing
        frame, the frame an arm goal is anchored in. Arm contact with anything,
        the legs, torso and ground included, rejects a sample.
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
                z_max, x_min = low_z[arm], cfg.min_target_x

                def keep(pose, z_max=z_max, x_min=x_min):
                    low = pose[:, 2] < z_max
                    return low if x_min is None else low & (pose[:, 0] >= x_min)

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
        self._crouch_tt += self.height_drop.square() * counted
        self._crouch_tp += self.height_drop * pelvis_drop * counted
        self._crouch_h += self.lowest_target_height * counted

        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        believed_w = _apply(root_pos, root_quat, self.believed_b)
        self.metrics["command_drift"] = torch.norm(believed_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(1) * arm
        shadow_w = _apply(root_pos, root_quat, self.shadow_b)
        shadow_drift = torch.norm(shadow_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(dim=1)
        self.metrics["estimator_drift"] = shadow_drift * arm

        within = (pos_error < self.cfg.reach_pos_tol).all(dim=1) & (rot_error < self.cfg.reach_rot_tol).all(dim=1)
        step_dt = self._env.step_dt
        self.hold_time = torch.where(within & self.arm_mode, self.hold_time + step_dt, torch.zeros_like(self.hold_time))
        held = self.hold_time >= self.cfg.reach_hold_s
        if self.cfg.resample_on_reach:
            self.just_reached = held
            timed_out = (self.time_left - step_dt <= 0.0) & ~self.just_reached & self.arm_mode
            ended = self.just_reached | timed_out
            reached_at_end = self.just_reached
        else:
            # the goal stays until its timer runs out: reached once, when first held; judged when it ends
            self.just_reached = held & ~self.goal_reached & self.arm_mode
            self.goal_reached |= self.just_reached
            ended = (self.time_left - step_dt <= 0.0) & self.arm_mode
            timed_out = ended & ~self.goal_reached
            reached_at_end = ended & self.goal_reached
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
        # before the resample below, so the next goal is drawn at the new level
        self._record_outcomes(reached_at_end, timed_out)
        # CommandTerm.compute counts time_left down next and resamples every env at <= 0
        if self.cfg.resample_on_reach:
            self.time_left[self.just_reached] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        self.invalidate_errors()
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        was_arm, was_settling = self.arm_mode[env_ids].clone(), self.settling[env_ids].clone()
        if self.cfg.walk_gate:
            if not self._at_reset:
                # navigation segments that ran out their timer; a reset's were judged at the episode end
                walked = ~was_arm & ~was_settling
                self._judge_nav_segments(env_ids[walked])
            prob = torch.where(self.walk_stage[env_ids] > 0, self.cfg.alternate_arm_goal_prob, 0.0)
        else:
            prob = torch.full((n,), self.cfg.arm_goal_prob, device=self.device)
        arm_goal = torch.rand(n, device=self.device) < prob
        # a settle ends in the mode drawn when it began; a mode switch first settles
        arm_goal = torch.where(was_settling, self.pending_arm[env_ids], arm_goal)
        settle_s = torch.where(arm_goal, self.cfg.settle_to_arm_s, self.cfg.settle_to_nav_s)
        settle = ~was_settling & (arm_goal != was_arm) & (settle_s > 0.0) & (not self._at_reset)
        self.settling[env_ids] = settle
        self.pending_arm[env_ids] = arm_goal
        arm_goal = arm_goal & ~settle
        velocity.seg_commanded[env_ids] = 0.0
        velocity.seg_tracked[env_ids] = 0.0

        self.arm_mode[env_ids] = arm_goal
        self.goal_best_error[env_ids] = float("inf")
        self.goal_reached[env_ids] = False
        self.goal_low[env_ids] = False
        self.hold_time[env_ids] = 0.0
        self.height_drop[env_ids] = 0.0
        self.lowest_target_height[env_ids] = 0.0
        self.needs_crouch[env_ids] = False
        self.use_estimate[env_ids] = torch.rand(n, device=self.device) < self.estimate_prob

        # -- settle: stop, arms at the rest pose, for the switch's settle time
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

        # -- arm goal: table targets, anchored in the world now
        arm_ids = env_ids[arm_goal]
        if len(arm_ids) == 0:
            return
        m = len(arm_ids)
        spread_level = self.spread_level[arm_ids]
        at_default = torch.rand(m, device=self.device) < self.cfg.rel_default_envs
        if self.cfg.squat_tables:
            # a low goal from the table of a squat depth up to the env's drop level, else a standing goal; the
            # squat tables are in the standing frame already, so nothing is shifted
            low = torch.rand(m, device=self.device) < self.cfg.low_goal_prob
            depth = (torch.rand(m, device=self.device) * (self.drop_level[arm_ids] + 1)).long()
            depth = depth.clamp(max=self.cfg.drop_levels)
            at_default &= ~low
            drop = torch.where(low, self._squat_drops[depth], 0.0)
        else:
            drop_level = self.drop_level[arm_ids].float()
            max_drop = self.cfg.max_height_drop * drop_level / max(self.cfg.drop_levels, 1)
            drop = torch.rand(m, device=self.device) * max_drop
            # a lowered goal starts from the low part of the region: the arms can't take the drop alone
            low = (drop > self.cfg.low_target_min_drop) if self.cfg.low_targets_when_lowered else torch.zeros_like(at_default)
        self.height_drop[arm_ids] = drop
        self.goal_low[arm_ids] = low
        targets_s = torch.empty(m, self.num_arms, 7, device=self.device)
        for arm in range(self.num_arms):
            count = self._level_counts[arm][spread_level]
            idx = (torch.rand(m, device=self.device) * count).long()
            if low.any() and not self.cfg.squat_tables:
                level = spread_level[low]
                pick = (torch.rand(len(level), device=self.device) * self._low_counts[arm][level]).long()
                idx[low] = self._low_index[arm][level, pick]
            pose = self._tables[arm][idx]
            if low.any() and self.cfg.squat_tables:
                d, level = depth[low], spread_level[low]
                pick = (torch.rand(len(d), device=self.device) * self._squat_counts[arm][d, level]).long()
                pose[low] = self._squat_tables[arm][d, pick]
            pose[at_default] = self.rest_pose_b[arm]
            targets_s[:, arm] = pose
        if not self.cfg.squat_tables:
            targets_s[..., 2] -= drop.unsqueeze(1)
        # the standing frame's origin is standing_height above the ground
        self.lowest_target_height[arm_ids] = targets_s[..., 2].min(dim=1)[0] + self.cfg.standing_height
        self.needs_crouch[arm_ids] = (targets_s[..., 2] < self._standing_min_z).any(dim=1)

        origin, quat = self.standing_frame_w()
        if self._at_reset:
            # the height scan still shows where the robot was before the reset
            # teleported it; the reset places it on its env origin's ground
            origin[arm_ids, 2] = self._env.scene.env_origins[arm_ids, 2] + self.cfg.standing_height
        self.anchor_w[arm_ids] = _apply(origin[arm_ids], quat[arm_ids], targets_s)
        self.believed_b[arm_ids] = _relative(
            self.robot.data.root_pos_w[arm_ids], self.robot.data.root_quat_w[arm_ids], self.anchor_w[arm_ids]
        )
        self.shadow_b[arm_ids] = self.believed_b[arm_ids]
        # stand still for it: zero now rather than at the velocity term's next update
        velocity.vel_command_b[arm_ids] = 0.0

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
    table_file: str = "arm_target_tables.pt"
    """Saved under the env's log_dir after the first build; loaded instead of rebuilding when present."""
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
    """Spread levels 0..spread_levels widen the region; the last one is the whole standing table."""
    first_level_fraction: float = 0.005
    """Share of each table (easiest first) that level 0 draws from; levels grow geometrically to 1."""
    drop_levels: int = 10
    """Drop levels 0..drop_levels lower the targets by up to max_height_drop * level / drop_levels.

    5 cm per level, as in runs 9-11 (5 levels, 0.25 m), so a resumed run's levels keep their depth."""
    max_height_drop: float = 0.5
    low_targets_when_lowered: bool = True
    """Draw a lowered goal's targets from the low part of the level's region (see the class docstring)."""
    low_target_quantile: float = 0.1
    low_target_min_x: float = -0.1
    """A lowered goal's targets are at least this far forward of the pelvis (x, standing pelvis frame)."""
    low_target_min_drop: float = 0.02
    level_window: int = 5
    """Arm goals per judgement: the level moves on the env's last level_window outcomes."""
    promote_rate: float = 0.8
    demote_rate: float = 0.4
    drop_promote_error: float | None = None
    """m. None: both axes move together on the reach rate (alternating, drop first). Set: the axes move
    independently, spread on the reach rate and drop when the mean closest approach over the window
    (both wrists' mean position error) is at most this, so lowered targets don't wait on 5 cm precision."""
    drop_demote_error: float | None = None
    """m: the drop axis moves back above this mean closest approach (default 1.5 x drop_promote_error)."""

    # -- walking gate and settle (see the class docstring); off by default
    walk_gate: bool = False
    """Per-env stages: walk only until navigation is tracked, then alternate. arm_goal_prob is then unused."""
    alternate_arm_goal_prob: float = 0.5
    """Arm-goal probability per event once an env alternates (walk_stage 1)."""
    gate_window: int = 5
    gate_promote_rate: float = 0.8
    gate_demote_rate: float = 0.4
    gate_min_path: float = 1.0
    """m: a navigation segment that commanded less is not judged."""
    gate_track_ratio: float = 0.8
    """A judged segment succeeds when it covered this share of its commanded path, without a fall."""
    settle_to_arm_s: float = 0.0
    """Zero-velocity settle before an arm goal that follows navigation (s); 0 disables."""
    settle_to_nav_s: float = 0.0
    """Zero-velocity settle before navigation that follows an arm goal (s): time to stand up; 0 disables."""

    # -- goal timing
    resample_on_reach: bool = True
    """True: a reached goal is replaced at once. False: it stays until its timer runs out, so holding still on
    target keeps earning; it counts as reached (and pays the bonus) once, when first held for reach_hold_s."""

    # -- the workspace (see "Squat tables" in the class docstring); all off by default
    min_target_x: float | None = None
    """m: table targets less than this far forward of the pelvis (standing frame x) are dropped."""
    balance_cell: float | None = None
    """m: resample each table so every occupied cell of this size holds the same number of targets."""
    balance_min_rows: int = 4
    """Cells with fewer targets than this before balancing (slivers at the workspace edge) are dropped."""
    build_size: int | None = None
    """Collision-free targets to collect per arm before filtering and balancing (None: table_size)."""
    squat_tables: bool = False
    """Low goals from tables built with the legs in feet-flat squats, one per drop level (see the docstring)."""
    low_goal_prob: float = 0.5
    """With squat_tables: the share of arm goals drawn from the low tables; the rest come from the standing one."""
    squat_table_size: int = 50_000
    """Targets per arm per squat depth, after filtering and balancing."""
    squat_build_size: int = 60_000
    """Low (below the standing low_target_quantile height) collision-free targets to collect per arm and depth."""
    squat_joint_names: tuple[str, str, str] = (".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint")
    """The pitch joints the squat bends: hip, knee, ankle, each matching both legs (left first)."""
    foot_body_names: list[str] | None = None
    """Bodies held where they stand while squatting (the soles stay flat on the ground)."""
    squat_limit_margin: float = 0.05
    """rad: the deepest squat stays this far inside the knee and ankle-pitch soft limits."""
    squat_pelvis_pitch: float = 0.15
    """rad: forward lean of the pelvis at the deepest squat (run G's crouch leaned 0.15 at 27 cm down)."""
    judge_axes_separately: bool = False
    """With squat_tables: spread moves on the reach rate of standing goals only, drop on the closest approach of
    low goals only (drop_promote_error / drop_demote_error), so neither axis climbs on the other's goals."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_goal_pose", 0.1)
    believed_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_believed_pose", 0.06)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_body_pose", 0.1)
