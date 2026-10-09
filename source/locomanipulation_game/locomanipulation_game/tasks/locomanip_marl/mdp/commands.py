"""Commands for the two-agent task: navigation OR an arm goal, never both.

At every command event ArmTargetsCommand gives an env either navigation (the velocity command is live, each wrist
holds its rest pose in the pelvis frame) or, once the env's warm start is over, an arm goal (zero velocity, a wrist
pose per arm from a table of standing-reachable poses, lowered by the depth curriculum and fixed in the world). The
policies see the arm command in the pelvis frame, exact at the event and then moved by the pelvis motion the env's
odometry reports (apply_pelvis_motion); rewards, reach detection and the curriculum use the true world target.
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
    """extras[key] = sum(total) / sum(count) over the logged envs; left out when none had a counted step."""
    steps = count.sum().item()
    if steps > 0:
        extras[key] = total.sum().item() / steps


class ModalVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that is zero during arm goals.

    A resampled moving command is replaced, with probability pure_turn_prob, by a turn in place (|yaw rate| in
    pure_turn_speed, either sign), and with probability pure_lateral_prob by a sideways walk (|vy| in
    pure_lateral_speed): uniform sampling almost never draws either. Logs the tracking error during navigation and
    the yaw rate during arm goals.
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
    """Per env, either navigation (each wrist holds its rest pose) or an arm goal (one wrist pose per arm in the world).

    **Events.** Reset or the timer running out. Once goals_enabled (the env sets it when its warm start ends), the
    next task is an arm goal with probability arm_goal_prob, lasting resampling_time_range; otherwise navigation for
    nav_time_range, with a fresh velocity command. During navigation each wrist holds its rest pose, the wrist pose
    at the default joint angles in the pelvis frame. During an arm goal the velocity command is zero
    (ModalVelocityCommand).

    **Goals.** Each arm's pose is drawn uniformly from its table: collision-free wrist poses at random arm joint
    angles with the robot standing, in the pelvis frame, those less than min_target_x ahead of the pelvis dropped,
    balanced over balance_cell cubes. The tables are built once per run and saved to <log_dir>/<table_file>. At the
    event both poses are placed in the standing frame (the pelvis's x, y and yaw, standing_height above the scanned
    ground), lowered by one z_offset ~ U[0, z_max], and kept fixed in the world for the goal.

    **Curriculum.** A depth level k_d per env, 0..levels, sets z_max = depth_per_level * k_d. Once the env's last
    level_window arm goals have ended, k_d rises when their mean closest approach (per goal, the least mean of both
    wrists' position errors) is within promote_error and falls when it is beyond demote_error; each move restarts
    the window. A fall during an arm goal drops k_d by one and restarts the window.

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
        self.goals_enabled = False  # set by the env
        self.arm_mode = flags()
        self.anchor_w = torch.zeros(shape, device=self.device)  # arm goal: world-fixed targets
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.shadow_b = torch.zeros(shape, device=self.device)  # moved by the estimate, always
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.shadow_b[..., 3] = 1.0
        self.target_drop = zeros()  # the goal's z_offset (m)
        self.needs_crouch = flags()  # a target below anything the table reaches standing
        self.use_estimate = flags()
        self.estimate_prob = 0.0  # set by the env
        self.hold_time = zeros()
        self.just_reached = flags()
        self.goal_reached = flags()  # this goal, once
        self._at_reset = False
        self._errors_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._errors_step = -1

        # -- curriculum: the closest approach of each env's last level_window arm goals
        self.level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.error_outcomes = torch.zeros(self.num_envs, cfg.level_window, device=self.device)
        self.outcome_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # each arm goal's closest approach (mean of both wrists' position error, m)
        self.goal_best_error = torch.full((self.num_envs,), float("inf"), device=self.device)

        # -- metrics. CommandTerm.reset logs their mean over the reset envs, then zeroes them.
        for name in ("command_drift", "estimator_drift", "goals_reached", "goals_missed", "crouch_goals_reached",
                     "crouch_goals_missed"):
            self.metrics[name] = zeros()
        # per-episode sums for the mode-split metrics, logged as weighted means by reset()
        self._goal_steps, self._goal_pos_err, self._goal_rot_err = zeros(), zeros(), zeros()
        self._rest_steps, self._rest_pos_err = zeros(), zeros()
        self._crouch_n, self._crouch_t, self._crouch_p = zeros(), zeros(), zeros()
        self._nav_n, self._nav_p = zeros(), zeros()
        self._best_err_sum, self._best_err_n = zeros(), zeros()
        # shadow drift summed over arm goals that ended since the env last read it (the env's drift gate)
        self.ended_goal_drift_sum = torch.zeros((), device=self.device)
        self.ended_goal_count = torch.zeros((), device=self.device)

        data = self._load_or_build_tables(env)
        self._tables: list[torch.Tensor] = data["tables"]
        # (num_arms,): the lowest wrist height each arm reaches standing, pelvis frame
        self._standing_min_z = torch.stack([table[:, 2].min() for table in self._tables])
        # (num_arms, 7): the wrist poses at the default joint angles, the navigation rest pose
        self.rest_pose_b = torch.stack(data["default_poses"])

    def __str__(self) -> str:
        cfg = self.cfg
        msg = "ArmTargetsCommand:\n"
        msg += f"\tBodies: {cfg.body_names}\n"
        msg += f"\tArm goal probability once enabled: {cfg.arm_goal_prob}, navigation {cfg.nav_time_range} s\n"
        msg += f"\tDepth levels: {cfg.levels + 1}, z_max = {cfg.depth_per_level} m per level\n"
        for arm, table in enumerate(self._tables):
            msg += f"\tTargets ({cfg.body_names[arm]}): {len(table)}\n"
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
        _add_weighted_mean(extras, "nav_pelvis_drop", self._nav_p[ids], self._nav_n[ids])
        _add_weighted_mean(extras, "goal_best_error", self._best_err_sum[ids], self._best_err_n[ids])
        for buffer in (self._goal_steps, self._goal_pos_err, self._goal_rot_err, self._rest_steps, self._rest_pos_err,
                       self._crouch_n, self._crouch_t, self._crouch_p, self._nav_n, self._nav_p, self._best_err_sum,
                       self._best_err_n):
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

    def z_max(self, level: torch.Tensor) -> torch.Tensor:
        """The deepest z_offset at each depth level (m)."""
        return self.cfg.depth_per_level * level.float()

    def update_levels(self, env_ids: Sequence[int], fell: torch.Tensor):
        """At episode end: a fall during an arm goal drops the depth level by one and restarts the env's window."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        down[env_ids] = fell & self.arm_mode[env_ids]
        self.level -= (down & (self.level > 0)).long()
        self.outcome_count = torch.where(down, 0, self.outcome_count)

    def _record_outcomes(self, ended: torch.Tensor):
        """Push this step's ended arm goals' closest approach into each env's window of its last level_window goals;
        move the level on a full window, and restart the window when it moves."""
        cfg, window = self.cfg, self.cfg.level_window
        slot = (self.outcome_count % window).unsqueeze(1)
        value = torch.where(ended, self.goal_best_error.clamp(max=10.0), self.error_outcomes.gather(1, slot).squeeze(1))
        self.error_outcomes.scatter_(1, slot, value.unsqueeze(1))
        self.outcome_count += ended.long()
        judged = ended & (self.outcome_count >= window)
        error = self.error_outcomes.mean(dim=1)
        up = judged & (error <= cfg.promote_error) & (self.level < cfg.levels)
        down = judged & (error > cfg.demote_error) & (self.level > 0)
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
    Table construction.
    """

    def _table_settings(self) -> dict:
        """The cfg fields a saved table depends on; a file saved with others is rebuilt, not loaded."""
        cfg = self.cfg
        return {
            "body_names": list(cfg.body_names), "min_target_x": cfg.min_target_x, "balance_cell": cfg.balance_cell,
            "balance_min_rows": cfg.balance_min_rows, "table_size": cfg.table_size, "build_size": cfg.build_size,
        }

    def _load_or_build_tables(self, env: ManagerBasedEnv) -> dict:
        """{"tables", "default_poses"}: loaded from <log_dir>/<table_file> if built with the same settings."""
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
            table, default_pose = self._build_table(env, arm, default_q, root_state, self.cfg.build_size)
            table = table[table[:, 0] >= self.cfg.min_target_x]
            table = _balance_cells(table, self.cfg.balance_cell, self.cfg.table_size, self.cfg.balance_min_rows, generator)
            print(
                f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}): {len(table)} targets after filtering and"
                f" balancing. Default pose (pelvis frame): {[round(v, 3) for v in default_pose.tolist()]}"
            )
            tables.append(table)
            default_poses.append(default_pose)
        data = {"settings": settings, "body_names": list(self.cfg.body_names), "tables": tables,
                "default_poses": default_poses}
        # leave the robot where the env's reset expects it
        self._step_state(env, default_q, root_state)
        if path:
            os.makedirs(log_dir, exist_ok=True)
            torch.save(data, path)
        return data

    def _build_table(
        self, env: ManagerBasedEnv, arm: int, base_q: torch.Tensor, root_state: torch.Tensor, size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """size collision-free wrist poses of one arm at random joint angles within its soft limits, the rest of the
        robot at base_q, in the pelvis frame. Returns (table, the pose at the arm's default angles), which must read
        collision-free: the check that the contact test works."""
        robot = self.robot
        body_idx = self.body_ids[arm]
        joint_ids, _ = robot.find_joints(self.cfg.joint_names[arm], preserve_order=True)
        sensor: ContactSensor = env.scene.sensors[self.cfg.contact_sensor_name]
        check_ids, _ = sensor.find_bodies(self.cfg.collision_body_names[arm], preserve_order=True)
        limits = robot.data.soft_joint_pos_limits[0, joint_ids]
        low, high = limits[:, 0], limits[:, 1]

        default_pose, default_ok = self._reach(env, body_idx, base_q, root_state, sensor, check_ids)
        if not default_ok.all():
            raise RuntimeError(
                f"{self.cfg.body_names[arm]}: the default pose reads as a collision in "
                f"{int((~default_ok).sum())} envs; the contact test is broken."
            )

        poses, total, tried = [], 0, 0
        for _ in range(self.cfg.max_build_batches):
            q = base_q.clone()
            q[:, joint_ids] = low + (high - low) * torch.rand(self.num_envs, len(joint_ids), device=self.device)
            pose, ok = self._reach(env, body_idx, q, root_state, sensor, check_ids)
            poses.append(pose[ok])
            total += int(ok.sum())
            tried += self.num_envs
            if total >= size:
                break
        table = torch.cat(poses)[:size]
        print(
            f"[INFO] ArmTargetsCommand({self.cfg.body_names[arm]}): {len(table)} targets kept,"
            f" {100.0 * total / tried:.1f}% of {tried} random arm poses collision-free."
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

    def _reach(self, env, body_idx, q, root_state, sensor, check_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Every env in joint state q, one step: (body pose in the pelvis frame, and collision-free)."""
        self._step_state(env, q, root_state)
        robot = self.robot
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, robot.data.body_pos_w[:, body_idx], robot.data.body_quat_w[:, body_idx]
        )
        peak_force = sensor.data.net_forces_w[:, check_ids].norm(dim=-1).max(dim=1)[0]
        ok = peak_force < self.cfg.collision_force_threshold
        return torch.cat([pos_b, quat_unique(quat_b)], dim=-1), ok

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pos_error, rot_error = self.errors()
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
        self._crouch_t += self.target_drop * counted
        self._crouch_p += pelvis_drop * counted
        # nothing rewards a pelvis height: a crouch kept while walking shows here
        walking = (1.0 - arm) * torch.isfinite(ground).float()
        self._nav_n += walking
        self._nav_p += pelvis_drop * walking

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
        self._record_outcomes(ended)

    def _resample_command(self, env_ids: Sequence[int]):
        self.invalidate_errors()
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        arm_goal = (torch.rand(n, device=self.device) < self.cfg.arm_goal_prob) & self.goals_enabled
        self.arm_mode[env_ids] = arm_goal
        self.goal_best_error[env_ids] = float("inf")
        self.goal_reached[env_ids] = False
        self.hold_time[env_ids] = 0.0
        self.target_drop[env_ids] = 0.0
        self.needs_crouch[env_ids] = False
        self.use_estimate[env_ids] = torch.rand(n, device=self.device) < self.estimate_prob

        # -- navigation: the rest pose in the pelvis frame, a fresh velocity command
        nav_ids = env_ids[~arm_goal]
        if len(nav_ids) > 0:
            self.believed_b[nav_ids] = self.rest_pose_b
            self.shadow_b[nav_ids] = self.rest_pose_b
            self.time_left[nav_ids] = torch.empty(len(nav_ids), device=self.device).uniform_(*self.cfg.nav_time_range)
            velocity._resample(nav_ids)

        # -- arm goal: table poses, lowered by one z_offset, anchored in the world now
        arm_ids = env_ids[arm_goal]
        m = len(arm_ids)
        if m == 0:
            return
        targets_s = torch.stack(
            [table[(torch.rand(m, device=self.device) * len(table)).long()] for table in self._tables], dim=1
        )
        z_offset = torch.rand(m, device=self.device) * self.z_max(self.level[arm_ids])
        targets_s[..., 2] -= z_offset.unsqueeze(1)
        self.target_drop[arm_ids] = z_offset
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
    """Per arm, the joints randomized to build its table."""
    collision_body_names: list[list[str]] = MISSING
    """Per arm, bodies that must have no contact for a table sample to count."""
    contact_sensor_name: str = "contact_forces"
    collision_force_threshold: float = 1.0
    height_scanner_name: str = "height_scanner"
    standing_height: float = MISSING
    """Pelvis height above the ground when standing: the standing frame's origin."""

    # -- tasks
    arm_goal_prob: float = 0.5
    """Once goals are enabled, the share of tasks that are arm goals."""
    nav_time_range: tuple[float, float] = (4.0, 8.0)
    """s: a navigation task's duration."""
    reach_pos_tol: float = 0.05
    reach_rot_tol: float = 0.35
    reach_hold_s: float = 1.0

    # -- depth curriculum
    levels: int = 10
    depth_per_level: float = 0.1
    """m: z_max = depth_per_level * k_d."""
    level_window: int = 5
    promote_error: float = 0.12
    demote_error: float = 0.18

    # -- tables
    table_file: str = "arm_target_tables_v3.pt"
    """Saved under the env's log_dir after the first build; loaded instead of rebuilding when present."""
    table_size: int = 100_000
    build_size: int = 300_000
    """Collision-free targets collected per arm before filtering and balancing."""
    max_build_batches: int = 4000
    min_target_x: float = 0.1
    """m: targets less than this far forward of the pelvis are dropped."""
    balance_cell: float = 0.1
    balance_min_rows: int = 4

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_goal_pose", 0.1)
    believed_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_believed_pose", 0.06)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = _frame_marker("/Visuals/Command/arm_body_pose", 0.1)
