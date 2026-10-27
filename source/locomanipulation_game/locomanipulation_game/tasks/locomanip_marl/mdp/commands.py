"""Commands for the two-agent task: navigation OR an arm goal, never both.

At every command event ArmTargetsCommand gives an env either navigation (the velocity command is live, each wrist
holds its rest pose, fixed in the pelvis frame) or, once the env's warm start is over, an arm goal (zero velocity, a
wrist pose per arm from a table of standing-reachable poses, lowered by the depth curriculum and fixed in the world).
The policies see the arm command in the pelvis frame, exact at the event and then moved by the pelvis motion
(apply_pelvis_motion; on the robot, FAST-LIO's odometry); rewards, reach detection and the curriculum use the world
target.
"""

from __future__ import annotations

import math
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

from .legs_rewards import standing_command

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
    pure_turn_speed, either sign), with probability pure_lateral_prob by a sideways walk (|vy| in pure_lateral_speed)
    and with probability small_command_prob by a small command (norm over (vx, vy, wz) in small_command_norm, a random
    direction, near the stand/walk gate): uniform sampling almost never draws any of them. In training the command
    changes at every navigation event (ArmTargetsCommand's nav_time_range); its own resampling_time_range only runs out
    inside arm goals, where it is zero anyway.

    **Filter.** `command`, what the observations, the tracking rewards, the gait clock and the metrics read, is the
    sampled command vel_command_b through a first-order low-pass with time constant command_filter_tau (sport mode's
    response), stepped once per policy step and zeroed at reset and during arm goals; 0: no filter.

    **Window.** The pelvis velocity (vx, vy, wz) and the command are each averaged over the last gait cycle (the env's
    gait_period): within a stride the pelvis sways, surges and twists about as much as a small command asks for.

    **Kernels.** The tracking rewards read lin_vel_std and ang_vel_std, each one stage of cfg.lin_vel_stds /
    cfg.ang_vel_stds for all envs, judged apart (_judge_kernels). A navigation segment, one velocity command held
    outside arm goals, ends at the next resample, arm goal or reset; it hits the linear kernel when its mean xy error
    (cycle-mean velocity) is under lin_vel_std, judged only if it commanded a walk (|v_xy| >= min_cos_speed), and the
    yaw kernel when its mean cycle-mean yaw-rate error is under ang_vel_std, judged only if it commanded a turn
    (|w_z| >= min_cos_speed). A segment ended by a fall misses. After every kernel_window_segments judged segments the
    stage narrows when at least kernel_promote_share hit and widens below kernel_demote_share: the kernels follow the
    error down, so standing still against a walk command stops paying.

    **Sport-mode metrics.** The sport-mode scorecard's tracking numbers (scripts/skrl/scorecard_metrics.py), measured
    on training's own navigation from the cycle means: a sample every sport_metric_stride steps per env once the
    window is full, sport_settle_s after a resample or reset and the window plus sport_settle_s after a push, while a
    command is given (|vx| or |vy| or |wz| over 0.05). The last sport_metric_samples samples of each kind are kept
    (sport_metrics): linear and angular error over all of them; cos and the speed ratio (median) where
    |v_cmd| >= 0.1; turn sense and the turn ratio (median) where |w_cmd| >= 0.1. The scorecard fits each controller's
    lag instead of leaving the transients out, and runs without pushes or randomization.

    **Logs.** Per episode: the navigation tracking error, the walking cos-sim of the commanded and the true xy
    velocity (|v_cmd| >= min_cos_speed), and the mean knee angle while standing and while walking.
    """

    cfg: ModalVelocityCommandCfg

    def __init__(self, cfg: ModalVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        zeros = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
        self._nav_steps, self._nav_err_xy, self._nav_err_yaw = zeros(), zeros(), zeros()
        self._walk_steps, self._walk_cos = zeros(), zeros()
        self._knee_ids = self.robot.find_joints(".*_knee_joint")[0]
        self._stand_steps, self._stand_knee, self._move_steps, self._move_knee = zeros(), zeros(), zeros(), zeros()
        # -- the gait-cycle window: velocity (vx, vy, wz) and command, newest at _slot - 1
        self._window = max(int(round(env.cfg.gait_period / env.step_dt)), 1)
        self._vel_samples = torch.zeros(self.num_envs, self._window, 3, device=self.device)
        self._cmd_samples = torch.zeros(self.num_envs, self._window, 3, device=self.device)
        self._count = zeros()
        self._slot = 0
        # -- tracking-kernel curriculum: per-env sums over the current navigation segment
        self._seg_steps, self._seg_err_xy, self._seg_err_yaw = zeros(), zeros(), zeros()
        self._seg_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.lin_kernel_stage = 0
        self.ang_kernel_stage = 0
        self._kernel_counts = {k: torch.zeros(2, device=self.device) for k in ("lin", "ang")}  # hits, judged
        self.lin_kernel_window_share = torch.tensor(float("nan"), device=self.device)
        self.ang_kernel_window_share = torch.tensor(float("nan"), device=self.device)
        # -- the command low-pass: one step of tau per policy step
        self.vel_command_filtered = torch.zeros_like(self.vel_command_b)
        self._filter_alpha = 1.0 - math.exp(-env.step_dt / cfg.command_filter_tau) if cfg.command_filter_tau > 0 else 1.0
        # -- sport-mode metrics: steps since the last resample, the sample rings
        self._since_cmd = zeros()
        self._settle_steps = int(round(cfg.sport_settle_s / env.step_dt))
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._rings = {key: [torch.zeros(cfg.sport_metric_samples, 2, device=self.device), 0, 0]
                       for key in ("all", "lin", "ang")}

    @property
    def command(self) -> torch.Tensor:
        """The command the policies and rewards see: the filtered one with command_filter_tau, else the sampled one."""
        return self.vel_command_filtered if self.cfg.command_filter_tau > 0 else self.vel_command_b

    @property
    def lin_vel_std(self) -> float:
        return self.cfg.lin_vel_stds[self.lin_kernel_stage]

    @property
    def ang_vel_std(self) -> float:
        return self.cfg.ang_vel_stds[self.ang_kernel_stage]

    def _end_segments(self, env_ids: torch.Tensor, fell: torch.Tensor | None = None):
        """Judge the navigation segments of env_ids that had steps, then clear their sums."""
        steps = self._seg_steps[env_ids]
        had = steps > 0
        if fell is None:
            fell = torch.zeros_like(had)
        cmd = self._seg_cmd[env_ids]
        min_speed = self.cfg.min_cos_speed
        for key, err, commanded, std in (
            ("lin", self._seg_err_xy, torch.norm(cmd[:, :2], dim=-1) >= min_speed, self.lin_vel_std),
            ("ang", self._seg_err_yaw, torch.abs(cmd[:, 2]) >= min_speed, self.ang_vel_std),
        ):
            judged = had & commanded
            hit = judged & ~fell & (err[env_ids] / steps.clamp(min=1.0) < std)
            self._kernel_counts[key] += torch.stack([hit.sum(), judged.sum()]).float()
        self._seg_steps[env_ids] = 0.0
        self._seg_err_xy[env_ids] = 0.0
        self._seg_err_yaw[env_ids] = 0.0
        self._judge_kernels()

    def _judge_kernels(self):
        cfg = self.cfg
        for key, stds in (("lin", cfg.lin_vel_stds), ("ang", cfg.ang_vel_stds)):
            counts = self._kernel_counts[key]
            if counts[1].item() < cfg.kernel_window_segments:
                continue
            share = counts[0] / counts[1]
            stage = getattr(self, f"{key}_kernel_stage")
            if share.item() >= cfg.kernel_promote_share and stage < len(stds) - 1:
                stage += 1
            elif share.item() < cfg.kernel_demote_share and stage > 0:
                stage -= 1
            setattr(self, f"{key}_kernel_stage", stage)
            setattr(self, f"{key}_kernel_window_share", share)
            counts.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"lin_kernel_stage": torch.tensor(self.lin_kernel_stage), "ang_kernel_stage": torch.tensor(self.ang_kernel_stage)}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        self.lin_kernel_stage = min(max(int(state["lin_kernel_stage"]), 0), len(self.cfg.lin_vel_stds) - 1)
        self.ang_kernel_stage = min(max(int(state["ang_kernel_stage"]), 0), len(self.cfg.ang_vel_stds) - 1)

    def _arm_mode(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.arm_command_name).arm_mode

    def _resample_command(self, env_ids: Sequence[int]):
        self._end_segments(torch.as_tensor(env_ids, device=self.device))
        self._since_cmd[env_ids] = 0.0
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device)
        ids = ids[~self.is_standing_env[ids]]
        u = torch.rand(len(ids), device=self.device)
        p_turn, p_lateral, p_small = self.cfg.pure_turn_prob, self.cfg.pure_lateral_prob, self.cfg.small_command_prob
        turn = ids[u < p_turn]
        lateral = ids[(u >= p_turn) & (u < p_turn + p_lateral)]
        for sub, axis, speed in ((turn, 2, self.cfg.pure_turn_speed), (lateral, 1, self.cfg.pure_lateral_speed)):
            if len(sub) == 0:
                continue
            sign = torch.where(torch.rand(len(sub), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[sub] = 0.0
            self.vel_command_b[sub, axis] = sign * torch.empty(len(sub), device=self.device).uniform_(*speed)
        small = ids[(u >= p_turn + p_lateral) & (u < p_turn + p_lateral + p_small)]
        if len(small) > 0:
            direction = torch.nn.functional.normalize(torch.randn(len(small), 3, device=self.device), dim=-1)
            norm = torch.empty(len(small), 1, device=self.device).uniform_(*self.cfg.small_command_norm)
            self.vel_command_b[small] = direction * norm

    def _update_command(self):
        super()._update_command()
        arm = self._arm_mode()
        self.vel_command_b[arm] = 0.0
        if self.cfg.command_filter_tau > 0:
            self.vel_command_filtered += self._filter_alpha * (self.vel_command_b - self.vel_command_filtered)
            self.vel_command_filtered[arm] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        nav = 1.0 - self._arm_mode().float()
        data = self.robot.data
        command = self.command
        vel = torch.cat([data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]], dim=-1)
        self._nav_steps += nav
        self._nav_err_xy += torch.norm(command[:, :2] - vel[:, :2], dim=-1) * nav
        self._nav_err_yaw += torch.abs(command[:, 2] - vel[:, 2]) * nav
        walk = nav * (torch.norm(command[:, :2], dim=-1) >= self.cfg.min_cos_speed).float()
        self._walk_steps += walk
        self._walk_cos += torch.nn.functional.cosine_similarity(command[:, :2], vel[:, :2], dim=-1, eps=1e-6) * walk
        knee = data.joint_pos[:, self._knee_ids].mean(dim=1)
        standing = standing_command(self._env, command).float() * nav
        self._stand_steps += standing
        self._stand_knee += knee * standing
        self._move_steps += nav - standing
        self._move_knee += knee * (nav - standing)
        # a navigation segment ends when an arm goal starts
        started = torch.nonzero((nav < 0.5) & (self._seg_steps > 0)).flatten()
        if len(started) > 0:
            self._end_segments(started)
        # the gait-cycle means; the kernels judge their errors
        self._vel_samples[:, self._slot] = vel
        self._cmd_samples[:, self._slot] = command
        self._slot = (self._slot + 1) % self._window
        self._count = torch.clamp(self._count + 1.0, max=self._window)
        vel_mean = self._vel_samples.sum(dim=1) / self._count.unsqueeze(-1)
        self._seg_steps += nav
        self._seg_err_xy += torch.norm(command[:, :2] - vel_mean[:, :2], dim=-1) * nav
        self._seg_err_yaw += torch.abs(command[:, 2] - vel_mean[:, 2]) * nav
        self._seg_cmd = torch.where(nav.unsqueeze(-1) > 0.5, command, self._seg_cmd)
        self._since_cmd += 1.0
        self._add_sport_samples(nav > 0.5, vel_mean)

    def _add_sport_samples(self, nav: torch.Tensor, vel: torch.Tensor):
        """One step's sport-mode metric samples (the class docstring's Sport-mode metrics) from vel, the cycle-mean
        velocity, and the cycle-mean command."""
        cfg, env = self.cfg, self._env
        step = env.common_step_counter
        cmd = self._cmd_samples.sum(dim=1) / self._count.clamp(min=1.0).unsqueeze(-1)
        valid = nav & (self._count >= self._window) & (self._since_cmd >= self._settle_steps)
        valid &= (self._env_ids + step) % cfg.sport_metric_stride == 0
        last_push = getattr(env, "last_push_step", None)  # no pushes (the scorecard): none to leave out
        if last_push is not None:
            valid &= (step - last_push) >= self._window + self._settle_steps
        valid &= (cmd[:, :2].abs().max(dim=1).values > 0.05) | (cmd[:, 2].abs() > 0.05)
        if not valid.any():
            return
        cmd, vel = cmd[valid], vel[valid]
        self._ring_add("all", torch.stack([torch.norm(vel[:, :2] - cmd[:, :2], dim=-1), (vel[:, 2] - cmd[:, 2]).abs()], dim=-1))
        speed = torch.norm(cmd[:, :2], dim=-1)
        lin = speed >= 0.1
        if lin.any():
            cos = torch.nn.functional.cosine_similarity(cmd[lin, :2], vel[lin, :2], dim=-1, eps=1e-6)
            self._ring_add("lin", torch.stack([cos, torch.norm(vel[lin, :2], dim=-1) / speed[lin]], dim=-1))
        ang = cmd[:, 2].abs() >= 0.1
        if ang.any():
            w, yaw = cmd[ang, 2], vel[ang, 2]
            self._ring_add("ang", torch.stack([yaw / w, (torch.sign(yaw) == torch.sign(w)).float()], dim=-1))

    def _ring_add(self, key: str, rows: torch.Tensor):
        ring = self._rings[key]
        buffer, slot, filled = ring
        size = len(buffer)
        rows = rows[-size:]
        buffer[(slot + torch.arange(len(rows), device=self.device)) % size] = rows
        ring[1] = (slot + len(rows)) % size
        ring[2] = min(filled + len(rows), size)

    def sport_metrics(self, min_samples: int = 2000) -> dict[str, float]:
        """The sport-mode metrics over the kept samples (nan where a kind has fewer than min_samples): turn_sense,
        speed_ratio, turn_ratio, cos, ang_err, lin_err, as the scorecard names them, and each kind's sample count."""
        out = {}
        for key, names in (("all", ("lin_err", "ang_err")), ("lin", ("cos", "speed_ratio")),
                           ("ang", ("turn_ratio", "turn_sense"))):
            buffer, _, filled = self._rings[key]
            out[f"n_{key}"] = float(filled)
            data = buffer[:filled]
            for col, name in enumerate(names):
                if filled < min_samples:
                    out[name] = float("nan")
                elif name in ("speed_ratio", "turn_ratio"):
                    out[name] = data[:, col].median().item()
                else:
                    out[name] = data[:, col].mean().item()
        return out

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        ended = torch.arange(self.num_envs, device=self.device)[ids]
        terminated = getattr(getattr(self._env, "termination_manager", None), "terminated", None)
        self._end_segments(ended, None if terminated is None else terminated[ended])
        extras = {}
        _add_weighted_mean(extras, "nav_error_vel_xy", self._nav_err_xy[ids], self._nav_steps[ids])
        _add_weighted_mean(extras, "nav_error_vel_yaw", self._nav_err_yaw[ids], self._nav_steps[ids])
        _add_weighted_mean(extras, "nav_cos_vel_xy", self._walk_cos[ids], self._walk_steps[ids])
        _add_weighted_mean(extras, "stand_knee", self._stand_knee[ids], self._stand_steps[ids])
        _add_weighted_mean(extras, "walk_knee", self._move_knee[ids], self._move_steps[ids])
        for buffer in (self._nav_steps, self._nav_err_xy, self._nav_err_yaw, self._walk_steps, self._walk_cos,
                       self._stand_steps, self._stand_knee, self._move_steps, self._move_knee, self._since_cmd,
                       self._count):
            buffer[ids] = 0.0
        self._vel_samples[ended] = 0.0
        self._cmd_samples[ended] = 0.0
        extras.update(super().reset(env_ids))
        self.vel_command_filtered[ended] = 0.0
        return extras


@configclass
class ModalVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ModalVelocityCommand
    arm_command_name: str = MISSING
    pure_turn_prob: float = 0.0
    pure_turn_speed: tuple[float, float] = (0.2, 0.97)       # rad/s
    pure_lateral_prob: float = 0.0
    pure_lateral_speed: tuple[float, float] = (0.15, 0.54)   # m/s
    small_command_prob: float = 0.0
    """Share of moving commands drawn small: norm over (vx, vy, wz) in small_command_norm, a random direction."""
    small_command_norm: tuple[float, float] = (0.05, 0.15)
    min_cos_speed: float = 0.1
    """The commanded |v_xy| (m/s) from which a step counts for nav_cos_vel_xy and a segment is judged for the linear
    kernel, and the commanded |w_z| (rad/s) from which a segment is judged for the yaw kernel."""
    command_filter_tau: float = 0.5
    """s: the first-order low-pass on the command the policies and rewards see (`command`); 0: none."""
    # -- tracking-kernel curriculum (ModalVelocityCommand._judge_kernels): the tracking terms' stds per stage,
    # narrowest last
    lin_vel_stds: tuple[float, ...] = MISSING
    """m/s: the xy velocity kernel."""
    ang_vel_stds: tuple[float, ...] = MISSING
    """rad/s: the yaw-rate kernel."""
    kernel_window_segments: int = 20000
    """Judged navigation segments (all envs) per window, each kernel apart."""
    kernel_promote_share: float = 0.5
    kernel_demote_share: float = 0.25
    # -- sport-mode metrics
    sport_metric_samples: int = 2**18
    """Samples kept per kind (ModalVelocityCommand.sport_metrics)."""
    sport_metric_stride: int = 8
    """A sample from each env every this many steps (staggered over the envs)."""
    sport_settle_s: float = 1.0
    """s: left out after a resample or reset, and (plus the window) after a push."""


class ArmTargetsCommand(CommandTerm):
    """Per env, either navigation (each wrist holds its rest pose) or an arm goal (one wrist pose per arm in the world).

    **Events.** Reset or the timer running out. Once goals_enabled (the env sets it when its warm start ends), the
    next task is an arm goal with probability arm_goal_prob, lasting resampling_time_range; otherwise navigation for
    nav_time_range, with a fresh velocity command. During navigation each wrist holds its rest pose, the wrist pose
    at the default joint angles, fixed in the pelvis frame. During an arm goal the velocity command is zero
    (ModalVelocityCommand).

    **Goals.** The tables hold collision-free wrist poses at random arm joint angles with the robot standing, in the
    pelvis frame, those less than min_target_x ahead of the pelvis dropped, balanced over balance_cell cubes. They
    are built once per run and saved to <log_dir>/<table_file>. A goal's height h above the ground is drawn
    uniformly from [z_min, z_max]. Each arm's pose is drawn uniformly from its table rows within arm_height_spread
    of h, so the wrists sit at h +- arm_height_spread with their table's reach at that height. Below what the
    standing table reaches, the rows come from its lowest band and both are lowered together by the difference
    (target_drop): the robot has to crouch. The poses are placed in the standing frame (the pelvis's x, y and yaw,
    standing_height above the scanned ground: the stance the tables were built at, assets.h1_2.TABLE_PELVIS_HEIGHT)
    and kept fixed in the world for the goal.

    **Curriculum.** z_max stays put, a height the standing arms reach; z_min falls with a depth level k_d per env,
    0..levels: z_min = z_min_start - depth_per_level * k_d. Goals keep coming from the whole range, high and low.
    Only the low goals (height within judge_band of z_min) are judged: once the env's last level_window of them have
    ended, k_d rises when their mean closest approach (per goal, the least mean of both wrists' position errors) is
    within promote_error and falls when it is beyond demote_error; each move restarts the window. A fall during any
    arm goal drops k_d by one and restarts the window.

    **Kernels.** The annealed tracking rewards read fine_pos_std and rot_std, one stage of cfg.fine_pos_stds /
    cfg.rot_stds for all envs. After every kernel_window_goals ended arm goals the stage narrows when enough of them
    came within its kernels, and widens when too few did (_judge_kernels), so the kernels follow the error down.

    **Reach.** A goal lasts its timer. It is reached once, when both wrists are first held inside the tolerances
    for reach_hold_s (the bonus is paid then); the tracking terms keep paying for staying on target.

    **What the policies see.** believed_b, the targets in the pelvis frame: exact at the event, then moved each
    step by the pelvis motion the env reports (apply_pelvis_motion).
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
        self.goals_enabled_at: int | None = None  # the env step the warm start ended (training only): the ramp's start
        self.arm_mode = flags()
        self.anchor_w = torch.zeros(shape, device=self.device)  # arm goal: world-fixed targets
        self.believed_b = torch.zeros(shape, device=self.device)  # what the policies see
        self.anchor_w[..., 3] = 1.0
        self.believed_b[..., 3] = 1.0
        self.target_drop = zeros()  # how far both poses were lowered below the standing table (m)
        self.goal_height = zeros()  # the goal's height h above the ground (m)
        self.needs_crouch = flags()  # a target below anything the table reaches standing
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
        # each arm goal's closest approach (mean of both wrists' position error, m) and least rotation error (rad)
        self.goal_best_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.goal_best_rot_error = torch.full((self.num_envs,), float("inf"), device=self.device)

        # -- tracking-kernel curriculum: one stage for all envs, judged on windows of ended arm goals
        if len(cfg.fine_pos_stds) != len(cfg.rot_stds):
            raise ValueError("fine_pos_stds and rot_stds need one entry per kernel stage")
        self.kernel_stage = 0
        self._kernel_hits = torch.zeros((), device=self.device)
        self._kernel_goals = torch.zeros((), device=self.device)
        self.kernel_window_share = torch.tensor(float("nan"), device=self.device)  # the last judged window's

        # -- metrics. CommandTerm.reset logs their mean over the reset envs, then zeroes them.
        for name in ("command_drift", "goals_reached", "goals_missed", "crouch_goals_reached", "crouch_goals_missed"):
            self.metrics[name] = zeros()
        # per-episode sums for the mode-split metrics, logged as weighted means by reset()
        self._goal_steps, self._goal_pos_err, self._goal_rot_err = zeros(), zeros(), zeros()
        self._rest_steps, self._rest_pos_err = zeros(), zeros()
        self._crouch_n, self._crouch_t, self._crouch_p = zeros(), zeros(), zeros()
        self._nav_n, self._nav_p = zeros(), zeros()
        self._best_err_sum, self._best_err_n = zeros(), zeros()

        data = self._load_or_build_tables(env)
        self._tables: list[torch.Tensor] = data["tables"]
        # (num_arms,): the lowest wrist height each arm reaches standing, pelvis frame
        self._standing_min_z = torch.stack([table[:, 2].min() for table in self._tables])
        # each table sorted by height, for drawing rows within arm_height_spread of a goal height
        order = [table[:, 2].argsort() for table in self._tables]
        self._sorted_tables = [table[o] for table, o in zip(self._tables, order)]
        self._sorted_z = [table[:, 2].contiguous() for table in self._sorted_tables]
        spread = cfg.arm_height_spread
        # the lowest band every arm's table fills (its 0.5th percentile up), and the highest goal it can hold
        floor = max(torch.quantile(z, 0.005).item() for z in self._sorted_z)
        top = min(torch.quantile(z, 0.995).item() for z in self._sorted_z)
        self._lowest_source = floor + spread
        if cfg.z_max - cfg.standing_height + spread > top:
            raise ValueError(f"z_max {cfg.z_max} m: the standing tables reach {top + cfg.standing_height:.2f} m;"
                             f" z_max + arm_height_spread must stay below it")
        if self.z_min(torch.tensor(cfg.levels)).item() >= cfg.z_max or cfg.z_min_start >= cfg.z_max:
            raise ValueError("z_min must stay below z_max at every depth level")
        # (num_arms, 7): the wrist poses at the default joint angles, the navigation rest pose
        self.rest_pose_b = torch.stack(data["default_poses"])

    def __str__(self) -> str:
        cfg = self.cfg
        msg = "ArmTargetsCommand:\n"
        msg += f"\tBodies: {cfg.body_names}\n"
        msg += f"\tArm goal probability once enabled: {cfg.arm_goal_prob}, navigation {cfg.nav_time_range} s\n"
        msg += (f"\tGoal heights: U[z_min, {cfg.z_max}] m, z_min = {cfg.z_min_start} - {cfg.depth_per_level} k_d m,"
                f" k_d 0..{cfg.levels}\n")
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
        """True targets in world, (num_envs, num_arms, 7): the anchor for an arm goal, the rest pose on the pelvis
        otherwise."""
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

    def apply_pelvis_motion(self, delta_pos: torch.Tensor, delta_quat: torch.Tensor, env_mask: torch.Tensor):
        """Re-express the arm-goal commands of env_mask's envs after the pelvis moved by (delta_pos, delta_quat), in
        the previous pelvis frame. A world-fixed target obeys T_prev o c_prev = T_new o c_new, so
        c_new = delta^-1 o c_prev."""
        mask = env_mask & self.arm_mode
        moved = _relative(delta_pos, delta_quat, self.believed_b)
        self.believed_b[mask] = moved[mask]

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

    def z_min(self, level: torch.Tensor) -> torch.Tensor:
        """The lowest goal height above the ground at each depth level (m)."""
        return self.cfg.z_min_start - self.cfg.depth_per_level * level.float()

    def update_levels(self, env_ids: Sequence[int], fell: torch.Tensor):
        """At episode end: a fall during an arm goal drops the depth level by one and restarts the env's window."""
        env_ids = torch.as_tensor(env_ids, device=self.device)
        down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        down[env_ids] = fell & self.arm_mode[env_ids]
        self.level -= (down & (self.level > 0)).long()
        self.outcome_count = torch.where(down, 0, self.outcome_count)

    def _record_outcomes(self, ended: torch.Tensor):
        """Push this step's ended low arm goals' closest approach into each env's window of its last level_window low
        goals; move the level on a full window, and restart the window when it moves."""
        cfg, window = self.cfg, self.cfg.level_window
        self._judge_kernels(ended)
        ended = ended & (self.goal_height <= self.z_min(self.level) + cfg.judge_band)
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

    @property
    def fine_pos_std(self) -> float:
        """The fine wrist-position kernel's std at the current kernel stage (m)."""
        return self.cfg.fine_pos_stds[self.kernel_stage]

    @property
    def rot_std(self) -> float:
        """The wrist-rotation kernel's std at the current kernel stage (rad)."""
        return self.cfg.rot_stds[self.kernel_stage]

    def _judge_kernels(self, ended: torch.Tensor):
        """Count the ended arm goals that came within the current stage's kernels (closest approach under
        fine_pos_std and least rotation error under rot_std, each the mean of both wrists, not necessarily at the same
        instant). After every kernel_window_goals of them: narrow the kernels a stage if at least kernel_promote_share
        did, widen them a stage if fewer than kernel_demote_share did, and start a new window."""
        cfg = self.cfg
        hit = ended & (self.goal_best_error < self.fine_pos_std) & (self.goal_best_rot_error < self.rot_std)
        self._kernel_hits += hit.sum()
        self._kernel_goals += ended.sum()
        if self._kernel_goals.item() < cfg.kernel_window_goals:
            return
        share = self._kernel_hits / self._kernel_goals
        self.kernel_window_share = share
        if share.item() >= cfg.kernel_promote_share and self.kernel_stage < len(cfg.fine_pos_stds) - 1:
            self.kernel_stage += 1
        elif share.item() < cfg.kernel_demote_share and self.kernel_stage > 0:
            self.kernel_stage -= 1
        self._kernel_hits.zero_()
        self._kernel_goals.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Curriculum state, saved with the env's state so a resumed or evaluated run keeps its levels."""
        return {"level": self.level.clone(), "kernel_stage": torch.tensor(self.kernel_stage)}

    def load_state_dict(self, state: dict[str, torch.Tensor]):
        saved = state["level"].to(self.device)
        pick = slice(None) if len(saved) == self.num_envs else torch.randint(0, len(saved), (self.num_envs,), device=self.device)
        self.level[:] = saved[pick].clamp(0, self.cfg.levels)
        self.kernel_stage = min(max(int(state["kernel_stage"]), 0), len(self.cfg.fine_pos_stds) - 1)

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
        rest = 1.0 - arm
        self._rest_steps += rest
        self._rest_pos_err += pos_error.mean(dim=1) * rest
        ground = ground_height(self.scanner)
        counted = arm * torch.isfinite(ground).float()
        pelvis_drop = torch.nan_to_num(self.cfg.standing_height - (self.robot.data.root_pos_w[:, 2] - ground), nan=0.0)
        self._crouch_n += counted
        self._crouch_t += self.target_drop * counted
        self._crouch_p += pelvis_drop * counted
        # the pelvis below the table stance while navigating (the sport stance stands 2.3 cm lower)
        walking = (1.0 - arm) * torch.isfinite(ground).float()
        self._nav_n += walking
        self._nav_p += pelvis_drop * walking

        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        believed_w = _apply(root_pos, root_quat, self.believed_b)
        self.metrics["command_drift"] = torch.norm(believed_w[..., :3] - self.anchor_w[..., :3], dim=-1).mean(1) * arm

        within = (pos_error < self.cfg.reach_pos_tol).all(dim=1) & (rot_error < self.cfg.reach_rot_tol).all(dim=1)
        step_dt = self._env.step_dt
        self.hold_time = torch.where(within & self.arm_mode, self.hold_time + step_dt, torch.zeros_like(self.hold_time))
        # reached once, when first held; judged when the goal's timer runs out
        self.just_reached = (self.hold_time >= self.cfg.reach_hold_s) & ~self.goal_reached & self.arm_mode
        self.goal_reached |= self.just_reached
        ended = (self.time_left - step_dt <= 0.0) & self.arm_mode
        timed_out = ended & ~self.goal_reached
        self.metrics["goals_reached"] += self.just_reached.float()
        self.metrics["goals_missed"] += timed_out.float()
        self.metrics["crouch_goals_reached"] += (self.just_reached & self.needs_crouch).float()
        self.metrics["crouch_goals_missed"] += (timed_out & self.needs_crouch).float()
        best = torch.minimum(self.goal_best_error, pos_error.mean(dim=1))
        self.goal_best_error = torch.where(self.arm_mode, best, self.goal_best_error)
        best_rot = torch.minimum(self.goal_best_rot_error, rot_error.mean(dim=1))
        self.goal_best_rot_error = torch.where(self.arm_mode, best_rot, self.goal_best_rot_error)
        self._best_err_sum += torch.where(ended, self.goal_best_error.clamp(max=10.0), 0.0)
        self._best_err_n += ended.float()
        # before CommandTerm.compute resamples the ended goals, so the next goal is drawn at the new level
        self._record_outcomes(ended)

    def _resample_command(self, env_ids: Sequence[int]):
        self.invalidate_errors()
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        arm_goal = (torch.rand(n, device=self.device) < self.arm_goal_prob) & self.goals_enabled
        self.arm_mode[env_ids] = arm_goal
        self.goal_best_error[env_ids] = float("inf")
        self.goal_best_rot_error[env_ids] = float("inf")
        self.goal_reached[env_ids] = False
        self.hold_time[env_ids] = 0.0
        self.target_drop[env_ids] = 0.0
        self.goal_height[env_ids] = 0.0
        self.needs_crouch[env_ids] = False

        # -- navigation: the rest pose in the pelvis frame, a fresh velocity command
        nav_ids = env_ids[~arm_goal]
        if len(nav_ids) > 0:
            self.believed_b[nav_ids] = self.rest_pose_b
            self.time_left[nav_ids] = torch.empty(len(nav_ids), device=self.device).uniform_(*self.cfg.nav_time_range)
            velocity._resample(nav_ids)

        # -- arm goal: a height in [z_min, z_max], table poses near it (lowered below the table), anchored in the world
        arm_ids = env_ids[arm_goal]
        m = len(arm_ids)
        if m == 0:
            return
        cfg = self.cfg
        z_min = self.z_min(self.level[arm_ids])
        height = z_min + torch.rand(m, device=self.device) * (cfg.z_max - z_min)
        height_s = height - cfg.standing_height  # in the standing frame
        source = height_s.clamp(min=self._lowest_source)
        targets_s = self._draw_rows(source)
        drop = source - height_s
        targets_s[..., 2] -= drop.unsqueeze(1)
        self.target_drop[arm_ids] = drop
        self.goal_height[arm_ids] = height
        self.needs_crouch[arm_ids] = (targets_s[..., 2] < self._standing_min_z).any(dim=1)

        origin, quat = self.standing_frame_w()
        if self._at_reset:
            # the height scan still shows where the robot was before the reset teleported it
            origin[arm_ids, 2] = self._env.scene.env_origins[arm_ids, 2] + self.cfg.standing_height
        self.anchor_w[arm_ids] = _apply(origin[arm_ids], quat[arm_ids], targets_s)
        self.believed_b[arm_ids] = _relative(
            self.robot.data.root_pos_w[arm_ids], self.robot.data.root_quat_w[arm_ids], self.anchor_w[arm_ids]
        )
        velocity.vel_command_b[arm_ids] = 0.0
        velocity.vel_command_filtered[arm_ids] = 0.0

    def _draw_rows(self, source: torch.Tensor) -> torch.Tensor:
        """(len(source), num_arms, 7): per arm a table row within arm_height_spread of source (pelvis-frame height)."""
        rows = []
        for table, z in zip(self._sorted_tables, self._sorted_z):
            lo = torch.searchsorted(z, source - self.cfg.arm_height_spread)
            hi = torch.searchsorted(z, source + self.cfg.arm_height_spread, right=True)
            pick = lo + (torch.rand(len(source), device=self.device) * (hi - lo)).long().clamp(max=(hi - lo - 1).clamp(min=0))
            rows.append(table[pick.clamp(max=len(table) - 1)])
        return torch.stack(rows, dim=1)

    @property
    def arm_goal_prob(self) -> float:
        """The share of tasks that are arm goals: cfg.arm_goal_prob, ramped linearly from cfg.arm_goal_ramp_start (or
        arm_goal_prob, if lower) over cfg.arm_goal_ramp_steps env steps after the warm start ends (outside training, or
        resumed: no ramp)."""
        cfg = self.cfg
        if self.goals_enabled_at is None or cfg.arm_goal_ramp_steps <= 0:
            return cfg.arm_goal_prob
        start = min(cfg.arm_goal_ramp_start, cfg.arm_goal_prob)
        frac = min(1.0, (self._env.common_step_counter - self.goals_enabled_at) / cfg.arm_goal_ramp_steps)
        return start + frac * (cfg.arm_goal_prob - start)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # one marker set per arm and kind: with both arms' poses in one point instancer, Isaac rendered the second
        # instance as stray lines ("FabricManager::initializePointInstancer mismatched prototypes")
        if debug_vis and not hasattr(self, "pose_visualizers"):
            self.pose_visualizers = [
                [VisualizationMarkers(cfg.replace(prim_path=f"{cfg.prim_path}_{arm}")) for arm in range(len(self.cfg.body_names))]
                for cfg in (self.cfg.goal_pose_visualizer_cfg, self.cfg.believed_pose_visualizer_cfg,
                            self.cfg.current_pose_visualizer_cfg)
            ]
        if hasattr(self, "pose_visualizers"):
            for markers in self.pose_visualizers:
                for marker in markers:
                    marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        # the robot can be de-initialized while the callback is still subscribed
        if not self.robot.is_initialized:
            return
        root_pos, root_quat = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        poses = (self.targets_w(), _apply(root_pos, root_quat, self.believed_b),
                 self.robot.data.body_link_pose_w[:, self.body_ids])
        for markers, pose in zip(self.pose_visualizers, poses):
            for arm, marker in enumerate(markers):
                marker.visualize(pose[:, arm, :3], pose[:, arm, 3:])


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
    """Pelvis height above the ground of the stance the tables were built at: the standing frame's origin."""

    # -- tasks
    arm_goal_prob: float = 0.5
    """Once goals are enabled, the share of tasks that are arm goals."""
    arm_goal_ramp_start: float = 0.1
    arm_goal_ramp_steps: int = 20000
    """Policy steps (200 s at 100 Hz): after the warm start the arm-goal share ramps from arm_goal_ramp_start to
    arm_goal_prob over these; 0: at once (the falls then spike to 85-99%)."""
    nav_time_range: tuple[float, float] = (4.0, 8.0)
    """s: a navigation task's duration."""
    reach_pos_tol: float = 0.05
    reach_rot_tol: float = 0.35
    reach_hold_s: float = 1.0

    # -- height curriculum: goal heights above the ground ~ U[z_min, z_max]
    z_max: float = 1.7
    """m: the highest goal height, fixed; the standing tables still reach it with arm_height_spread to spare."""
    z_min_start: float = 1.0
    """m: the lowest goal height at depth level 0 (about the lowest the standing arms reach)."""
    levels: int = 8
    depth_per_level: float = 0.1
    """m: z_min = z_min_start - depth_per_level * k_d."""
    arm_height_spread: float = 0.15
    """m: each wrist sits within this of the goal height."""
    judge_band: float = 0.3
    """m: goals no higher than z_min + this are the low goals the depth level is judged on."""
    level_window: int = 5
    promote_error: float = 0.12
    demote_error: float = 0.18

    # -- tracking-kernel curriculum (ArmTargetsCommand._judge_kernels): the annealed reward terms' stds per stage,
    # narrowest last
    fine_pos_stds: tuple[float, ...] = MISSING
    """m: the fine wrist-position kernel."""
    rot_stds: tuple[float, ...] = MISSING
    """rad: the wrist-rotation kernel."""
    kernel_window_goals: int = 20000
    """Ended arm goals (all envs) per judged window."""
    kernel_promote_share: float = 0.5
    kernel_demote_share: float = 0.25

    # -- tables
    table_file: str = "arm_target_tables_v3.pt"
    """Loaded when present and built with the same settings (a relative path: under the env's log_dir); otherwise
    built and saved there."""
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
