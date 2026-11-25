"""Main's cooperative-task commands as a law, and the segment process that runs it in imagination (torch only).

Stage 2 through the world model trains a command encoder on imagined rollouts of main's task
(``LocoManip-Marl-Pulse-v0``), so the commands an imagined rollout holds must be drawn as main's env draws them:

- ``VelocityLaw``: the legs' velocity command, ``ModalVelocityCommand._resample_command``
  (``locomanip_marl/mdp/commands.py:240-255``, Isaac Lab's ``UniformVelocityCommand._resample_command`` under it):
  uniform over the ranges, ``rel_standing_envs`` of the draws zero, and of the moving ones ``pure_turn_prob`` a turn
  in place and ``pure_lateral_prob`` a walk sideways, at their speed ranges, either sign;
- ``ArmGoalLaw``: both wrists' goals, ``ArmTargetsCommand._draw_arm_goals`` (``commands.py:897-941``) with
  ``pelvis_fixed_goals`` (the Pulse / Direct task's: the drawn pose is the pelvis-frame command for the whole goal):
  a goal height uniform in ``z_range`` above the ground, per arm a table row within ``height_spread`` of it
  (``_draw_rows``, ``commands.py:974-982``), lowered together below what the table reaches
  (``_lowest_source``, ``commands.py:510-512``), moved toward the rest pose with ``offtable_prob`` (``_toward_rest``,
  ``commands.py:943-956``), or the rest pose itself with ``rest_prob``. Left out: the collision screen and
  redraws (``collision_check``) and the joint seeds, which need the arm kinematics, not the command;
- ``CommandProcess``: segments of geometric length, ending with probability ``dt / mean_hold_s`` per step (main draws
  each command's hold from ``nav_time_range`` / ``resampling_time_range``, 4-8 s, mean 6 s), the velocity command and
  the arm goal on their own timers as in Marl-Direct (``marl_direct_env_cfg.py``, ``independent_of_legs``). A new
  segment is a clip segment (its command the caller's clip's, ``refresh``) with probability ``clip_prob``, else a law
  draw. The seen velocity command is main's low-pass of the drawn one (``command_filter_tau``,
  ``commands.py:171-172, 257-263``), carried across segments as main's is.

Quaternions are w-first, qw >= 0, as ``ArmTargetsCommand.believed_b``.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch


def _rand(gen: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.rand(*shape, generator=gen, device=gen.device)


class VelocityLaw:
    """Main's velocity-command draw (module docstring)."""

    def __init__(self, ranges: tuple[tuple[float, float], ...], standing_prob: float, turn_prob: float,
                 lateral_prob: float, turn_speed: tuple[float, float], lateral_speed: tuple[float, float],
                 filter_tau: float = 0.0):
        self.ranges = torch.tensor(ranges, dtype=torch.float32)  # (3, 2): vx, vy, wz
        self.standing_prob, self.turn_prob, self.lateral_prob = standing_prob, turn_prob, lateral_prob
        self.turn_speed, self.lateral_speed = turn_speed, lateral_speed
        self.filter_tau = filter_tau

    @classmethod
    def from_cfg(cls, cfg) -> VelocityLaw:
        """From a ``ModalVelocityCommandCfg`` (``MarlCommandsCfg.base_velocity``, locomanip_marl_env_cfg.py:159-176)."""
        r = cfg.ranges
        return cls((tuple(r.lin_vel_x), tuple(r.lin_vel_y), tuple(r.ang_vel_z)), cfg.rel_standing_envs,
                   cfg.pure_turn_prob, cfg.pure_lateral_prob, tuple(cfg.pure_turn_speed),
                   tuple(cfg.pure_lateral_speed), getattr(cfg, "command_filter_tau", 0.0))

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` commands ``[vx, vy, wz]`` (pelvis frame, m/s, rad/s), (n, 3)."""
        lo, hi = self.ranges[:, 0].to(gen.device), self.ranges[:, 1].to(gen.device)
        # UniformVelocityCommand._resample_command: each axis uniform, then standing with rel_standing_envs (<=)
        v = lo + (hi - lo) * _rand(gen, n, 3)
        standing = _rand(gen, n) <= self.standing_prob
        # ModalVelocityCommand._resample_command (commands.py:245-255): of the moving, u < p_turn turns in place,
        # p_turn <= u < p_turn + p_lateral walks sideways; a fresh sign and speed each
        u = _rand(gen, n)
        turn = ~standing & (u < self.turn_prob)
        lateral = ~standing & (u >= self.turn_prob) & (u < self.turn_prob + self.lateral_prob)
        sign = torch.where(_rand(gen, n) < 0.5, -1.0, 1.0)
        turn_rate = sign * (self.turn_speed[0] + (self.turn_speed[1] - self.turn_speed[0]) * _rand(gen, n))
        side = sign * (self.lateral_speed[0] + (self.lateral_speed[1] - self.lateral_speed[0]) * _rand(gen, n))
        zero = torch.zeros_like(v)
        v = torch.where(turn[:, None], torch.stack([zero[:, 0], zero[:, 1], turn_rate], -1), v)
        v = torch.where(lateral[:, None], torch.stack([zero[:, 0], side, zero[:, 2]], -1), v)
        # _update_command zeroes the standing envs' command
        return torch.where(standing[:, None], zero, v)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _unique(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., :1] < 0, -q, q)


def _toward(rest: torch.Tensor, goal: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Poses (..., 7) ``scale`` (..., 1) of the way from ``rest`` to ``goal``: position linearly, orientation along
    the shortest rotation (``ArmTargetsCommand._toward_rest``, commands.py:943-956)."""
    pos = rest[..., :3] + scale * (goal[..., :3] - rest[..., :3])
    q_rest = rest[..., 3:]
    conj = q_rest * torch.tensor([1.0, -1.0, -1.0, -1.0], device=rest.device)
    d = _unique(_quat_mul(conj, goal[..., 3:]))  # isaaclab's axis_angle_from_quat takes w >= 0 first
    half = torch.atan2(d[..., 1:].norm(dim=-1, keepdim=True), d[..., :1])
    axis = d[..., 1:] / d[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-12)
    h = half * scale
    q = _quat_mul(q_rest, torch.cat([h.cos(), h.sin() * axis], -1))
    return torch.cat([pos, _unique(q)], -1)


class ArmGoalLaw:
    """Main's arm-goal draw with pelvis-fixed goals (module docstring)."""

    def __init__(self, tables: list[torch.Tensor], rest: torch.Tensor, standing_height: float,
                 z_range: tuple[float, float] = (1.0, 1.7), height_spread: float = 0.15, offtable_prob: float = 0.0,
                 offtable_scales: tuple[float, float] = (0.25, 1.0), rest_prob: float = 0.0):
        # ArmTargetsCommandCfg's defaults: z_min 1.0, z_max 1.7, arm_height_spread 0.15 (commands.py:1080-1085)
        self.sorted_tables, self.sorted_z = [], []
        for table in tables:
            order = table[:, 2].argsort()
            self.sorted_tables.append(table[order].float())
            self.sorted_z.append(table[order, 2].contiguous().float())
        self.rest = rest.float()  # (2, 7): the wrists at the default joint angles, the navigation rest pose
        self.standing_height, self.z_range, self.height_spread = standing_height, z_range, height_spread
        self.offtable_prob, self.offtable_scales, self.rest_prob = offtable_prob, offtable_scales, rest_prob
        # commands.py:509-517: the lowest band every arm's table fills, and the highest goal it can hold
        floor = max(torch.quantile(z, 0.005).item() for z in self.sorted_z)
        top = min(torch.quantile(z, 0.995).item() for z in self.sorted_z)
        self.lowest_source = floor + height_spread
        if z_range[1] - standing_height + height_spread > top:
            raise ValueError(f"z_max {z_range[1]} m: the standing tables reach {top + standing_height:.2f} m;"
                             f" z_max + height_spread must stay below it")
        if z_range[0] >= z_range[1]:
            raise ValueError("z_min must stay below z_max")

    @classmethod
    def from_table(cls, path: str | Path, standing_height: float, **kwargs) -> ArmGoalLaw:
        """From an ``ArmTargetsCommand`` table file (``{"tables": [(rows, 7)] * 2, "default_poses": [(7,)] * 2}``,
        pelvis frame at the stance the tables were built at). ``standing_height`` (m) is the env's
        ``arm_targets.standing_height``; ``kwargs`` the other ``ArmGoalLaw`` fields (``from_cmd_cfg`` reads them)."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(list(data["tables"]), torch.stack(list(data["default_poses"])), standing_height, **kwargs)

    @staticmethod
    def cmd_cfg_kwargs(cfg) -> dict:
        """The law's fields from an ``ArmTargetsCommandCfg`` (refuses a world-fixed goal: its pelvis-frame command
        moves with the pelvis, which a law of pelvis-frame poses does not model)."""
        if not getattr(cfg, "pelvis_fixed_goals", False):
            raise ValueError("ArmGoalLaw models pelvis_fixed_goals only (LocoManip-Marl-Direct-v0 / -Pulse-v0)")
        return {"z_range": (cfg.z_min, cfg.z_max), "height_spread": cfg.arm_height_spread,
                "offtable_prob": cfg.offtable_prob, "offtable_scales": tuple(cfg.offtable_scale_range),
                "rest_prob": cfg.rest_goal_prob}

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` goals, both wrists' poses in the pelvis frame ``[x, y, z, qw, qx, qy, qz]``, qw >= 0, (n, 2, 7)."""
        device = gen.device
        z_min, z_max = self.z_range
        height_s = z_min + _rand(gen, n) * (z_max - z_min) - self.standing_height  # in the standing frame
        source = height_s.clamp(min=self.lowest_source)
        rows = []
        for table, z in zip(self.sorted_tables, self.sorted_z):
            table, z = table.to(device), z.to(device)
            lo = torch.searchsorted(z, source - self.height_spread)
            hi = torch.searchsorted(z, source + self.height_spread, right=True)
            pick = lo + (_rand(gen, n) * (hi - lo)).long().clamp(max=(hi - lo - 1).clamp(min=0))
            rows.append(table[pick.clamp(max=len(table) - 1)])
        goal = torch.stack(rows, 1)
        goal[..., 2] -= (source - height_s).unsqueeze(1)  # both lowered by the difference
        rest = self.rest.to(device).expand(n, 2, 7)
        if self.offtable_prob > 0.0:
            low, high = self.offtable_scales
            scale = (low + (high - low) * _rand(gen, n)).view(n, 1, 1)
            off = _rand(gen, n) < self.offtable_prob
            goal = torch.where(off.view(n, 1, 1), _toward(rest, goal, scale), goal)
        rest_goal = _rand(gen, n) < self.rest_prob
        return torch.where(rest_goal.view(n, 1, 1), rest, goal)


class CommandProcess:
    """Both commands' segment process over imagined steps (module docstring), functional: the state is a dict.

    State keys: ``velocity`` (n, 3) the command the policies and rewards see (main's ``command``, low-passed),
    ``velocity_target`` (n, 3) the drawn one (``vel_command_b``), ``arm`` (n, 2, 7) the pelvis-frame arm goal,
    ``clip_velocity`` / ``clip_arm`` (n,) bool the rows on a clip segment, ``new_velocity`` / ``new_arm`` (n,) bool
    the rows whose segment started this step. Rows are selected by masks, not indices (``torch.where``).
    """

    def __init__(self, dt: float, mean_hold_s: float = 6.0, clip_prob: float = 0.25,
                 velocity_law: VelocityLaw | None = None, arm_law: ArmGoalLaw | None = None):
        self.dt, self.mean_hold_s, self.clip_prob = dt, mean_hold_s, clip_prob
        self.end_prob = dt / mean_hold_s
        self.velocity_law, self.arm_law = velocity_law, arm_law
        tau = velocity_law.filter_tau if velocity_law is not None else 0.0
        # ModalVelocityCommand: alpha = 1 - exp(-step_dt / command_filter_tau), 1 without the filter (commands.py:172)
        self.filter_alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0

    def start(self, recorded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The state of an imagined rollout from a buffered env state: the commands the env held there
        (``velocity``, its low-passed command; ``arm``; ``velocity_target`` the drawn one, default ``velocity``), on
        a law segment."""
        velocity = recorded["velocity"].clone()
        n, device = velocity.shape[0], velocity.device
        no = torch.zeros(n, dtype=torch.bool, device=device)
        target = recorded.get("velocity_target", velocity)
        return {"velocity": velocity, "velocity_target": target.clone(), "arm": recorded["arm"].clone(),
                "clip_velocity": no, "clip_arm": no.clone(), "new_velocity": no.clone(), "new_arm": no.clone()}

    def step(self, state: dict[str, torch.Tensor], gen: torch.Generator) -> dict[str, torch.Tensor]:
        """One control step: each command's segment ends with probability ``dt / mean_hold_s`` and a new one starts,
        a clip segment with probability ``clip_prob`` (its command left for ``refresh``) else a law draw; then the
        seen velocity moves toward the drawn one (main resamples, then low-passes, in one step)."""
        n = state["velocity"].shape[0]
        out = dict(state)
        for key, law in (("velocity", self.velocity_law), ("arm", self.arm_law)):
            target_key = "velocity_target" if key == "velocity" else "arm"
            new = _rand(gen, n) < self.end_prob
            clip = _rand(gen, n) < self.clip_prob
            out[f"new_{key}"] = new
            out[f"clip_{key}"] = torch.where(new, clip, state[f"clip_{key}"])
            if law is not None:
                drawn = law.sample(n, gen).to(state[target_key])
                mask = (new & ~clip).view(n, *([1] * (drawn.dim() - 1)))
                out[target_key] = torch.where(mask, drawn, state[target_key])
        out["velocity"] = state["velocity"] + self.filter_alpha * (out["velocity_target"] - state["velocity"])
        return out

    @staticmethod
    def refresh(state: dict[str, torch.Tensor], clip_velocity: torch.Tensor | None = None,
                clip_arm: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Set the clip rows' drawn commands to the clip's own (the caller's clip tables at the rollout's frames)."""
        out = dict(state)
        if clip_velocity is not None:
            out["velocity_target"] = torch.where(state["clip_velocity"][:, None], clip_velocity,
                                                 state["velocity_target"])
        if clip_arm is not None:
            out["arm"] = torch.where(state["clip_arm"][:, None, None], clip_arm, state["arm"])
        return out
