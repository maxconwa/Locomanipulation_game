"""Each agent's goal as a world model sees it: what the task asks for, exogenous to the robot's state.

A goal is two things. Its state (goal_state, (N, Gs)) is what the env holds: for the legs the sampled and the filtered
velocity command, the episode's time and the reward kernels' current stds; for the arms the believed wrist targets,
whether they are an arm goal and the time left in it. Its features (goal_features, (N, Gf)) are what the networks see:
the state with the time turned into each leg's gait phase exactly as the policies observe it (mdp.gait_phase_sin /
gait_phase_cos). advance_goal steps the state one policy step as the env would between its events (the command held,
its low-pass stepped, the clocks advanced, the stds held), so a rollout in imagination drives the goal without the
simulator; resamples, resets and the curriculum's stage changes are the env's alone.

Pure torch, no Isaac Lab import: goal_state and goal_spec read the env's attributes, goal_features and advance_goal
only tensors and a GoalSpec, so they run inside imagination and in unit tests. Both are differentiable and never
write into their inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# the commands' names in MarlCommandsCfg (locomanip_marl_env_cfg.py; ARM_COMMAND there)
VELOCITY_COMMAND = "base_velocity"
ARM_COMMAND = "arm_targets"
# s: the world state's episode time is clipped here (MarlObservationsCfg.world's episode_time)
EPISODE_TIME_CLIP_S = 6.0

# name -> size, in order
LEGS_GOAL_STATE = {"command": 3, "command_filtered": 3, "time": 1, "kernel_stds": 2, "sport_gait_std": 1}
LEGS_GOAL_FEATURES = {"command": 3, "command_filtered": 3, "gait_phase_sin": 2, "gait_phase_cos": 2, "stds": 3}
ARMS_GOAL_STATE = {"wrist_targets": 14, "arm_mode": 1, "time_left": 1}
ARMS_GOAL_FEATURES = dict(ARMS_GOAL_STATE)


@dataclass(frozen=True)
class GoalSpec:
    """The constants goal_features and advance_goal need, read once from the env (goal_spec)."""

    agent: str
    step_dt: float
    """s per policy step."""
    gait_period: float = 0.7
    """s per stride of the gait clock (cfg.gait_period)."""
    zero_cmd_eps: float = 0.03
    """The stand/walk gate on the filtered command's norm (cfg.zero_cmd_eps)."""
    filter_alpha: float = 1.0
    """The command low-pass's step (ModalVelocityCommand._filter_alpha); 1: none (the command is the sampled one)."""
    time_clip: float = EPISODE_TIME_CLIP_S
    """s: the world state's episode-time clip (not used by the goal, which keeps the full time for the gait clock)."""

    @property
    def state_layout(self) -> dict[str, int]:
        return LEGS_GOAL_STATE if self.agent == "legs" else ARMS_GOAL_STATE

    @property
    def feature_layout(self) -> dict[str, int]:
        return LEGS_GOAL_FEATURES if self.agent == "legs" else ARMS_GOAL_FEATURES

    @property
    def state_dim(self) -> int:
        return sum(self.state_layout.values())

    @property
    def feature_dim(self) -> int:
        return sum(self.feature_layout.values())


def goal_spec(env, agent: str) -> GoalSpec:
    """The GoalSpec of agent ("legs" or "arms") on env (a LocoManipMarlEnv)."""
    if agent not in ("legs", "arms"):
        raise ValueError(f"No goal for agent {agent!r}")
    velocity = env.command_manager.get_term(VELOCITY_COMMAND)
    alpha = float(velocity._filter_alpha) if velocity.cfg.command_filter_tau > 0 else 1.0
    return GoalSpec(agent=agent, step_dt=float(env.step_dt), gait_period=float(env.cfg.gait_period),
                    zero_cmd_eps=float(env.cfg.zero_cmd_eps), filter_alpha=alpha)


def _sport_gait_std(env) -> float:
    """The legs' sport_gait reward's std at this step (mdp.sport_gait_exp: annealed with env.common_step_counter); 0
    where the term is absent or weighs 0."""
    manager = getattr(env, "reward_managers", {}).get("legs")
    if manager is None or "sport_gait" not in manager.active_terms:
        return 0.0
    cfg = manager.get_term_cfg("sport_gait")
    if cfg.weight == 0.0:
        return 0.0
    params = cfg.params
    std = float(params["std"])
    if params.get("std_end") is not None:
        std += (params["std_end"] - std) * min(env.common_step_counter / max(params.get("std_steps", 0), 1), 1.0)
    return std


def goal_state(env, agent: str) -> torch.Tensor:
    """agent's raw goal state now, (num_envs, Gs) (the module docstring; LEGS_GOAL_STATE / ARMS_GOAL_STATE).

    legs: the sampled velocity command, the filtered one the policies and rewards read (ModalVelocityCommand.command),
    the episode's time (s, as mdp.leg_phase computes it), the linear and yaw tracking kernels' stds and sport_gait's.
    arms: the believed wrist targets in the pelvis frame (ArmTargetsCommand.command: per arm x, y, z, qw, qx, qy, qz),
    the arm-goal flag and the time left on the arm command's timer (s)."""
    n, dev = env.num_envs, env.device
    if agent == "legs":
        velocity = env.command_manager.get_term(VELOCITY_COMMAND)
        time = (env.episode_length_buf * env.step_dt).unsqueeze(1)
        stds = torch.tensor([[velocity.lin_vel_std, velocity.ang_vel_std, _sport_gait_std(env)]], device=dev)
        return torch.cat([velocity.vel_command_b, velocity.command, time, stds.expand(n, 3)], dim=1)
    if agent == "arms":
        arm = env.command_manager.get_term(ARM_COMMAND)
        return torch.cat([arm.command, arm.arm_mode.float().unsqueeze(1), arm.time_left.unsqueeze(1)], dim=1)
    raise ValueError(f"No goal for agent {agent!r}")


def leg_phase(spec: GoalSpec, time: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    """mdp.leg_phase from the episode time (N,) and the filtered command (N, 3): each leg's phase in [0, 1), half a
    stride apart while a walk is commanded, both 0 while standing. (N, 2)."""
    standing = torch.norm(command[:, :3], dim=1) < spec.zero_cmd_eps
    period = spec.gait_period
    phase = torch.where(standing, torch.zeros_like(time, dtype=torch.float), (time % period) / period)
    offset = torch.where(standing, 0.0, 0.5)
    return torch.stack([phase, (phase + offset) % 1.0], dim=-1)


def goal_features(spec: GoalSpec, gstate: torch.Tensor) -> torch.Tensor:
    """What the networks see of a goal state, (N, Gf) (LEGS_GOAL_FEATURES / ARMS_GOAL_FEATURES). The legs': the two
    commands, each leg's gait phase as sin and cos (equal to the legs' gait_phase / gait_phase_cos observations), the
    three stds."""
    if spec.agent != "legs":
        return gstate
    command, filtered, time, stds = gstate[:, 0:3], gstate[:, 3:6], gstate[:, 6], gstate[:, 7:10]
    angle = 2.0 * math.pi * leg_phase(spec, time, filtered)
    return torch.cat([command, filtered, torch.sin(angle), torch.cos(angle), stds], dim=1)


def advance_goal(spec: GoalSpec, gstate: torch.Tensor) -> torch.Tensor:
    """The goal state one policy step later, between the env's events. legs: the sampled command held, the filtered
    one stepped once (ModalVelocityCommand._update_command's vel_command_filtered += alpha * (raw - filtered)), the
    time + step_dt, the stds held. arms: the time left - step_dt (CommandTerm.compute), the rest held. A new tensor."""
    if spec.agent == "legs":
        command, filtered = gstate[:, 0:3], gstate[:, 3:6]
        if spec.filter_alpha != 1.0:
            filtered = filtered + spec.filter_alpha * (command - filtered)
        else:
            filtered = command
        return torch.cat([command, filtered, gstate[:, 6:7] + spec.step_dt, gstate[:, 7:]], dim=1)
    return torch.cat([gstate[:, :-1], gstate[:, -1:] - spec.step_dt], dim=1)
