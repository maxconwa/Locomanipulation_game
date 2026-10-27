"""Reward terms for the two-agent task.

Both agents' actions live in one ActionManager vector, and self-contact pairs involve both agents' links, so those
terms here are restricted to one agent.
"""

from __future__ import annotations

import numpy as np
import re
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import base_height_l2
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ..golem_safety import GOLEM_LIMITS
from .commands import ground_height
from .legs_rewards import STANCE_THRESHOLD, joint_deviation_l2, leg_phase, standing_command

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_term_rate_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """action_rate_l2 restricted to one action term's slice of the action vector."""
    manager = env.action_manager
    start = 0
    for name, dim in zip(manager.active_terms, manager.action_term_dim):
        if name == action_name:
            break
        start += dim
    a = manager.action[:, start : start + dim]
    prev = manager.prev_action[:, start : start + dim]
    return torch.sum(torch.square(a - prev), dim=1)


def action_beyond_clip(env, agent: str, action_name: str, clip: float) -> torch.Tensor:
    """How far the agent's raw actions for action_name's joints are beyond the env's clip, plus what that action
    term's position bound cut off, summed over its joints (action units). Nothing else pulls a policy mean back once
    it drifts past the clip."""
    start = 0
    for name in env.cfg.agent_action_terms[agent]:
        dim = env.action_manager.get_term(name).action_dim
        if name == action_name:
            break
        start += dim
    excess = (env.actions[agent][:, start : start + dim].abs() - clip).clamp(min=0.0).sum(dim=1)
    return excess + env.action_manager.get_term(action_name).beyond_bounds.sum(dim=1)


class self_contacts_involving(ManagerTermBase):
    """Number of self-contact pairs above `threshold` that include at least one of `own_links`.

    Reads the per-link filtered sensors that TerrainSceneCfg builds: sensor i belongs to sensor_link_names[i] and
    its force-matrix column j is filter_link_names[i + 1 + j] (each pair appears once).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        own = set(cfg.params["own_links"])
        filter_links = cfg.params["filter_link_names"]
        self._sensors: list[ContactSensor] = []
        self._masks: list[torch.Tensor] = []
        for i, (name, link) in enumerate(zip(cfg.params["sensor_names"], cfg.params["sensor_link_names"])):
            mask = torch.tensor([link in own or other in own for other in filter_links[i + 1 :]], device=env.device)
            if mask.any():
                self._sensors.append(env.scene.sensors[name])
                self._masks.append(mask)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_names: list[str],
        sensor_link_names: list[str],
        filter_link_names: list[str],
        own_links: list[str],
        threshold: float = 0.1,
    ) -> torch.Tensor:
        count = torch.zeros(env.num_envs, device=env.device)
        for sensor, mask in zip(self._sensors, self._masks):
            # (N, T, 1, M, 3) -> peak over history, per filter link -> (N, M)
            peak = sensor.data.force_matrix_w_history.norm(dim=-1).max(dim=1)[0][:, 0]
            count += torch.sum((peak > threshold) & mask, dim=1)
        return count


class _CycleMean(ManagerTermBase):
    """Base for terms that read a quantity's mean over each env's last gait cycle (the env cfg's gait_period): within
    a stride the pelvis sways, surges and twists about as much as a small command asks for, so only the cycle mean
    shows the motion the command asks for."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv, dims: int):
        super().__init__(cfg, env)
        self._window = max(int(round(env.cfg.gait_period / env.step_dt)), 1)
        self._samples = torch.zeros(env.num_envs, self._window, dims, device=env.device)
        self._count = torch.zeros(env.num_envs, device=env.device)
        self._slot = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._samples[ids] = 0.0
        self._count[ids] = 0.0

    def _push(self, x: torch.Tensor) -> torch.Tensor:
        """Add this step's (num_envs, dims) sample; returns the mean over the window so far."""
        self._samples[:, self._slot] = x
        self._slot = (self._slot + 1) % self._window
        self._count = torch.clamp(self._count + 1.0, max=self._window)
        return self._samples.sum(dim=1) / self._count.unsqueeze(1)


class track_velocity_cycle_exp(_CycleMean):
    """exp(-err^2 / std^2) of the velocity command against the pelvis velocity's gait-cycle mean. component "lin_xy":
    (vx, vy) in the pelvis frame, std the command's lin_vel_std; "ang_z": the yaw rate, std its ang_vel_std (the
    command's tracking-kernel curriculum)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env, 2 if cfg.params["component"] == "lin_xy" else 1)

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, component: str) -> torch.Tensor:
        data, term = env.scene["robot"].data, env.command_manager.get_term(command_name)
        if component == "lin_xy":
            velocity, command, std = data.root_lin_vel_b[:, :2], term.command[:, :2], term.lin_vel_std
        else:
            velocity, command, std = data.root_ang_vel_b[:, 2:3], term.command[:, 2:3], term.ang_vel_std
        mean = self._push(velocity)
        return torch.exp(-torch.sum(torch.square(command - mean), dim=1) / std**2)


class yaw_twist_l2(_CycleMean):
    """Squared gap between the pelvis yaw rate and its gait-cycle mean (rad^2/s^2): the twist within a stride, which
    tracking the cycle-mean yaw rate leaves free (a policy without it twisted at 1.2 rad/s RMS)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env, 1)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2:3]
        return torch.square(yaw_rate - self._push(yaw_rate)).squeeze(1)


class joint_mean_deviation_l2(_CycleMean):
    """Squared gap between asset_cfg's joint angles' gait-cycle mean and their defaults, summed over the joints,
    outside arm goals (rad^2; during them times arm_goal_scale, 0 by default). Standing it holds the default pose
    (sport mode's stance); walking, the cycle mean, so the swing itself is free: sport mode's cycle mean while walking
    sits within 0.04 rad of its stance. Without it the stance sank from a 0.9 to a 1.2 rad knee once arm goals started,
    base_height on in navigation notwithstanding: the pelvis height alone let the crouch through on bent ankles."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._ids = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names, preserve_order=True)[0]
        super().__init__(cfg, env, len(self._ids))

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, arm_command_name: str,
                 arm_goal_scale: float = 0.0) -> torch.Tensor:
        data = env.scene[asset_cfg.name].data
        mean = self._push(data.joint_pos[:, self._ids] - data.default_joint_pos[:, self._ids])
        arm = env.command_manager.get_term(arm_command_name).arm_mode
        return torch.sum(torch.square(mean), dim=1) * torch.where(arm, arm_goal_scale, 1.0)


def arm_target_pos_exp(env: ManagerBasedRLEnv, command_name: str, arm: int, std: float) -> torch.Tensor:
    """exp(-d^2 / std^2) on one arm's wrist position error."""
    pos_error, _ = env.command_manager.get_term(command_name).errors()
    return torch.exp(-pos_error[:, arm].square() / std**2)


def arm_target_pos_exp_annealed(env: ManagerBasedRLEnv, command_name: str, arm: int) -> torch.Tensor:
    """arm_target_pos_exp with the command's current fine_pos_std (its tracking-kernel curriculum)."""
    return arm_target_pos_exp(env, command_name, arm, env.command_manager.get_term(command_name).fine_pos_std)


def arm_target_quat_exp_annealed(env: ManagerBasedRLEnv, command_name: str, arm: int) -> torch.Tensor:
    """exp(-theta^2 / std^2) on one arm's wrist rotation error (rad), std the command's current rot_std (its
    tracking-kernel curriculum)."""
    term = env.command_manager.get_term(command_name)
    _, rot_error = term.errors()
    return torch.exp(-rot_error[:, arm].square() / term.rot_std**2)


def base_height_l2_navigation(
    env: ManagerBasedRLEnv, target_height: float, arm_command_name: str, sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Isaac Lab's base_height_l2, outside arm goals only: their crouch is free, a crouch while walking is not."""
    arm = env.command_manager.get_term(arm_command_name).arm_mode
    return base_height_l2(env, target_height, sensor_cfg=sensor_cfg) * (~arm).float()


def arm_goal_reached(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 on the step after both wrists were first held on the goal: a one-off bonus, so finishing pays more than
    hovering just outside the tolerance."""
    return env.command_manager.get_term(command_name).just_reached.float()


def yaw_rate_l2_during_arm_goal(env: ManagerBasedRLEnv, arm_command_name: str) -> torch.Tensor:
    """Squared pelvis yaw rate during an arm goal: turning moves the world-fixed targets in the pelvis frame."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate) * env.command_manager.get_term(arm_command_name).arm_mode


class sport_gait_exp(_CycleMean):
    """exp(-|s - s_ref|^2 / std^2) over asset_cfg's joints (the legs, in LOWER_JOINT_NAMES's order), outside arm goals:
    sport mode's swing. s is each joint's angle less its gait-cycle mean; s_ref is sport mode's at this gait phase
    (leg_phase's left leg) and velocity command (clamped to the bags' range), as scripts/fit_sport_gait.py fitted it to
    the sport-mode bags (reference_file) less its cycle mean, and zero while standing is commanded. The velocity
    rewards alone left the turn to the action noise: the policy held both hip yaws 0.3 rad into the turn where sport
    mode swings them +-0.1 rad about zero with each step, and its knees swung a quarter of sport mode's. The swing
    alone, not the pose: on the whole pose (std 0.3) standing still in the stance earned 64% of the term, and a policy
    stood through 66% of its commands. The stance is posture's, hip_yaw_mean's and hip_roll_pos's. std narrows linearly
    to std_end by std_steps env steps (std_end None: fixed): at a fixed 0.1 a policy swinging a quarter of sport mode's
    amplitude earned almost nothing more than standing, and its swing never grew."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._ids = env.scene[asset_cfg.name].find_joints(asset_cfg.joint_names, preserve_order=True)[0]
        super().__init__(cfg, env, len(self._ids))
        ref = np.load(cfg.params["reference_file"])
        self._harmonics = int(ref["harmonics"])
        # (features, basis, joints) without the constant basis function: the fit's cycle mean dropped
        coef = torch.tensor(ref["coef"], device=env.device).view(-1, 1 + 2 * self._harmonics, len(self._ids))
        self._coef = coef[:, 1:].flatten(0, 1)
        self._low = torch.tensor(ref["v_low"], device=env.device)
        self._high = torch.tensor(ref["v_high"], device=env.device)

    def swing(self, env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        """Sport mode's joint angles less their cycle mean, for each env's phase and command, (num_envs, joints)."""
        cmd = env.command_manager.get_term(command_name).command[:, :3]
        v = torch.maximum(torch.minimum(cmd, self._high), self._low)
        features = torch.cat([torch.ones_like(v[:, :1]), v, v.abs()], dim=1)
        angle = 2.0 * torch.pi * leg_phase(env, command_name)[:, :1]
        basis = torch.cat([f(h * angle) for h in range(1, self._harmonics + 1) for f in (torch.cos, torch.sin)], dim=1)
        x = (features.unsqueeze(2) * basis.unsqueeze(1)).flatten(1)
        return torch.where(standing_command(env, cmd).unsqueeze(1), 0.0, x @ self._coef)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, arm_command_name: str,
                 std: float, reference_file: str, std_end: float | None = None, std_steps: int = 0) -> torch.Tensor:
        q = env.scene[asset_cfg.name].data.joint_pos[:, self._ids]
        swing = q - self._push(q)
        error = torch.sum(torch.square(swing - self.swing(env, command_name)), dim=1)
        if std_end is not None:
            std += (std_end - std) * min(env.common_step_counter / max(std_steps, 1), 1.0)
        arm = env.command_manager.get_term(arm_command_name).arm_mode
        return torch.exp(-error / std**2) * (~arm).float()


def joint_deviation_l2_modal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, arm_command_name: str, arm_goal_scale: float,
    command_name: str = "base_velocity", turn_free_joint_names: list[str] | None = None
) -> torch.Tensor:
    """joint_deviation_l2, times arm_goal_scale during arm goals. Of asset_cfg's joints, turn_free_joint_names
    (regexes) cost nothing while command_name commands a turn (|w_z| at least its min_cos_speed)."""
    arm_mode = env.command_manager.get_term(arm_command_name).arm_mode
    scale = torch.where(arm_mode, arm_goal_scale, 1.0)
    if not turn_free_joint_names:
        return joint_deviation_l2(env, asset_cfg) * scale
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids if isinstance(asset_cfg.joint_ids, list) else list(range(asset.num_joints))
    free_ids = set(asset.find_joints(turn_free_joint_names)[0])
    free = torch.tensor([j in free_ids for j in ids], device=env.device)
    term = env.command_manager.get_term(command_name)
    turning = torch.abs(term.command[:, 2]) >= term.cfg.min_cos_speed
    dev = torch.square(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    dev = torch.where(turning.unsqueeze(1) & free, 0.0, dev)
    return torch.sum(dev, dim=1) * scale


def knee_valgus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, margin: float = 0.0) -> torch.Tensor:
    """How far each knee sits medial of its hip-ankle line beyond margin, summed over the legs (m).

    Seen from the front in the pelvis's heading frame (yaw only, z up): the y of the line from the hip to the ankle at
    the knee's height, minus the knee's y, toward the midline. It is 0 at the default pose, where hip, knee and ankle
    share one y, and grows when the knees fall inward, as in a crouch on hip yaw turned in. Knees bowed outward
    cost nothing. asset_cfg's bodies are the left hip, knee and ankle, then the right (preserve_order).
    """
    asset = env.scene[asset_cfg.name]
    pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (n, 6, 3)
    n, k = pos_w.shape[:2]
    quat = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
    rel = quat_apply_inverse(quat, (pos_w - asset.data.root_pos_w.unsqueeze(1)).reshape(-1, 3)).view(n, 2, 3, 3)
    hip, knee, ankle = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]  # (n, leg, xyz)
    # the ankle is below the hip; the clamp only guards the division
    s = (knee[..., 2] - hip[..., 2]) / (ankle[..., 2] - hip[..., 2]).clamp(max=-0.05)
    line_y = hip[..., 1] + s * (ankle[..., 1] - hip[..., 1])
    medial = (line_y - knee[..., 1]) * torch.tensor([1.0, -1.0], device=pos_w.device)  # the right leg's midline is +y
    return (medial - margin).clamp(min=0.0).sum(dim=1)


def feet_swing_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rest_height: float,
    lift_height: float,
    command_name: str = "base_velocity",
    terrain_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """How far each foot in its swing phase is below a swing reference, summed over the feet (m).

    The gait clock (leg_phase, which the legs observe) puts a foot in swing for phase >= STANCE_THRESHOLD while a
    walk is commanded. The reference rises from rest_height to rest_height + lift_height and back,
    rest_height + lift_height * sin(pi * s) over the swing's progress s in [0, 1]. A foot pays for being below it
    whether it touches the ground or not, so dragging it through the swing costs. asset_cfg's bodies are the feet in
    leg_phase's order (left, right).
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfg is not None:
        foot_z = foot_z - ground_height(env.scene.sensors[terrain_sensor_cfg.name]).unsqueeze(1)
    phase = leg_phase(env, command_name)
    swing = phase >= STANCE_THRESHOLD
    progress = ((phase - STANCE_THRESHOLD) / (1.0 - STANCE_THRESHOLD)).clamp(0.0, 1.0)
    reference = rest_height + lift_height * torch.sin(torch.pi * progress)
    return torch.sum((reference - foot_z).clamp(min=0.0) * swing, dim=1)


def feet_swing_height_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rest_height: float,
    lift_height: float,
    command_name: str = "base_velocity",
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.03,
) -> torch.Tensor:
    """exp(-shortfall^2 / std^2) per foot in its swing phase, summed over the feet; stance feet give 0.

    The positive counterpart of feet_swing_clearance, on its gait clock, reference and terrain correction: shortfall is
    how far the foot is below the swing reference (m), overshoot free. The legs reward is floored at 0 per step, and on
    the steps under the floor (half of a shuffling policy's walking steps) the clearance penalty gives no gradient; a
    positive term lifts those steps' total and pays for a lift exactly there.
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfg is not None:
        foot_z = foot_z - ground_height(env.scene.sensors[terrain_sensor_cfg.name]).unsqueeze(1)
    phase = leg_phase(env, command_name)
    swing = phase >= STANCE_THRESHOLD
    progress = ((phase - STANCE_THRESHOLD) / (1.0 - STANCE_THRESHOLD)).clamp(0.0, 1.0)
    reference = rest_height + lift_height * torch.sin(torch.pi * progress)
    shortfall = (reference - foot_z).clamp(min=0.0)
    return torch.sum(torch.exp(-torch.square(shortfall) / std**2) * swing, dim=1)


def feet_swing_height_tracking_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    rest_height: float,
    lift_height: float,
    command_name: str = "base_velocity",
    terrain_sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.03,
) -> torch.Tensor:
    """sin(pi * s) * exp(-(foot_z - reference)^2 / std^2) per foot in its swing phase, summed over the feet; stance
    feet give 0.

    feet_swing_clearance's gait clock, reference and terrain correction, the error two-sided (the foot goes to the
    reference's height, not past it) and weighted by the reference's own lift share sin(pi * s): at the swing's ends
    the reference sits at rest_height, where a dragged foot matches it and feet_swing_height_exp paid it 0.31 of a
    full lift. Weighted, a dragged foot earns 0.15 of a full lift at std 0.03 (half the lift height: 0.54).
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfg is not None:
        foot_z = foot_z - ground_height(env.scene.sensors[terrain_sensor_cfg.name]).unsqueeze(1)
    phase = leg_phase(env, command_name)
    swing = phase >= STANCE_THRESHOLD
    progress = ((phase - STANCE_THRESHOLD) / (1.0 - STANCE_THRESHOLD)).clamp(0.0, 1.0)
    lift_share = torch.sin(torch.pi * progress)
    reference = rest_height + lift_height * lift_share
    kernel = torch.exp(-torch.square(foot_z - reference) / std**2)
    return torch.sum(lift_share * kernel * swing, dim=1)


class golem_limits(ManagerTermBase):
    """The measured joint state against GOLEM's e-stop thresholds (golem_safety.GOLEM_LIMITS, relax_safety_split),
    summed over asset_cfg's joints. The target clip bounds only the targets: a joint pushed past its target by the
    robot's weight still e-stops the robot.

    quantity:
    - "position": how far each joint is inside `margin` (rad) of its position thresholds (rad); `joint_margins`
      (joint-name regex -> rad) overrides the margin for the joints it matches.
    - "velocity", "torque": how far |dq| or the motor's PD torque is beyond `margin` x the threshold, as a share of
      the threshold.
    - "velocity_l2", "torque_l2": (dq / threshold)^2 or (tau / threshold)^2; margin unused.

    Torque is the motor's PD demand at the measured state with the motor's own damping, as GolemEstopMonitor computes
    it (GOLEM reads the motor's torque estimate).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        self._ids = list(range(self._asset.num_joints)) if isinstance(ids, slice) else list(ids)
        rows = {row[0]: row for row in GOLEM_LIMITS}
        names = [self._asset.joint_names[i] for i in self._ids]
        missing = [n for n in names if n not in rows]
        if missing:
            raise ValueError(f"golem_limits: no GOLEM limits for {missing}")
        # (motor, low, high, velocity, torque, clip offset, estop offset, velocity ratio, torque ratio)
        table = torch.tensor([rows[n][1:] for n in names], device=env.device)
        low, high, vel, tau, _, q_off, vel_ratio, tau_ratio = table.T
        self._q_low, self._q_high = low + q_off, high - q_off
        self._dq_max, self._tau_max = vel * vel_ratio, tau * tau_ratio
        overrides = cfg.params.get("joint_margins") or {}
        margins = []
        for name in names:
            matched = [m for pattern, m in overrides.items() if re.fullmatch(pattern, name)]
            margins.append(matched[0] if matched else cfg.params.get("margin", 0.0))
        self._q_margin = torch.tensor(margins, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, quantity: str, margin: float = 0.0,
                 joint_margins: dict[str, float] | None = None):
        data, ids = self._asset.data, self._ids
        if quantity == "position":
            q, m = data.joint_pos[:, ids], self._q_margin
            inside = (self._q_low + m - q).clamp(min=0.0) + (q - (self._q_high - m)).clamp(min=0.0)
            return inside.sum(dim=1)
        if quantity.startswith("velocity"):
            ratio = data.joint_vel[:, ids].abs() / self._dq_max
        else:
            error = data.joint_pos_target[:, ids] - data.joint_pos[:, ids]
            torque = data.default_joint_stiffness[:, ids] * error - data.default_joint_damping[:, ids] * data.joint_vel[:, ids]
            ratio = torque.abs() / self._tau_max
        if quantity.endswith("_l2"):
            return ratio.square().sum(dim=1)
        return (ratio - margin).clamp(min=0.0).sum(dim=1)


class arm_joint_target_rate_l2(ManagerTermBase):
    """Squared change of the arm joints' position targets since the last policy step, summed (rad^2), during arm goals.

    The arms' target is the measured position plus the IK step plus the residual, so this charges the dithering the
    action_rate term on the residual alone misses: an earlier policy's arms oscillated at about 3.2 Hz while reaching
    (Isaac and the real robot), their targets moving 1.5-2.5 rad/s against 0.35 while walking. Squared, it charges a
    smooth reach (a few joints at ~1 rad/s) far less than that. Zero on an episode's first step.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ids = env.scene[cfg.params["asset_cfg"].name].find_joints(cfg.params["asset_cfg"].joint_names, preserve_order=True)[0]
        self._prev = torch.zeros(env.num_envs, len(self._ids), device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, arm_command_name: str) -> torch.Tensor:
        target = env.scene[asset_cfg.name].data.joint_pos_target[:, self._ids]
        rate = torch.sum(torch.square(target - self._prev), dim=1)
        rate = torch.where(self._fresh, torch.zeros_like(rate), rate)
        self._prev[:] = target
        self._fresh[:] = False
        arm = env.command_manager.get_term(arm_command_name).arm_mode
        return rate * arm.float()
