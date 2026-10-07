"""GOLEM's safety-layer e-stops, checked in the sim at every physics substep.

On the robot, GOLEM's h12_safety_layer (preset relax_safety_split, the one the real launch uses) clips each position
target into the URDF range shrunk by 0.001 rad, and e-stops the robot when the measured state leaves the e-stop range,
polled at 500 Hz: a joint within its position offset of a URDF limit, |dq| above the URDF velocity limit times its
ratio, or |tau_est| above the URDF torque limit times its ratio. GolemEstopMonitor applies those thresholds,
tightened by GolemEstopCfg's margins, for the golem_estop termination. Torque is the motor's PD demand at the current
state. The table is copied from GOLEM (core/joint_limits.py and config/relax_safety_split.yaml at commit 462c479).
"""

from __future__ import annotations

import re

import torch

from isaaclab.utils import configclass

GOLEM_PRESET = "relax_safety_split"

# (motor, URDF low, URDF high, URDF velocity, URDF torque, clip position_offset,
#  estop position_offset, estop velocity_ratio, estop torque_ratio); Unitree LowCmd motor order
GOLEM_LIMITS = [
    ("left_hip_yaw_joint", -0.43, 0.43, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_hip_pitch_joint", -3.14, 2.5, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_hip_roll_joint", -0.43, 3.14, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_knee_joint", -0.12, 2.19, 14.0, 300.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_ankle_pitch_joint", -0.897334, 0.523598, 9.0, 60.0, 0.001, 0.0001, 3.0, 1.0),
    ("left_ankle_roll_joint", -0.261799, 0.261799, 9.0, 40.0, 0.001, 0.0001, 3.0, 1.0),
    ("right_hip_yaw_joint", -0.43, 0.43, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_hip_pitch_joint", -3.14, 2.5, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_hip_roll_joint", -3.14, 0.43, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_knee_joint", -0.12, 2.19, 14.0, 300.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_ankle_pitch_joint", -0.897334, 0.523598, 9.0, 60.0, 0.001, 0.0001, 3.0, 1.0),
    ("right_ankle_roll_joint", -0.261799, 0.261799, 9.0, 40.0, 0.001, 0.0001, 3.0, 1.0),
    ("torso_joint", -2.35, 2.35, 23.0, 200.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_shoulder_pitch_joint", -3.14, 1.57, 9.0, 40.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_shoulder_roll_joint", -0.38, 3.4, 9.0, 40.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_shoulder_yaw_joint", -2.66, 3.01, 20.0, 18.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_elbow_joint", -0.95, 3.18, 20.0, 18.0, 0.001, 0.0001, 1.0, 1.2),
    ("left_wrist_roll_joint", -3.01, 2.75, 31.4, 19.0, 0.001, 0.0001, 1.0, 1.0),
    ("left_wrist_pitch_joint", -0.4625, 0.4625, 31.4, 19.0, 0.001, -0.1, 1.0, 2.0),
    ("left_wrist_yaw_joint", -1.27, 1.27, 31.4, 19.0, 0.001, -0.1, 1.0, 2.0),
    ("right_shoulder_pitch_joint", -3.14, 1.57, 9.0, 40.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_shoulder_roll_joint", -3.4, 0.38, 9.0, 40.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_shoulder_yaw_joint", -3.01, 2.66, 20.0, 18.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_elbow_joint", -0.95, 3.18, 20.0, 18.0, 0.001, 0.0001, 1.0, 1.2),
    ("right_wrist_roll_joint", -2.75, 3.01, 31.4, 19.0, 0.001, 0.0001, 1.0, 1.0),
    ("right_wrist_pitch_joint", -0.4625, 0.4625, 31.4, 19.0, 0.001, -0.1, 1.0, 2.0),
    ("right_wrist_yaw_joint", -1.27, 1.27, 31.4, 19.0, 0.001, -0.1, 1.0, 2.0),
]
GOLEM_JOINT_NAMES = [row[0] for row in GOLEM_LIMITS]
GOLEM_TARGET_CLIP = 0.001
"""GOLEM's clip.position_offset (every joint): position targets are clipped to the URDF range shrunk by this."""

CAUSES = ("velocity", "position", "torque")


@configclass
class GolemEstopCfg:
    """Sim thresholds: GOLEM's e-stops tightened for sim-to-real error (encoder noise, GOLEM's 500 Hz polling
    against the 200 Hz physics, the torque estimate)."""

    velocity_margin: float = 0.9
    """Trip at this fraction of GOLEM's velocity e-stop."""
    torque_margin: float = 0.9
    """Trip at this fraction of GOLEM's torque e-stop."""
    position_margin: float = 0.02
    """Trip this far (rad) inside GOLEM's position e-stop."""
    joint_position_margins: dict[str, tuple[float, float]] = {}
    """Per-joint overrides of position_margin: joint-name regex -> (margin at the lower limit, at the upper)."""


class GolemEstopMonitor:
    """Per-env, per-cause e-stop flags, ORed over the physics substeps of a policy step."""

    def __init__(self, robot, cfg: GolemEstopCfg, num_envs: int, device: str):
        self.robot = robot
        self.cfg = cfg
        self.joint_ids = robot.find_joints(GOLEM_JOINT_NAMES, preserve_order=True)[0]
        table = torch.tensor([row[1:] for row in GOLEM_LIMITS], device=device)
        low, high, vel, tau, _, q_off, vel_ratio, tau_ratio = table.T
        margin_low = torch.full_like(low, cfg.position_margin)
        margin_high = torch.full_like(high, cfg.position_margin)
        for pattern, (m_low, m_high) in cfg.joint_position_margins.items():
            hits = [i for i, n in enumerate(GOLEM_JOINT_NAMES) if re.fullmatch(pattern, n)]
            if not hits:
                raise ValueError(f"joint_position_margins: {pattern!r} matches no GOLEM joint")
            margin_low[hits], margin_high[hits] = m_low, m_high
        self.q_low = low + q_off + margin_low
        self.q_high = high - q_off - margin_high
        self.dq_max = vel * vel_ratio * cfg.velocity_margin
        self.tau_max = tau * tau_ratio * cfg.torque_margin
        self.flags = {c: torch.zeros(num_envs, dtype=torch.bool, device=device) for c in CAUSES}
        self.joint_hits = {c: torch.zeros(len(GOLEM_LIMITS), device=device) for c in CAUSES}
        self._skip_torque = torch.ones(num_envs, dtype=torch.bool, device=device)
        print(f"[INFO] GolemEstopMonitor ({GOLEM_PRESET}, velocity x{cfg.velocity_margin}, torque x{cfg.torque_margin},"
              f" position {cfg.position_margin} rad inside"
              f"{f', {cfg.joint_position_margins}' if cfg.joint_position_margins else ''}): " + ", ".join(
                  f"{n} q [{lo:.3f}, {hi:.3f}] dq {dq:.1f} tau {t:.1f}" for n, lo, hi, dq, t in zip(
                      GOLEM_JOINT_NAMES, self.q_low.tolist(), self.q_high.tolist(), self.dq_max.tolist(), self.tau_max.tolist())))

    def update(self):
        """Check the current state (the robot's data after the last substep) and OR it into the flags."""
        data = self.robot.data
        q = data.joint_pos[:, self.joint_ids]
        bad = {
            "position": (q < self.q_low) | (q > self.q_high),
            "velocity": data.joint_vel[:, self.joint_ids].abs() > self.dq_max,
            # computed_torque is from the last write before this state; stale right after a reset
            "torque": (self._motor_torque().abs() > self.tau_max) & ~self._skip_torque.unsqueeze(1),
        }
        self._skip_torque[:] = False
        for cause, joints in bad.items():
            hit = joints.any(dim=1)
            self.joint_hits[cause] += (joints & (hit & ~self.flags[cause]).unsqueeze(1)).float().sum(dim=0)
            self.flags[cause] |= hit

    def _motor_torque(self) -> torch.Tensor:
        """The motor's PD demand at the current state, with the motor's own damping (not the randomized passive part,
        which the motor doesn't produce)."""
        data, ids = self.robot.data, self.joint_ids
        error = data.joint_pos_target[:, ids] - data.joint_pos[:, ids]
        return data.joint_stiffness[:, ids] * error - data.default_joint_damping[:, ids] * data.joint_vel[:, ids]

    def tripped(self) -> torch.Tensor:
        return self.flags["velocity"] | self.flags["position"] | self.flags["torque"]

    def clear(self):
        for flag in self.flags.values():
            flag.zero_()

    def reset(self, env_ids):
        for flag in self.flags.values():
            flag[env_ids] = False
        self._skip_torque[env_ids] = True

    def log(self) -> dict[str, torch.Tensor]:
        """Share of envs tripped this step by each cause, and the trip counts since the last log by joint group."""
        out = {f"Safety/estop_{c}": self.flags[c].float().mean() for c in CAUSES}
        for c in CAUSES:
            hits = self.joint_hits[c]
            for group, prefix in (("legs", ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")),
                                  ("torso", ("torso",)), ("arms", ("left_s", "right_s", "left_e", "right_e", "left_w", "right_w"))):
                mask = torch.tensor([n.startswith(prefix) for n in GOLEM_JOINT_NAMES], device=hits.device)
                out[f"Safety/{c}_trips_{group}"] = hits[mask].sum()
            hits.zero_()
        return out
