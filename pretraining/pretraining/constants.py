"""The H1-2's joints, gains, limits and tracked bodies of the pretraining, in Unitree motor order.

Transcribed from h12_train (this repository's own code, branch ``supermimic``, commit b175f1c):
``src/h12_train/robot/h12_constants.py`` (gains, effort limits, armature, action-scale rule, golem_safety clip and
estop derivation, keyframe) and ``src/h12_train/tasks.py`` (tracked bodies, anchor body). The joint ranges are
``robot/xmls/h1_2_magpie.xml``'s, which equal golem_safety's joint table. Pure Python: importable without Isaac.
"""

import re
from dataclasses import dataclass

# Unitree LowCmd motor order, equal to main's locomanipulation_game.assets.h1_2.ALL_JOINTS_NAMES.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LEG_JOINTS = slice(0, 12)  # the legs agent's joints
UPPER_JOINTS = slice(12, 27)  # the upper-body agent's: torso and arms

SIM_DT = 0.005  # s, physics step
DECIMATION = 4  # physics steps per policy step: 50 Hz control


def _per_joint(table: dict[str, float]) -> dict[str, float]:
    """Expand a {pattern: value} table (the first fully matching pattern applies) to {joint: value}."""
    return {j: next(v for p, v in table.items() if re.fullmatch(p, j)) for j in JOINT_NAMES}


# Reflected inertias, kg m^2, of Unitree's motor models (unitree_rl_lab).
_ARMATURE_M107_15 = 0.063259741
_ARMATURE_M107_24 = 0.160478022
_ARMATURE_N7520_14P3 = 0.01017752
_ARMATURE_N5020_16 = 0.003609725

# (kp N m/rad, kd N m s/rad, effort N m, armature kg m^2) per joint group. Legs: golem_lowerbody_rl almi.yaml; torso
# and arms: golem_upper_ik safety_split.yaml. Each ankle joint's armature is two N5020-class motors'.
_ACTUATORS: dict[str, tuple[float, float, float, float]] = {
    ".*_hip_(yaw|pitch|roll)_joint": (200.0, 2.5, 200.0, _ARMATURE_M107_15),
    ".*_knee_joint": (300.0, 4.0, 300.0, _ARMATURE_M107_24),
    ".*_ankle_pitch_joint": (40.0, 2.0, 60.0, 2 * _ARMATURE_N5020_16),
    ".*_ankle_roll_joint": (40.0, 2.0, 40.0, 2 * _ARMATURE_N5020_16),
    "torso_joint": (160.0, 10.0, 200.0, _ARMATURE_M107_15),
    ".*_shoulder_pitch_joint": (240.0, 12.0, 40.0, _ARMATURE_N7520_14P3),
    ".*_shoulder_roll_joint": (200.0, 12.0, 40.0, _ARMATURE_N7520_14P3),
    ".*_shoulder_yaw_joint": (150.0, 12.0, 18.0, _ARMATURE_N5020_16),
    ".*_elbow_joint": (150.0, 12.0, 18.0, _ARMATURE_N5020_16),
    ".*_wrist_.*_joint": (120.0, 12.0, 19.0, _ARMATURE_N5020_16),
}
ACTUATOR_GROUPS: dict[str, str] = {
    "hips": ".*_hip_(yaw|pitch|roll)_joint", "knees": ".*_knee_joint", "ankle_pitch": ".*_ankle_pitch_joint",
    "ankle_roll": ".*_ankle_roll_joint", "torso": "torso_joint", "shoulder_pitch": ".*_shoulder_pitch_joint",
    "shoulder_roll": ".*_shoulder_roll_joint", "shoulder_yaw": ".*_shoulder_yaw_joint", "elbows": ".*_elbow_joint",
    "wrists": ".*_wrist_.*_joint",
}
KP: dict[str, float] = _per_joint({p: v[0] for p, v in _ACTUATORS.items()})
KD: dict[str, float] = _per_joint({p: v[1] for p, v in _ACTUATORS.items()})
EFFORT: dict[str, float] = _per_joint({p: v[2] for p, v in _ACTUATORS.items()})
ARMATURE: dict[str, float] = _per_joint({p: v[3] for p, v in _ACTUATORS.items()})

# rad per unit action: mjlab's 0.25 * effort / kp, capped at 0.25 (the loco-manipulation game's leg scale).
_ACTION_SCALE_CAP = 0.25
ACTION_SCALE: dict[str, float] = {j: min(_ACTION_SCALE_CAP, _ACTION_SCALE_CAP * EFFORT[j] / KP[j]) for j in JOINT_NAMES}

# KNEES_BENT_KEYFRAME: knees bent, feet flat; the pelvis DEFAULT_ROOT_HEIGHT above the floor.
DEFAULT_JOINT_POS: dict[str, float] = _per_joint({
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_pitch_joint": -0.3,
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": 0.3,
    ".*": 0.0,
})
DEFAULT_ROOT_HEIGHT = 0.993  # m

# rad: the MJCF's joint ranges, equal to golem_safety's joint table (robots/h12.yaml).
JOINT_RANGES: dict[str, tuple[float, float]] = _per_joint({
    ".*_hip_yaw_joint": (-0.43, 0.43),
    ".*_hip_pitch_joint": (-3.14, 2.5),
    "left_hip_roll_joint": (-0.43, 3.14),
    "right_hip_roll_joint": (-3.14, 0.43),
    ".*_knee_joint": (-0.12, 2.19),
    ".*_ankle_pitch_joint": (-0.897334, 0.523598),
    ".*_ankle_roll_joint": (-0.261799, 0.261799),
    "torso_joint": (-2.35, 2.35),
    ".*_shoulder_pitch_joint": (-3.14, 1.57),
    "left_shoulder_roll_joint": (-0.38, 3.4),
    "right_shoulder_roll_joint": (-3.4, 0.38),
    "left_shoulder_yaw_joint": (-2.66, 3.01),
    "right_shoulder_yaw_joint": (-3.01, 2.66),
    ".*_elbow_joint": (-0.95, 3.18),
    "left_wrist_roll_joint": (-3.01, 2.75),
    "right_wrist_roll_joint": (-2.75, 3.01),
    ".*_wrist_pitch_joint": (-0.4625, 0.4625),
    ".*_wrist_yaw_joint": (-1.27, 1.27),
})

# golem_safety relax_safety_split limits.clip.position_offset, rad: every position target is clipped to the joint
# range shrunk by it.
_CLIP_POSITION_OFFSET = 0.001
TARGET_LIMITS: dict[str, tuple[float, float]] = {
    j: (lo + _CLIP_POSITION_OFFSET, hi - _CLIP_POSITION_OFFSET) for j, (lo, hi) in JOINT_RANGES.items()
}

# rad/s: golem_safety robots/h12.yaml's joint velocity limits.
_VELOCITY_LIMITS = _per_joint({
    ".*_hip_.*_joint": 23.0, ".*_knee_joint": 14.0, ".*_ankle_.*_joint": 9.0, "torso_joint": 23.0,
    ".*_shoulder_(pitch|roll)_joint": 9.0, ".*_shoulder_yaw_joint": 20.0, ".*_elbow_joint": 20.0,
    ".*_wrist_.*_joint": 31.4,
})
# relax_safety_split limits.estop (config.cpp derive_limits): position offset rad, velocity and torque ratios.
_ESTOP_POSITION_OFFSET = _per_joint({".*_wrist_(pitch|yaw)_joint": -0.10, ".*": 0.0001})
_ESTOP_VELOCITY_RATIO = _per_joint({".*_ankle_.*_joint": 3.0, ".*": 1.0})
_ESTOP_TORQUE_RATIO = _per_joint({
    ".*_ankle_pitch_joint": 1.58334, ".*_elbow_joint": 1.2, ".*_wrist_(pitch|yaw)_joint": 2.0, ".*": 1.0,
})


@dataclass(frozen=True)
class EstopLimit:
    """golem_safety's estop limits of one joint: it latches on q outside [q_low, q_high] rad, |dq| > dq rad/s or
    |tau| > tau N m."""

    q_low: float
    q_high: float
    dq: float
    tau: float


ESTOP_LIMITS: dict[str, EstopLimit] = {
    j: EstopLimit(
        q_low=lo + _ESTOP_POSITION_OFFSET[j],
        q_high=hi - _ESTOP_POSITION_OFFSET[j],
        dq=_VELOCITY_LIMITS[j] * _ESTOP_VELOCITY_RATIO[j],
        tau=EFFORT[j] * _ESTOP_TORQUE_RATIO[j],
    )
    for j, (lo, hi) in JOINT_RANGES.items()
}

# The G1's tracked bodies by their H1-2 names (tasks.py TRACKED_BODIES).
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
ANCHOR_BODY = "pelvis"  # tasks.py apply_gencoord
IMU_BODY = "torso_link"  # the actors' IMU (URDF imu_link) is rigidly on it
