# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/config.py.
# Changes: TRACKED_BODIES from pretraining.constants; LEG_JOINTS / ARM_JOINTS are constants.LEG_JOINTS / UPPER_JOINTS;
# the stage-2 cfgs (CommandRewardCfg, CommandSamplingCfg, RandomisationCfg) left out.
"""The two-agent split of the H1-2 and the stage-1 reward weights and kernel widths.

The legs agent drives the 12 leg joints (indices 0-11), the arms agent the torso
and the 14 arm joints (12-26), so the concatenated action ``[legs, arms]`` is the
Unitree motor order (``constants.JOINT_NAMES``), which the world model and the env's single
``joint_pos`` action term take as is.
"""

from __future__ import annotations

from dataclasses import dataclass

from pretraining.constants import LEG_JOINTS as _LEG_JOINTS
from pretraining.constants import TRACKED_BODIES
from pretraining.constants import UPPER_JOINTS as _UPPER_JOINTS

AGENTS = ("legs", "arms")
NUM_JOINTS = 27
LEG_JOINTS = _LEG_JOINTS  # slice(0, 12)
ARM_JOINTS = _UPPER_JOINTS  # slice(12, 27): torso_joint, then the left and right arm
ACTION_DIMS = {"legs": 12, "arms": 15}
JOINT_SLICES = {"legs": LEG_JOINTS, "arms": ARM_JOINTS}

# Tracked bodies (constants.TRACKED_BODIES) by role, as indices into that tuple.
PELVIS = TRACKED_BODIES.index("pelvis")
LEG_BODIES = tuple(
    TRACKED_BODIES.index(n)
    for n in (
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    )
)
UPPER_BODIES = tuple(
    TRACKED_BODIES.index(n)
    for n in (
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
)
FEET = (
    TRACKED_BODIES.index("left_ankle_roll_link"),
    TRACKED_BODIES.index("right_ankle_roll_link"),
)
# m: the ankle_roll_link frame's height above the floor with the sole flat on it (the
# MJCF's foot geometry: measured on the training model at the knees-bent keyframe, whose
# height puts the soles on the floor).
ANKLE_GROUND_HEIGHT = 0.0455
WRISTS = (
    TRACKED_BODIES.index("left_wrist_yaw_link"),
    TRACKED_BODIES.index("right_wrist_yaw_link"),
)
# m, wrist_yaw_link frame: the Magpie tool frame (site <side>_grasp_frame), the arms'
# end effector. Its orientation is the wrist link's.
GRASP_OFFSET = (0.231991, 0.0, 0.000625)
EE_DIM = 9  # per end effector: position (3, m) then rot6d (6)


@dataclass(frozen=True)
class Kernel:
    """One Gaussian-kernel reward term, ``weight * exp(-err^2 / std^2)``."""

    weight: float
    std: float


@dataclass(frozen=True)
class EstopPenaltyCfg:
    """Soft penalties that keep each agent's own joints inside golem_safety's estop limits.

    Per joint, the excess of ``|dq| / dq_estop`` and of the PD torque demand
    ``|kp (target - q) - kd dq| / tau_estop`` over ``margin`` (``constants.ESTOP_LIMITS``)
    is summed over the agent's joints and weighted. The velocity is the predicted next
    state's; the torque demand is the first substep's: the state before the step and the
    clamped target of the step's action.

    The position term is on the predicted next ``q``: per joint, how deep it is inside the
    band of ``position_margin`` rad before the nearer estop position limit, as a share of
    the band (0 outside the band, 1 at the limit, more past it), summed and weighted. The
    estop range is 0.1 mm inside the MJCF's joint range, so a joint resting on its stop
    latches the estop.
    """

    margin: float = 0.9  # share of the estop limit where the penalties start
    velocity: float = -5.0  # per unit of |dq| / dq_estop past the margin
    torque: float = -5.0  # per unit of the torque demand / tau_estop past the margin
    position: float = 0.0  # per joint at its estop position limit; 0 turns the term off
    position_margin: float = 0.05  # rad before the limit where the position term starts


@dataclass(frozen=True)
class ImitationRewardCfg:
    """Stage-1 per-agent reward terms; each agent also gets SuperMimic's tracking reward.

    The team reward is ``supermimic.utils.tracking_reward`` unchanged (pelvis 0.5/0.3 m,
    0.5/0.4 rad; all bodies 1.0 each at 0.3 m, 0.4 rad, 1 m/s, 3.14 rad/s; action rate
    -0.1, joint limits -2.0). The legs' terms are its kernels on the pelvis and the six
    leg bodies; the arms' are kernels on the torso and arm bodies and the two tool frames
    in the pelvis frame.
    """

    legs_root_pos: Kernel = Kernel(0.5, 0.3)
    legs_root_ori: Kernel = Kernel(0.5, 0.4)
    legs_body_pos: Kernel = Kernel(1.0, 0.3)
    legs_body_ori: Kernel = Kernel(1.0, 0.4)
    legs_body_lin_vel: Kernel = Kernel(1.0, 1.0)
    legs_body_ang_vel: Kernel = Kernel(1.0, 3.14)
    arms_body_pos: Kernel = Kernel(1.0, 0.15)
    arms_body_ori: Kernel = Kernel(1.0, 0.4)
    arms_ee_pos: Kernel = Kernel(1.0, 0.1)
    arms_ee_ori: Kernel = Kernel(0.5, 0.4)
    estop: EstopPenaltyCfg = EstopPenaltyCfg()
