"""The H1-2 with Magpie grippers: joint and link names, its stance, its PD actuators and the action scales that go
with them.

Gains, stance and command behaviour copy Unitree's sport mode (h1_ai_sport 1.2.0.0), read from its /lowcmd and
/lowstate in the 2026-10-19 sport-mode bags (docs/experiments/sportmode_ablations.md).
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# repo_root/source/locomanipulation_game/locomanipulation_game/assets/h1_2.py
REPO_ROOT = Path(__file__).resolve().parents[4]
CL_ASSETS_DIR = Path(os.environ.get("CL_ASSETS_DIR", REPO_ROOT / "third_party" / "CL_Assets"))
H1_2_MAGPIE_USD = CL_ASSETS_DIR / "isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd"

FOOT_LINK_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
ANKLE_JOINT_NAMES = ["left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"]
KNEE_LINK_NAMES = ["left_knee_link", "right_knee_link"]
HIP_YAW_ROLL_JOINT_NAMES = ["left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint"]

# Unitree LowCmd motor order (H1_2_JointIndex in unitree_sdk2_python's h1_2 example).
# Actions and observations use preserve_order=True on these lists: do not reorder.
LOWER_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
LOWER_LINK_NAMES = [
    "left_hip_yaw_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
]
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM_LINK_NAMES = [
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]
TORSO_JOINT_NAME = "torso_joint"
TORSO_LINK_NAME = "torso_link"
PELVIS_LINK_NAME = "pelvis"
FINGER_LINK_NAMES = ["lg_left_finger", "lg_right_finger", "rg_left_finger", "rg_right_finger"]

ALL_JOINTS_NAMES = LOWER_JOINT_NAMES + [TORSO_JOINT_NAME] + ARM_JOINT_NAMES
ALL_LINKS_NAMES = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES + [TORSO_LINK_NAME] + ARM_LINK_NAMES + FINGER_LINK_NAMES
# Real bodies (they move and have mass) with no collision geometry in the USD.
NO_COLLIDER_LINK_NAMES = [
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
COLLISION_LINK_NAMES = [n for n in ALL_LINKS_NAMES if n not in NO_COLLIDER_LINK_NAMES]
# Must stay a pattern: the hinge count depends on the Magpie USD.
GRIPPER_JOINTS = ["[lr]g_.*_hinge_.*"]

# Sport mode's stance: hip pitch and knee are the measured angles' medians over its 4 min of standing in the bags,
# both legs averaged (L/R hip pitch -0.381/-0.369, knee 0.694/0.700; hip yaw and roll about 0). The ankle pitch puts
# the foot flat under a level pelvis (hip + knee + ankle = 0): sport mode stands level (0.2 deg), and its measured
# ankles (-0.364/-0.432) would tilt the foot 4 deg, an offset of the real ankle reading. Walking, its gait-cycle mean
# is close: knee 0.74, hip pitch -0.39. The pelvis then stands STANCE_PELVIS_HEIGHT above the ground (forward
# kinematics on CL_Assets' h1_2 MJCF).
STANCE_PELVIS_HEIGHT = 0.9798
# The knee links' horizontal distance in the stance (the same forward kinematics); the feet are as far apart.
STANCE_KNEE_DISTANCE = 0.326
# The pelvis height of the stance the arm-goal tables were built at (knee 0.5 rad): the arm goals are placed in a
# standing frame this high (mdp.ArmTargetsCommand), 2.3 cm above the sport stance's pelvis.
TABLE_PELVIS_HEIGHT = 1.0024

_SPAWN_CFG = sim_utils.UsdFileCfg(
    usd_path=str(H1_2_MAGPIE_USD),
    activate_contact_sensors=True,
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.1, 0.1, 0.1),  # matches the URDF's "dark"
        metallic=0.3,
        roughness=0.5,
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=4,
    ),
)

_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, STANCE_PELVIS_HEIGHT),
    joint_pos={
        ".*_hip_yaw_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": -0.375,
        ".*_knee_joint": 0.697,
        ".*_ankle_pitch_joint": -0.322,
        ".*_ankle_roll_joint": 0.0,
        "torso_joint": 0.0,
        ".*_shoulder_pitch_joint": 0.0,
        ".*_shoulder_roll_joint": 0.0,
        ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_joint": 0.0,
        ".*_wrist_.*_joint": 0.0,
        "[lr]g_.*_hinge_.*": 0.0,   # jaws open; passive links at rest
    },
    joint_vel={".*": 0.0},
)


def _pd(joint_names_expr, effort, velocity, stiffness, damping) -> IdealPDActuatorCfg:
    """Explicit PD: the torque kp (target - q) - kd qd computed at every physics step and clipped at the effort limit
    (the robot's motor PD, not PhysX's implicit drive)."""
    return IdealPDActuatorCfg(joint_names_expr=joint_names_expr, effort_limit=effort, effort_limit_sim=effort,
                              velocity_limit=velocity, velocity_limit_sim=velocity, stiffness=stiffness,
                              damping=damping, armature=0.01)


# Sport mode's PD gains on every joint (stiffness Nm/rad, damping Nm s/rad), from its /lowcmd.
_ACTUATORS = {
    "legs": _pd(LOWER_JOINT_NAMES[:4] + LOWER_JOINT_NAMES[6:10],
                {".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0}, {".*_hip_.*_joint": 23.0, ".*_knee_joint": 14.0},
                {".*_hip_.*_joint": 250.0, ".*_knee_joint": 400.0}, {".*_hip_.*_joint": 6.5, ".*_knee_joint": 10.0}),
    "feet": _pd(ANKLE_JOINT_NAMES, {".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0}, 9.0,
                {".*_ankle_pitch_joint": 80.0, ".*_ankle_roll_joint": 40.0},
                {".*_ankle_pitch_joint": 1.5, ".*_ankle_roll_joint": 0.4}),
    "torso": _pd([TORSO_JOINT_NAME], 200.0, 23.0, {"torso_joint": 600.0}, {"torso_joint": 6.0}),
    "arms": _pd(ARM_JOINT_NAMES,
                {".*_shoulder_pitch_joint": 40.0, ".*_shoulder_roll_joint": 40.0, ".*_shoulder_yaw_joint": 18.0,
                 ".*_elbow_joint": 18.0, ".*_wrist_.*_joint": 19.0},
                {".*_shoulder_pitch_joint": 9.0, ".*_shoulder_roll_joint": 9.0, ".*_shoulder_yaw_joint": 20.0,
                 ".*_elbow_joint": 20.0, ".*_wrist_.*_joint": 31.4},
                {".*_shoulder_pitch_joint": 250.0, ".*_shoulder_roll_joint": 400.0, ".*_shoulder_yaw_joint": 80.0,
                 ".*_elbow_joint": 80.0, ".*_wrist_.*_joint": 20.0},
                {".*_shoulder_pitch_joint": 6.0, ".*_shoulder_roll_joint": 10.0, ".*_shoulder_yaw_joint": 2.0,
                 ".*_elbow_joint": 2.0, ".*_wrist_.*_joint": 0.5}),
    # Locked open by stiff implicit drives, which hold each hinge at default in place of the missing linkage
    # constraint (an explicit PD at 2 ms would not hold them stably).
    "grippers": ImplicitActuatorCfg(
        joint_names_expr=GRIPPER_JOINTS,
        effort_limit_sim=10.0,   # from the URDF
        velocity_limit_sim=3.14,
        stiffness=100.0,
        damping=5.0,
        armature=0.01,
    ),
}

# Action scales (rad of joint target per action unit), matched to those gains: the scale that asked for a given
# torque at our earlier gains (legs 0.25 at hip 200 / knee 300 / ankle 40 Nm/rad; the arms' residual 0.2 at shoulder
# 120, elbow 80, wrist 40) times earlier kp / sport kp. With the earlier scales, sport mode's gains marched in place
# (the sport-mode ablations' gains run): the same action asked for up to twice the torque.
LEGS_ACTION_SCALE = {".*_hip_.*_joint": 0.2, ".*_knee_joint": 0.1875, ".*_ankle_pitch_joint": 0.125,
                     ".*_ankle_roll_joint": 0.25}
ARMS_RESIDUAL_SCALE = {".*_shoulder_pitch_joint": 0.096, ".*_shoulder_roll_joint": 0.06, ".*_shoulder_yaw_joint": 0.3,
                       ".*_elbow_joint": 0.2, ".*_wrist_.*_joint": 0.4}

H1_2_MAGPIE_CFG = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=_ACTUATORS,
)
