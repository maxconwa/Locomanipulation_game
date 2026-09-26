import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# repo_root/source/locomanipulation_game/locomanipulation_game/assets/h1_2.py
# parents[0]=assets, [1]=locomanipulation_game, [2]=source/locomanipulation_game, [3]=source, [4]=repo_root
REPO_ROOT = Path(__file__).resolve().parents[4]
CL_ASSETS_DIR = Path(os.environ.get("CL_ASSETS_DIR", REPO_ROOT / "third_party" / "CL_Assets"))
H1_2_MAGPIE_USD = CL_ASSETS_DIR / "isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd"




FOOT_JOINT_NAMES = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
FOOT_LINK_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

ANKLE_JOINT_NAMES = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"] + FOOT_JOINT_NAMES
ANKLE_LINK_NAMES = ["left_ankle_pitch_link", "right_ankle_pitch_link"] + FOOT_LINK_NAMES


KNEE_JOINT_NAMES = ["left_knee_joint", "right_knee_joint"]
KNEE_LINK_NAMES = ["left_knee_link", "right_knee_link"]

HIP_YAW_ROLL_JOINT_NAMES = ["left_hip_yaw_joint", "right_hip_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint"]
HIP_YAW_ROLL_LINK_NAMES = ["left_hip_yaw_link", "right_hip_yaw_link", "left_hip_roll_link", "right_hip_roll_link"]


HIP_JOINT_NAMES = ["left_hip_pitch_joint", "right_hip_pitch_joint"] + HIP_YAW_ROLL_JOINT_NAMES
HIP_LINK_NAMES = ["left_hip_pitch_link", "right_hip_pitch_link"] + HIP_YAW_ROLL_LINK_NAMES

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



SHOULDER_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
]
SHOULDER_LINK_NAMES = [
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
]

ELBOW_JOINT_NAMES = ["left_elbow_joint", "right_elbow_joint"]
ELBOW_LINK_NAMES = ["left_elbow_link", "right_elbow_link"]


WRIST_JOINT_NAMES = [
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WRIST_LINK_NAMES = [
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
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

TORSO_JOINTS_NAME = "torso_joint"
TORSO_LINK_NAME = "torso_link"
PELVIS_LINK_NAME = "pelvis"
FINGER_LINK_NAMES = ["lg_left_finger", "lg_right_finger", "rg_left_finger", "rg_right_finger"]



ALL_JOINTS_NAMES = LOWER_JOINT_NAMES + [TORSO_JOINTS_NAME] + ARM_JOINT_NAMES
ALL_LINKS_NAMES = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES + [TORSO_LINK_NAME] + ARM_LINK_NAMES + FINGER_LINK_NAMES


# Real bodies (they move and have mass) with no collision geometry in the USD.
NO_COLLIDER_LINK_NAMES = [
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]
COLLISION_LINK_NAMES = [n for n in ALL_LINKS_NAMES if n not in NO_COLLIDER_LINK_NAMES]




CONTROLLED_JOINTS = LOWER_JOINT_NAMES + ARM_JOINT_NAMES
CONTROLLED_LINKS = LOWER_LINK_NAMES + ARM_LINK_NAMES




# Must stay a pattern: the hinge count depends on the Magpie USD.
GRIPPER_JOINTS = ["[lr]g_.*_hinge_.*"]

STANDING_PELVIS_HEIGHT = 1.0024
SOLE_OFFSET = 0.0450


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
    pos=(0.0, 0.0, STANDING_PELVIS_HEIGHT),
    joint_pos={
        ".*_hip_yaw_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": -0.3,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.2,
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

_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=HIP_JOINT_NAMES + KNEE_JOINT_NAMES,
        effort_limit_sim={".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0},
        velocity_limit_sim={".*_hip_.*_joint": 23.0, ".*_knee_joint": 14.0},
        stiffness={".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0},
        damping={".*_hip_.*_joint": 2.5, ".*_knee_joint": 4.0},
        armature=0.01,
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=ANKLE_JOINT_NAMES,
        effort_limit_sim={".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0},
        velocity_limit_sim=9.0,
        stiffness=40.0,
        damping=2.0,
        armature=0.01,
    ),
    "torso": ImplicitActuatorCfg(
        joint_names_expr=[TORSO_JOINTS_NAME],
        effort_limit_sim=200.0,
        velocity_limit_sim=23.0,
        stiffness=300.0,
        damping=3.0,
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=ARM_JOINT_NAMES,
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 40.0,
            ".*_shoulder_roll_joint": 40.0,
            ".*_shoulder_yaw_joint": 18.0,
            ".*_elbow_joint": 18.0,
            ".*_wrist_.*_joint": 19.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 9.0,
            ".*_shoulder_roll_joint": 9.0,
            ".*_shoulder_yaw_joint": 20.0,
            ".*_elbow_joint": 20.0,
            ".*_wrist_.*_joint": 31.4,
        },
        stiffness={
            ".*_shoulder_.*_joint": 120.0,   
            ".*_elbow_joint": 80.0,          
            ".*_wrist_.*_joint": 40.0,      
        },
        damping={
            ".*_shoulder_.*_joint": 2.0,   
            ".*_elbow_joint": 1.0,         
            ".*_wrist_.*_joint": 1.0,      
        },
        armature=0.01,
    ),
    # Locked open. Stiff gains hold each hinge at default, substituting for the
    # missing linkage constraint.
    "grippers": ImplicitActuatorCfg(
        joint_names_expr=GRIPPER_JOINTS,
        effort_limit_sim=10.0,   # from the URDF
        velocity_limit_sim=3.14,
        stiffness=100.0,
        damping=5.0,
        armature=0.01,
    ),
}


H1_2_MAGPIE_CFG = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=_ACTUATORS,
)