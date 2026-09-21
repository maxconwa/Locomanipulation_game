import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.string import resolve_matching_names

# repo_root/source/locomanipulation_game/locomanipulation_game/assets/h1_2.py
# parents[0]=assets, [1]=locomanipulation_game, [2]=source/locomanipulation_game, [3]=source, [4]=repo_root
REPO_ROOT = Path(__file__).resolve().parents[4]
CL_ASSETS_DIR = Path(os.environ.get("CL_ASSETS_DIR", REPO_ROOT / "third_party" / "CL_Assets"))
H1_2_MAGPIE_USD = CL_ASSETS_DIR / "isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd"


LEG_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Never actuated by any round -- held by its actuator alone. Still observed.
TORSO_JOINTS = ["torso_joint"]

# 27. The observation set, identical across rounds. Legs first so the first 12
# slots match LOWER_BODY_JOINTS and the arm block sits at a fixed offset.
# Grippers excluded: locked.
BODY_JOINTS_NAMES = LEG_JOINT_NAMES + TORSO_JOINTS + ARM_JOINT_NAMES

# Semantic aliases. The *_NAMES lists above are the ordered ground truth; these
# say what a list MEANS to a round. LOWER/UPPER is the IBR split, BODY is the
# observation set, CONTROLLED is what the debug task actuates.
LEG_JOINTS = LEG_JOINT_NAMES
ARM_JOINTS = ARM_JOINT_NAMES
BODY_JOINTS = BODY_JOINTS_NAMES
LOWER_BODY_JOINTS = LEG_JOINT_NAMES
UPPER_BODY_JOINTS = TORSO_JOINTS + ARM_JOINT_NAMES
CONTROLLED_JOINTS = BODY_JOINTS_NAMES



# Patterns, not lists: the set matters, the order does not. `LEG_ONLY` also
# expresses an INTENT that survives a joint being added; an explicit list
# would silently miss it.
LEG_ONLY = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
ANKLE_ONLY = [".*_ankle_.*_joint"]
HIP_YAW_ROLL = [".*_hip_yaw_joint", ".*_hip_roll_joint"]

# Must stay a pattern: the hinge count depends on the Magpie USD.
GRIPPER_JOINTS = ["[lr]g_.*_hinge_.*"]

# Body names, verified against the check_h1_2.py body list.
FEET = ".*_ankle_roll_link"
KNEES = ".*_knee_link"
TRUNK = ["pelvis", "torso_link"]

LEFT_EE = "left_wrist_yaw_link"
RIGHT_EE = "right_wrist_yaw_link"


LIVOX_MOUNT_POS = (0.04874, 0.0, 0.67980)
LIVOX_MOUNT_ROT = (0.99280, 0.0, 0.11979, 0.0)   # (w,x,y,z), 13.76 deg nose-down
IMU_MOUNT_POS = (-0.04452, -0.01891, 0.27756)

LIVOX_VFOV_DEG = (-7.0, 52.0)   # Mid-360 datasheet

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
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.01,
        rest_offset=0.0,
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
        joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
        effort_limit_sim={".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0},
        velocity_limit_sim={".*_hip_.*_joint": 23.0, ".*_knee_joint": 14.0},
        stiffness={".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0},
        damping={".*_hip_.*_joint": 2.5, ".*_knee_joint": 4.0},
        armature=0.01,
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=ANKLE_ONLY,
        effort_limit_sim={".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0},
        velocity_limit_sim=9.0,
        stiffness=40.0,
        damping=2.0,
        armature=0.01,
    ),
    "torso": ImplicitActuatorCfg(
        joint_names_expr=TORSO_JOINTS,
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