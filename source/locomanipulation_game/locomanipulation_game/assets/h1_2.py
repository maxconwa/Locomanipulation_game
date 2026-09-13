import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg



# This file lives at: repo_root/source/locomanipulation_game/locomanipulation_game/assets/h1_2.py
# parents[0]=assets, [1]=locomanipulation_game, [2]=source/locomanipulation_game, [3]=source, [4]=repo_root
REPO_ROOT = Path(__file__).resolve().parents[4]
CL_ASSETS_DIR = Path(os.environ.get("CL_ASSETS_DIR", REPO_ROOT / "third_party" / "CL_Assets"))
H1_2_MAGPIE_USD = CL_ASSETS_DIR / "isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd"


LEG_JOINTS = [
    ".*_hip_yaw_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint",
    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
]

TORSO_JOINTS = ["torso_joint"]

ARM_JOINTS = [
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
]


# IBR split: which joints each policy controls.
LOWER_BODY_JOINTS = LEG_JOINTS
UPPER_BODY_JOINTS = TORSO_JOINTS + ARM_JOINTS


BODY_JOINTS = LEG_JOINTS + TORSO_JOINTS + ARM_JOINTS  # 27

GRIPPER_JOINTS = ["[lr]g_.*_hinge_.*"]


CONTROLLED_JOINTS = BODY_JOINTS


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
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=4,
        
    ),
)


_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),  # pelvis height in meters; tune so feet start just above ground
    joint_pos={
        # Matches the FixStand pose in unitree_rl_lab
        # deploy/robots/h1_2/config/config.yaml, i.e. the pose the real robot is
        # interpolated to before a policy takes over. This is also the offset
        # that actions are measured from, so deployment must start from here.
        ".*_hip_yaw_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": -0.3,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.2,
        ".*_ankle_roll_joint": 0.0,
        "torso_joint": 0.0,
        ".*_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
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
        armature=0.01
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_.*_joint"],
        effort_limit_sim={".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0},
        velocity_limit_sim=9.0,
        stiffness=40.0,
        damping=2.0,
        armature=0.01
    ),
    "torso": ImplicitActuatorCfg(
        joint_names_expr=["torso_joint"],
        effort_limit_sim=200.0,
        velocity_limit_sim=23.0,
        stiffness=300.0,
        damping=3.0,
        armature=0.01

    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
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
            ".*_shoulder_.*_joint": 120.0,   # ALMI
            ".*_elbow_joint": 80.0,          # ALMI
            ".*_wrist_.*_joint": 40.0,       # no ALMI reference; wrists are absent there
        },
        damping={
            ".*_shoulder_.*_joint": 2.0,   # ALMI
            ".*_elbow_joint": 1.0,         # ALMI
            ".*_wrist_.*_joint": 1.0,      # no ALMI reference
        },
        armature=0.01

    ),

    # Locked open. No policy commands these; stiff gains hold every hinge at
    # its default, which also substitutes for the missing linkage constraint.
    "grippers": ImplicitActuatorCfg(
        joint_names_expr=GRIPPER_JOINTS,
        effort_limit_sim=10.0,   # from the URDF
        velocity_limit_sim=3.14,
        stiffness=100.0,
        damping=5.0,
    ),
}


H1_2_MAGPIE_CFG = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=_ACTUATORS,
)