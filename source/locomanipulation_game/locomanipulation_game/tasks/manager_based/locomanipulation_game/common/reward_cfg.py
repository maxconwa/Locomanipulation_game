from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .. import mdp
from locomanipulation_game.assets.h1_2 import (STANDING_PELVIS_HEIGHT,
                                                FOOT_JOINT_NAMES, FOOT_LINK_NAMES, 
                                                ANKLE_JOINT_NAMES, ANKLE_LINK_NAMES,
                                                KNEE_JOINT_NAMES, KNEE_LINK_NAMES,
                                                HIP_YAW_ROLL_JOINT_NAMES, HIP_YAW_ROLL_LINK_NAMES,
                                                HIP_JOINT_NAMES, HIP_LINK_NAMES,
                                                LOWER_JOINT_NAMES, LOWER_LINK_NAMES,
)

from isaaclab.managers import SceneEntityCfg
from .scenes import SELF_CONTACT_SENSOR_NAMES


# --- reward parameters (h1_2_lower_config.py, class rewards) ---
BASE_HEIGHT_TARGET = STANDING_PELVIS_HEIGHT 
FEET_SWING_HEIGHT = 0.08
MIN_DIST = 0.3
MAX_DIST = 0.6
MAX_CONTACT_FORCE = 1400.0
TRACKING_STD = 0.5      # ALMI tracking_sigma = 0.25 = std^2


@configclass
class LowerRewardsCfg:
    # --- task ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": TRACKING_STD},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": TRACKING_STD},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)



    # --- posture ---
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # --- gait shaping (the ALMI-specific part) ---
    contact = RewTerm(
        func=mdp.contact_matches_phase,
        weight=0.18,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES),
            "command_name": "base_velocity",
        },
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES),
            "target_height": FEET_SWING_HEIGHT,
            "terrain_sensor_cfg":SceneEntityCfg("height_scanner"),
        },
    )
    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES),
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_pair_distance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES),
            "min_dist": MIN_DIST,
            "max_dist": MAX_DIST,
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_pair_distance,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=KNEE_LINK_NAMES),
            "min_dist": MIN_DIST,
            "max_dist": MAX_DIST / 2,   # ALMI halves it for knees
        },
    )
    hip_pos = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_ROLL_JOINT_NAMES)},
    )

    # --- standing still ---
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES),
            "command_name": "base_velocity",
        },
    )
    stance_base_vel = RewTerm(
        func=mdp.stance_base_vel,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )

    # --- effort and smoothness (leg joints only) ---
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    ankle_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)},
    )
    ankle_action_rate = RewTerm(
        func=mdp.ankle_action_rate_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)},
    )

    # --- safety ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)},
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES),
            "max_force": MAX_CONTACT_FORCE,
        },
    )
    self_collision = RewTerm(
        func=mdp.self_contacts,
        weight=-1.0,
        params={"sensor_names": SELF_CONTACT_SENSOR_NAMES, "threshold": 0.1}
    )

