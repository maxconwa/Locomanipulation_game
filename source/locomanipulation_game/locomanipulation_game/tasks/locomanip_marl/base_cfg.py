"""The scene, the legs' base reward set (ALMI-Open's h1_2_lower config), events and terminations that
locomanip_marl_env_cfg.py extends.

Field order matters: configclass keeps a base class's slots, so the reward, event and termination managers run these
terms in this order (with the subclasses' additions after them).
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import (
    ALL_JOINTS_NAMES,
    ANKLE_JOINT_NAMES,
    COLLISION_LINK_NAMES,
    FOOT_LINK_NAMES,
    H1_2_MAGPIE_CFG,
    HIP_YAW_ROLL_JOINT_NAMES,
    KNEE_LINK_NAMES,
    LOWER_JOINT_NAMES,
    STANDING_PELVIS_HEIGHT,
)

from . import mdp

HEIGHT_SCAN_RAISE = 20.0
SELF_CONTACT_LINK_NAMES = COLLISION_LINK_NAMES[:-1]
SELF_CONTACT_SENSOR_NAMES = [f"self_contact_{link}" for link in SELF_CONTACT_LINK_NAMES]
CONTACT_HISTORY = 4 # >= env decimation: one slot per physics substep

TERRAINS_FLAT_CFG = TerrainGeneratorCfg(
    num_rows=10,
    num_cols=20,
    size=(8.0, 8.0),
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    seed=0,
    curriculum=True,          # rows by difficulty; every patch is the same flat plane
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAINS_FLAT_CFG,
        max_init_terrain_level=0,   # start everyone flat; None means all levels
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = H1_2_MAGPIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # All bodies: the reward terms need feet, knees, pelvis and torso.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=CONTACT_HISTORY, track_air_time=True
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, HEIGHT_SCAN_RAISE)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/imu_link",
        debug_vis=True,
    )
    def __post_init__(self):
        # PhysX filters one body against many, never many against many, so each
        # link gets its own sensor. Each filters only against the links after it
        # in COLLISION_LINK_NAMES, so every pair is reported by exactly one sensor.
        for i, link in enumerate(SELF_CONTACT_LINK_NAMES):
            setattr(self, SELF_CONTACT_SENSOR_NAMES[i], ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + link,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + other for other in COLLISION_LINK_NAMES[i + 1:]], #slice avoids double counting contacts
                history_length=CONTACT_HISTORY,
            ))


# --- reward parameters (h1_2_lower_config.py, class rewards) ---
BASE_HEIGHT_TARGET = STANDING_PELVIS_HEIGHT
FEET_SWING_HEIGHT = 0.08
MIN_DIST = 0.3
MAX_DIST = 0.6
MAX_CONTACT_FORCE = 1400.0
ALMI_TRACKING_STD = 0.5  # ALMI tracking_sigma = 0.25 = std^2; LegsRewardsCfg replaces both tracking terms


@configclass
class LowerRewardsCfg:
    # --- task ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": ALMI_TRACKING_STD},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": ALMI_TRACKING_STD},
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


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 4.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Offsets from init_state.pos, within each env's own origin. Keep z
            # small and non-negative: a large drop fills early training with
            # spurious contacts and false terminations.
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.1),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,   # additive, unlike _by_scale
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True),
            "position_range": (-0.2, 0.2), "velocity_range": (-0.5, 0.5)
            },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"distance_buffer": 3.0},
        time_out=True,
    )
