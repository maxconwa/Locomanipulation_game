import math
 
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
 
from locomanipulation_game.assets.h1_2 import (
    ARM_JOINTS,
    BODY_JOINTS,
    LOWER_BODY_JOINTS,
)
 
from . import mdp
from .legs_r0_env_cfg import (
    ANKLE_ONLY,
    BASE_HEIGHT_TARGET,
    FEET,
    FEET_SWING_HEIGHT,
    HIP_YAW_ROLL,
    KNEES,
    LEG_ONLY,
    MAX_CONTACT_FORCE,
    MAX_DIST,
    MIN_DIST,
    TRACKING_STD,
    TRUNK,
    LocoManipulationSceneCfg,
)
from .upper_r0_env_cfg import (
    LEFT_EE,
    POS_STD_COARSE,
    POS_STD_FINE,
    QUAT_STD,
    RIGHT_EE,
)

ARM_REWARD_SCALE = 0.5

SELF_COLLISION_BODIES = [
    ".*_shoulder_.*_link",
    ".*_elbow.*_link",
    ".*_wrist_.*_link",
    ".*_hip_roll_link",
    ".*_knee_link",
]
SELF_COLLISION_THRESHOLD = 20.0

@configclass
class WBCommandsCfg:
    """Both command sets """
 
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_EE,
        resampling_time_range=(3.0, 5.0),
        make_quat_unique=True,
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.45), pos_y=(0.05, 0.45), pos_z=(0.15, 0.55),
            roll=(-0.5, 0.5), pitch=(-0.5, 0.5), yaw=(-0.5, 0.5),
        ),
    )
    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=RIGHT_EE,
        resampling_time_range=(3.0, 5.0),
        make_quat_unique=True,
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.45), pos_y=(-0.45, -0.05), pos_z=(0.15, 0.55),
            roll=(-0.5, 0.5), pitch=(-0.5, 0.5), yaw=(-0.5, 0.5),
        ),
    )

@configclass
class WBActionsCfg:
    """ONE action term over all 26 joints, not two."""
 
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_JOINTS + ARM_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

@configclass
class WBObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 295 total: 3+3+3 base state, 3 velocity command, 7+7 pose commands,
        # 27+27 joints, 187 height scan, 26 actions, 2 gait phase.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        left_ee_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin, params={"command_name": "base_velocity"}
        )

 
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
 
    policy: PolicyCfg = PolicyCfg()


@configclass
class WBEventCfg:
    """The legs round's full event set. The robot balances again, so the
    perturbations that upper-r0 removed all come back."""
 
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
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.05),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.5, 0.5)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
    )

@configclass
class WBRewardsCfg:
    """Union of both round-0 reward sets, written out rather than inherited.
 
    Deliberately NOT `class WBRewardsCfg(LowerRewardsCfg, UpperRewardsCfg)`.
    Both parents define `torques`, `dof_vel`, `dof_acc`, `action_rate`,
    `dof_pos_limits` and `self_collision`; under multiple inheritance the MRO
    would silently keep one of each and drop the other, with no error and no
    obvious symptom. Spelled out here with leg_/arm_ prefixes so both survive.
 
    Registered against PositiveRewardRLEnv, like the legs round: `fell` is back,
    so an all-negative early phase would pay the policy to fall over. Note the
    clamp also deletes the gradient from the penalty terms whenever the weighted
    sum is negative -- expect Train/mean_reward to sit at zero for a while
    before it starts climbing, and read the per-term curves instead.
    """
 
    # --- locomotion task ---
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
 
    # --- reaching task (scaled -- see ARM_REWARD_SCALE) ---
    track_left_pos = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5 * ARM_REWARD_SCALE,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": POS_STD_COARSE,
        },
    )
    track_left_pos_fine = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5 * ARM_REWARD_SCALE,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": POS_STD_FINE,
        },
    )
    track_left_quat = RewTerm(
        func=mdp.track_ee_quat_exp,
        weight=1.0 * ARM_REWARD_SCALE,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": QUAT_STD,
        },
    )
    track_right_pos = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5 * ARM_REWARD_SCALE,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": POS_STD_COARSE,
        },
    )
    track_right_pos_fine = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5 * ARM_REWARD_SCALE,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": POS_STD_FINE,
        },
    )
    track_right_quat = RewTerm(
        func=mdp.track_ee_quat_exp,
        weight=1.0 * ARM_REWARD_SCALE,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": QUAT_STD,
        },
    )
 
    # --- posture ---
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
 
    # --- gait shaping ---
    contact = RewTerm(
        func=mdp.contact_matches_phase,
        weight=0.18,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "command_name": "base_velocity",
        },
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "target_height": FEET_SWING_HEIGHT,
        },
    )
    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_pair_distance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "min_dist": MIN_DIST,
            "max_dist": MAX_DIST,
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_pair_distance,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=KNEES),
            "min_dist": MIN_DIST,
            "max_dist": MAX_DIST / 2,
        },
    )
    hip_pos = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_ROLL)},
    )
 
    # --- standing still ---
    # Legs only. Do NOT widen these to the arms: at zero velocity command the
    # arms still have live pose commands to track, and penalising arm deviation
    # from default would fight the reaching task exactly when the robot is
    # most able to do it.
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY),
            "command_name": "base_velocity",
        },
    )
    stance_base_vel = RewTerm(
        func=mdp.stance_base_vel,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
 
    # --- effort and smoothness, legs ---
    leg_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    leg_dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    leg_dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    ankle_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_ONLY)},
    )
    ankle_action_rate = RewTerm(
        func=mdp.ankle_action_rate_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_ONLY)},
    )
 
    # --- effort and smoothness, arms ---
    arm_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    arm_dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    arm_dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
 
    # One global term over all 26 actions. Both round-0 envs had this at -0.01
    # over their own 12 or 14, so the per-joint pressure is unchanged; only the
    # summed magnitude roughly doubles.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
 
    # --- safety ---
    leg_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    arm_dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "max_force": MAX_CONTACT_FORCE,
        },
    )
    self_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=SELF_COLLISION_BODIES
            ),
            "threshold": SELF_COLLISION_THRESHOLD,
        },
    )

@configclass
class WBTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TRUNK),
            "threshold": 1.0,
        },
    )

@configclass
class LocoManipulationWBR0EnvCfg(ManagerBasedRLEnvCfg):
    scene: LocoManipulationSceneCfg = LocoManipulationSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: WBObservationsCfg = WBObservationsCfg()
    actions: WBActionsCfg = WBActionsCfg()
    commands: WBCommandsCfg = WBCommandsCfg()
    events: WBEventCfg = WBEventCfg()
    rewards: WBRewardsCfg = WBRewardsCfg()
    terminations: WBTerminationsCfg = WBTerminationsCfg()
 
    def __post_init__(self):
        self.decimation = 4
        # Between the two rounds' values (legs 180 s, upper 20 s). Long enough
        # that a fall is the usual way an episode ends rather than the clock,
        # which is what makes `fell` and the termination penalty meaningful;
        # short enough to get a spread of initial conditions. Worth revisiting
        # once you can see the ratio in Episode_Termination/*.
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)