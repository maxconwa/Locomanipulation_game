from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import ARM_JOINTS, BODY_JOINTS

from . import mdp
# The terrain, robot, contact sensor and light are shared with the legs round.
# This couples the two files: switch the legs round to generated terrain later
# and this round follows silently. Copy the class in if they should diverge.
from .legs_r0_env_cfg import LocoManipulationSceneCfg

# --- body names: VERIFY against robot.data.body_names ---
# ARM_JOINTS ends at .*_wrist_yaw_joint so these links exist, but the magpie
# USD adds gripper links (the lg_/rg_ prefix GRIPPER_JOINTS matches) beyond
# them. Commanding the wrist means the gripper hangs off the end of the pose
# being tracked and the actual grasp point is offset by the Magpie's length.
LEFT_EE = "left_wrist_yaw_link"
RIGHT_EE = "right_wrist_yaw_link"

# --- weld height ---
# The pelvis is fixed in place, so this is not an initial condition but a
# permanent mounting height. It has to clear the FixStand leg pose or the feet
# rest on the plane and the position-held legs fight the ground reaction.
# VERIFY visually with --num_envs 4.
PELVIS_HEIGHT = 1.2

# --- tracking parameters ---
POS_STD_COARSE = 0.30   # m, gradient across the whole workspace
POS_STD_FINE = 0.10     # m, pays off only in the last few centimetres
QUAT_STD = 0.50         # 



@configclass
class UpperCommandsCfg:
    # Always zero in this round. It exists so the observation vector has the
    # slot, so the walking-plus-reaching round can warm-start from here.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )
    # Ranges are in the PELVIS frame, not the shoulder: the pelvis sits ~0.4 m
    # below the shoulders, so useful z is mostly positive and x is forward of
    # the body. These are guesses -- run with --num_envs 4 and watch where the
    # debug_vis markers land before committing GPU hours, because an
    # unreachable target flattens the reward into a constant and teaches
    # nothing while looking exactly like slow training.
    #
    # The +/-0.5 rad euler ranges are a ~30 deg cone around the default wrist
    # orientation. Widen once position tracking converges; starting wide makes
    # orientation the hard part of a task that has not learned the easy part.
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_EE,
        resampling_time_range=(3.0, 5.0),
        # Quaternion double cover: q and -q are the same rotation, so without
        # this the sampler can hand the policy either sign for one target.
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
class UpperActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class UpperObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Both are trivially zero while the pelvis is welded. They stay in the
        # vector for the walking round, where they are not.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # 7-D each: position then w-first quaternion, in the pelvis frame.
        left_ee_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )
        # All 27 body joints, not just the 14 actuated: fixes the obs space
        # across rounds, and the arm policy needs leg state once the legs move
        # underneath it.
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
        actions = ObsTerm(func=mdp.last_action)

        # No gait_phase: no gait here, and no contact_matches_phase reward for
        # it to serve.
        #
        # No explicit end-effector pose either. The policy can do forward
        # kinematics from joint_pos, so it is not required -- but adding it is
        # the first thing to try if tracking learns slowly, since it turns
        # "infer where my hand is" into a subtraction.

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UpperEventCfg:
    """Much thinner than the legs round.

    No reset_base: write_root_state_to_sim does not move a welded root, so the
    term would be a silent no-op. No push_robot for the same reason -- and
    nothing here is balancing anyway. Both return in the round with a
    locomotion policy underneath.

    The two startup randomizations are nearly inert with a fixed pelvis (no
    foot contact to vary friction against, and the weld carries some of the
    torso inertia) but they cost nothing and matter again later.
    """

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
    # The only meaningful reset left: start each episode from a different arm
    # configuration so the policy cannot memorise one trajectory per target.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.2, 0.2)},
    )

@configclass
class UpperRewardsCfg:
    """Exponential tracking kernels rather than the reach task's error penalties.

    With a welded pelvis and no fall termination, negative-only error terms
    would be safe here -- the usual objection (falling becomes the cheap way to
    stop accruing penalty) needs a termination to bite, and this round has
    none. Linear error terms would also give better final precision, since an
    exp kernel flattens as it converges.

    The kernels are kept anyway because this reward structure has to survive
    the walking round, where `fell` comes back and negative-only becomes wrong
    again, and because exp(-e^2/std^2) puts these on the same scale as the legs
    round's track_lin_vel_xy_exp so the two are comparable in TensorBoard.

    If the goal is instead to measure how accurately the arms CAN track, swap
    the six track_* terms for mdp.ee_position_error at weight -1.0 and
    mdp.ee_orientation_error at -0.5, per arm.
    """

    # --- task: coarse kernel to find the target, fine kernel to settle on it.
    # One kernel cannot do both: wide gives no precision incentive, narrow is
    # flat until you are already close. ---
    track_left_pos = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": POS_STD_COARSE,
        },
    )
    track_left_pos_fine = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": POS_STD_FINE,
        },
    )
    track_left_quat = RewTerm(
        func=mdp.track_ee_quat_exp,
        weight=1.0,
        params={
            "command_name": "left_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_EE),
            "std": QUAT_STD,
        },
    )
    track_right_pos = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": POS_STD_COARSE,
        },
    )
    track_right_pos_fine = RewTerm(
        func=mdp.track_ee_pos_exp,
        weight=1.5,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": POS_STD_FINE,
        },
    )
    track_right_quat = RewTerm(
        func=mdp.track_ee_quat_exp,
        weight=1.0,
        params={
            "command_name": "right_ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_EE),
            "std": QUAT_STD,
        },
    )

    # No alive term and no termination penalty: nothing can terminate early
    # except the timeout, which is truncation. Peak positive is 8.0/step.

    # --- effort and smoothness ---
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- safety ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    # undesired_contacts returns a COUNT of bodies over threshold, so this is
    # -1.0 per contacting link per second against an 8.0/s task ceiling.
    # Calibrate off the first run.
    self_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*_shoulder_.*_link",
                    ".*_elbow.*_link",
                    ".*_wrist_.*_link",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class UpperTerminationsCfg:
    # Timeout only. The pelvis is welded, so there is no fall to detect and the
    # legs round's illegal_contact term on the trunk would never fire.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LocoManipulationUpperR0EnvCfg(ManagerBasedRLEnvCfg):
    scene: LocoManipulationSceneCfg = LocoManipulationSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: UpperObservationsCfg = UpperObservationsCfg()
    actions: UpperActionsCfg = UpperActionsCfg()
    commands: UpperCommandsCfg = UpperCommandsCfg()
    events: UpperEventCfg = UpperEventCfg()
    rewards: UpperRewardsCfg = UpperRewardsCfg()
    terminations: UpperTerminationsCfg = UpperTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4          # policy 50 Hz, physics 200 Hz
        # 20 s, not the legs round's 300: what matters here is how fast and how
        # accurately a target is reached, and at 3-5 s per resample this
        # already holds several reaches. More resets per unit compute too.
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Weld the pelvis. configclass deep-copies the default per instance, so
        # mutating scene.robot here does not leak back into H1_2_MAGPIE_CFG or
        # the legs round.
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.pos = (0.0, 0.0, PELVIS_HEIGHT)

        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.lookat = (0.0, 0.0, PELVIS_HEIGHT)
        self.scene.height_scanner = None