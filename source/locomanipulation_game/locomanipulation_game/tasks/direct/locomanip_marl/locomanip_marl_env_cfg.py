"""Two agents, one H1-2 on flat ground: the legs track a velocity command, the arms a wrist pose each. MAPPO via skrl.

At every command event an env gets navigation (a velocity command, the arms at their rest pose) or an arm goal (zero
velocity, a wrist pose per arm fixed in the world), with a settle between modes (mdp.ArmTargetsCommand). Low arm
goals come from tables built in squats, so the legs must crouch to reach them. The policies see the arm command in
the pelvis frame, moved by leg odometry as on the robot; rewards score the true target.

Built for GOLEM's deployment: joint targets are bounded the way its safety layer clips them and by PD torque, its
e-stops end the episode (golem_safety.py), the joints carry RoboCasa's passive damping and armature, and targets
land 0-20 ms late. torso_joint belongs to neither agent: its actuator holds it at default.
"""

import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import (
    ALL_JOINTS_NAMES,
    ARM_JOINT_NAMES,
    ARM_LINK_NAMES,
    COLLISION_LINK_NAMES,
    FINGER_LINK_NAMES,
    FOOT_LINK_NAMES,
    LOWER_JOINT_NAMES,
    LOWER_LINK_NAMES,
    PELVIS_LINK_NAME,
    STANDING_PELVIS_HEIGHT,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.common.reward_cfg import (
    BASE_HEIGHT_TARGET,
    LowerRewardsCfg,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.common.scenes import (
    SELF_CONTACT_LINK_NAMES,
    SELF_CONTACT_SENSOR_NAMES,
    TERRAINS_FLAT_CFG,
    TerrainSceneCfg,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.legs_r0_env_cfg import (
    EventCfg,
    TerminationsCfg,
)

from . import mdp
from .golem_safety import GOLEM_TARGET_CLIP, GolemEstopCfg
from .odometry import PelvisEstimatorCfg

LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[7:]
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"
ARM_COMMAND = "arm_targets"
ODOMETRY_HISTORY = 4  # frames in the estimator's window

# Each agent is charged for the self-contact pairs that include one of its links.
LEGS_OWN_LINKS = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES
ARMS_OWN_LINKS = ARM_LINK_NAMES + FINGER_LINK_NAMES

# agent -> the action term it drives, in possible_agents order (the env concatenates actions in that order)
AGENT_ACTION_TERMS = {"legs": "joint_pos", "arms": "arm_pos"}
# The 27 motor joints (not the gripper hinges).
BODY_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint", ".*_shoulder_.*_joint",
               ".*_elbow_joint", ".*_wrist_.*_joint"]

POLICY_DT = 0.02          # decimation 4 x sim dt 0.005; reward weights are per second
GOAL_BONUS = 5.0          # paid once per reached goal
# Each agent's reward includes this share of the other's command-following terms.
LEGS_SHARE_OF_ARMS = 0.5
ARMS_SHARE_OF_LEGS = 0.1
# The legs' gait shaping during arm goals, as a share of its weight: these terms resist a crouch.
ARM_GOAL_SHAPING_SCALE = {"lin_vel_z": 0.0, "ang_vel_xy": 0.5, "hip_pos": 0.2}
TRACKING_STD = 0.25       # velocity tracking kernel: standing still under a turn command earns little
CLIP_ACTIONS = 10.0       # the env clamps every raw policy action to +-this
TORQUE_HEADROOM = 0.85    # targets ask for at most this share of a joint's effort limit (the e-stop trips at 0.9)


@configclass
class MarlCommandsCfg:
    # first: arm_targets zeroes and resamples it
    base_velocity = mdp.ModalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 10.0),
        rel_standing_envs=0.05,
        heading_command=False,  # a yaw rate, like a joystick
        debug_vis=True,
        ranges=mdp.ModalVelocityCommandCfg.Ranges(lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)),
        arm_command_name=ARM_COMMAND,
        # uniform sampling almost never draws a turn in place or a pure sideways walk
        pure_turn_prob=0.2,
        pure_lateral_prob=0.1,
    )
    arm_targets = mdp.ArmTargetsCommandCfg(
        asset_name="robot",
        velocity_command_name="base_velocity",
        body_names=[LEFT_EE_BODY, RIGHT_EE_BODY],
        joint_names=[LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES],
        collision_body_names=[
            ["left_(shoulder|elbow|wrist)_.*", "lg_.*"],
            ["right_(shoulder|elbow|wrist)_.*", "rg_.*"],
        ],
        foot_body_names=FOOT_LINK_NAMES,
        standing_height=STANDING_PELVIS_HEIGHT,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
    )


@configclass
class MarlActionsCfg:
    # Named joint_pos so the manager-based reward ankle_action_rate_l2 finds the legs' term, and first: that reward
    # indexes the whole action vector by the term's joint order.
    joint_pos = mdp.BoundedJointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        target_margin=-GOLEM_TARGET_CLIP,
        torque_headroom=TORQUE_HEADROOM,
    )
    arm_pos = mdp.IKResidualArmActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.2,
        use_default_offset=False,
        preserve_order=True,
        body_names=[LEFT_EE_BODY, RIGHT_EE_BODY],
        arm_joint_names=[LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES],
        command_name=ARM_COMMAND,
        residual_cutoff_hz=3.0,
        target_margin=-GOLEM_TARGET_CLIP,
        torque_headroom=TORQUE_HEADROOM,
    )


def _imu(func, noise: float, **kwargs) -> ObsTerm:
    return ObsTerm(func=func, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-noise, n_max=noise), **kwargs)


def _joints(func, noise: float, names=ALL_JOINTS_NAMES, **kwargs) -> ObsTerm:
    return ObsTerm(
        func=func,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=names, preserve_order=True)},
        noise=Unoise(n_min=-noise, n_max=noise),
        **kwargs,
    )


_WRISTS = SceneEntityCfg("robot", body_names=[LEFT_EE_BODY, RIGHT_EE_BODY], preserve_order=True)
_HISTORY = {"history_length": ODOMETRY_HISTORY, "flatten_history_dim": True}


@configclass
class MarlObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        """What both actors see. The deploy controller builds the same vector, term by term."""

        base_lin_acc = _imu(mdp.imu_lin_acc, 0.5)
        base_ang_vel = _imu(mdp.imu_ang_vel, 0.2)
        projected_gravity = _imu(mdp.imu_projected_gravity, 0.05)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        arm_goal = ObsTerm(func=mdp.arm_goal_active, params={"command_name": ARM_COMMAND})
        ee_targets = ObsTerm(func=mdp.arm_targets_in_root_xyzw, params={"command_name": ARM_COMMAND})
        # zeros: a crouch is not commanded
        height_drop = ObsTerm(func=mdp.arm_target_height_drop, params={"command_name": ARM_COMMAND, "visible": False})
        joint_pos = _joints(mdp.joint_pos_rel, 0.01)
        joint_vel = _joints(mdp.joint_vel_rel, 1.5)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LegsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.applied_action, params={"action_name": AGENT_ACTION_TERMS["legs"]})

    @configclass
    class ArmsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.applied_action, params={"action_name": AGENT_ACTION_TERMS["arms"]})
        wrist_poses = ObsTerm(func=mdp.body_pose_in_root_xyzw, params={"asset_cfg": _WRISTS},
                              noise=Unoise(n_min=-0.005, n_max=0.005))
        wrist_errors = ObsTerm(func=mdp.arm_target_error_in_root, params={"command_name": ARM_COMMAND},
                               noise=Unoise(n_min=-0.005, n_max=0.005))

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged and noise-free; with both actors' observations, the env state both critics see."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        ee_poses = ObsTerm(func=mdp.body_pose_in_root_xyzw, params={"asset_cfg": _WRISTS})
        true_ee_targets = ObsTerm(func=mdp.true_arm_targets_in_root_xyzw, params={"command_name": ARM_COMMAND})
        pelvis_height = ObsTerm(func=mdp.pelvis_height_above_ground)
        target_drop = ObsTerm(func=mdp.arm_target_height_drop, params={"command_name": ARM_COMMAND, "visible": True})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OdometryCfg(ObsGroup):
        """The pelvis-motion estimator's input, not a policy input: sensor-level noise, ODOMETRY_HISTORY frames up
        to the step's end. Leg torques say which foot carries the robot."""

        base_lin_acc = _imu(mdp.imu_lin_acc, 0.05, **_HISTORY)
        base_ang_vel = _imu(mdp.imu_ang_vel, 0.02, **_HISTORY)
        projected_gravity = _imu(mdp.imu_projected_gravity, 0.005, **_HISTORY)
        joint_pos = _joints(mdp.joint_pos_rel, 0.001, **_HISTORY)
        joint_vel = _joints(mdp.joint_vel_rel, 0.05, **_HISTORY)
        leg_torques = _joints(mdp.joint_effort, 2.0, names=LOWER_JOINT_NAMES, **_HISTORY)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    legs: LegsCfg = LegsCfg()
    arms: ArmsCfg = ArmsCfg()
    critic: CriticCfg = CriticCfg()
    odometry: OdometryCfg = OdometryCfg()


_SELF_CONTACT_PARAMS = {
    "sensor_names": SELF_CONTACT_SENSOR_NAMES,
    "sensor_link_names": SELF_CONTACT_LINK_NAMES,
    "filter_link_names": COLLISION_LINK_NAMES,
    "threshold": 0.1,
}


@configclass
class LegsRewardsCfg(LowerRewardsCfg):
    """The manager-based game's legs reward, changed for the shared body.

    action_rate and self_collision count the legs only; base_height and stand_still apply during navigation only,
    so a crouch can emerge during arm goals, and the gait shaping that resists a crouch is scaled down then
    (ARM_GOAL_SHAPING_SCALE). feet_swing_clearance replaces feet_swing_height, which a dragging foot never pays.
    """

    feet_swing_height = None
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-1.0, params={"arm_command_name": ARM_COMMAND}
    )
    action_beyond_clip = RewTerm(
        func=mdp.action_beyond_clip,
        weight=-0.02,
        params={"agent": "legs", "action_name": AGENT_ACTION_TERMS["legs"], "clip": CLIP_ACTIONS},
    )
    feet_swing_clearance = RewTerm(
        func=mdp.feet_swing_clearance,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True),
            "rest_height": 0.045,  # the ankle's standing height
            "lift_height": 0.055,
            "command_name": "base_velocity",
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    def __post_init__(self):
        self.alive.weight = 1.0
        self.track_ang_vel_z.weight = 1.5
        self.track_lin_vel_xy.params["std"] = TRACKING_STD
        self.track_ang_vel_z.params["std"] = TRACKING_STD
        self.action_rate.func = mdp.action_term_rate_l2
        self.action_rate.params = {"action_name": AGENT_ACTION_TERMS["legs"]}
        self.self_collision.func = mdp.self_contacts_involving
        self.self_collision.params = {**_SELF_CONTACT_PARAMS, "own_links": LEGS_OWN_LINKS}
        self.base_height.func = mdp.base_height_l2_navigation
        self.base_height.params = {
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "arm_command_name": ARM_COMMAND,
        }
        self.stand_still.func = mdp.stand_still_navigation
        self.stand_still.params = {**self.stand_still.params, "arm_command_name": ARM_COMMAND}
        for name, func in (
            ("lin_vel_z", mdp.lin_vel_z_l2_modal),
            ("ang_vel_xy", mdp.ang_vel_xy_l2_modal),
            ("hip_pos", mdp.joint_deviation_l2_modal),
        ):
            term = getattr(self, name)
            term.func = func
            term.params = {**term.params, "arm_command_name": ARM_COMMAND, "arm_goal_scale": ARM_GOAL_SHAPING_SCALE[name]}


def _arm_tracking(arm: int, func, std: float) -> RewTerm:
    return RewTerm(func=func, weight=1.0, params={"command_name": ARM_COMMAND, "arm": arm, "std": std})


def _arm_joints(func, weight: float) -> RewTerm:
    return RewTerm(func=func, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)})


@configclass
class ArmsRewardsCfg:
    # exp kernels: the per-step total stays positive
    left_ee_pos = _arm_tracking(0, mdp.arm_target_pos_exp, 0.25)
    left_ee_pos_fine = _arm_tracking(0, mdp.arm_target_pos_exp, 0.05)
    left_ee_quat = _arm_tracking(0, mdp.arm_target_quat_exp, 0.5)
    right_ee_pos = _arm_tracking(1, mdp.arm_target_pos_exp, 0.25)
    right_ee_pos_fine = _arm_tracking(1, mdp.arm_target_pos_exp, 0.05)
    right_ee_quat = _arm_tracking(1, mdp.arm_target_quat_exp, 0.5)
    # weights are per second (the manager multiplies by dt), so this is GOAL_BONUS per goal
    goal_reached = RewTerm(func=mdp.arm_goal_reached, weight=GOAL_BONUS / POLICY_DT, params={"command_name": ARM_COMMAND})
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-2.0, params={"arm_command_name": ARM_COMMAND}
    )
    torques = _arm_joints(mdp.joint_torques_l2, -1.0e-5)
    dof_vel = _arm_joints(mdp.joint_vel_l2, -1.0e-3)
    dof_acc = _arm_joints(mdp.joint_acc_l2, -2.5e-7)
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.01, params={"action_name": AGENT_ACTION_TERMS["arms"]})
    dof_pos_limits = _arm_joints(mdp.joint_pos_limits, -5.0)
    self_collision = RewTerm(
        func=mdp.self_contacts_involving, weight=-1.0, params={**_SELF_CONTACT_PARAMS, "own_links": ARMS_OWN_LINKS}
    )


# Each agent's command-following terms, which the other agent shares in.
ARM_TRACKING_TERMS = ["left_ee_pos", "left_ee_pos_fine", "left_ee_quat",
                      "right_ee_pos", "right_ee_pos_fine", "right_ee_quat", "goal_reached"]
LEG_TRACKING_TERMS = ["track_lin_vel_xy", "track_ang_vel_z"]


@configclass
class MarlRewardsCfg:
    legs: LegsRewardsCfg = LegsRewardsCfg()
    arms: ArmsRewardsCfg = ArmsRewardsCfg()

    def __post_init__(self):
        # logged as Episode_Reward/legs/arms_<term> and Episode_Reward/arms/legs_<term>
        for name in ARM_TRACKING_TERMS:
            term: RewTerm = getattr(self.arms, name)
            setattr(self.legs, f"arms_{name}", term.replace(weight=LEGS_SHARE_OF_ARMS * term.weight))
        for name in LEG_TRACKING_TERMS:
            term = getattr(self.legs, name)
            setattr(self.arms, f"legs_{name}", term.replace(weight=ARMS_SHARE_OF_LEGS * term.weight))


@configclass
class MarlEventCfg(EventCfg):
    # RoboCasa's MuJoCo robot gives every joint damping 10 and armature 0.1 on top of the PD
    passive_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINTS), "damping_distribution_params": (8.0, 12.0),
                "operation": "add"},
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINTS),
                "armature_distribution_params": (0.08, 0.12), "operation": "abs"},
    )
    # frames without an inertial in the URDF, which the USD import gave 1 kg each
    massless_frames = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["head_camera_link", "imu_link", "livox_link", "logo_link"]),
                "mass_distribution_params": (0.01, 0.01), "operation": "abs"},
    )

    def __post_init__(self):
        # resets on the ground, near still, joints near default
        self.reset_base.params = {
            **self.reset_base.params,
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        }
        self.reset_joints.params = {**self.reset_joints.params, "position_range": (-0.05, 0.05), "velocity_range": (-0.1, 0.1)}


@configclass
class MarlTerminationsCfg(TerminationsCfg):
    golem_estop = DoneTerm(func=mdp.golem_estop)


@configclass
class MarlCurriculumCfg:
    arm_target_levels = CurrTerm(func=mdp.arm_target_levels, params={"command_name": ARM_COMMAND})


@configclass
class LocoManipMarlEnvCfg(DirectMARLEnvCfg):
    decimation = 4  # policy 50 Hz, physics 200 Hz
    episode_length_s = 20.0
    possible_agents = ["legs", "arms"]
    # placeholders: the env sets all three from the managers
    action_spaces = {"legs": 12, "arms": 14}
    observation_spaces = {"legs": 1, "arms": 1}
    state_space = 1

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    events: MarlEventCfg = MarlEventCfg()
    commands: MarlCommandsCfg = MarlCommandsCfg()
    actions: MarlActionsCfg = MarlActionsCfg()
    observations: MarlObservationsCfg = MarlObservationsCfg()
    rewards: MarlRewardsCfg = MarlRewardsCfg()
    terminations: MarlTerminationsCfg = MarlTerminationsCfg()
    curriculum: MarlCurriculumCfg = MarlCurriculumCfg()
    estimator: PelvisEstimatorCfg = PelvisEstimatorCfg(foot_body_names=FOOT_LINK_NAMES)
    # GOLEM's e-stops tightened by a margin; wider at the joints behind RoboCasa's e-stops (the knee's only at
    # extension, so the squat keeps its depth)
    golem_estop: GolemEstopCfg = GolemEstopCfg(joint_position_margins={
        ".*_ankle_roll_joint": (0.05, 0.05),
        ".*_ankle_pitch_joint": (0.05, 0.05),
        ".*_knee_joint": (0.05, 0.02),
    })
    # joint targets land this many physics substeps (5 ms each) late, drawn per env and episode: the deploy
    # loop's latency. The agents observe their actions undelayed.
    action_delay_substeps: tuple[int, int] = (0, 4)
    agent_action_terms: dict[str, str] = AGENT_ACTION_TERMS
    clip_actions: float = CLIP_ACTIONS
    # added after the rewards are floored at 0 on terminating (not timed-out) steps
    termination_penalty: float = -5.0
    # soft limits as a share of the hard range, over the asset's 0.9: the knee and ankle pitch bound a feet-flat squat
    soft_joint_pos_limit_factors: dict[str, float] = {".*_knee_joint": 0.95, ".*_ankle_pitch_joint": 0.95}

    def __post_init__(self):
        self.scene.terrain.terrain_generator = TERRAINS_FLAT_CFG
        # contacts every physics step (the gait-phase terms and leg odometry read them), the IMU too (its
        # finite-difference lin_acc divides by the physics dt), the height scan once per policy step
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt
        # the self-contact sensors once per policy step: they are slow; a contact shorter than that can be missed
        for name in SELF_CONTACT_SENSOR_NAMES:
            sensor = getattr(self.scene, name)
            sensor.update_period = self.decimation * self.sim.dt
            sensor.history_length = 1
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
