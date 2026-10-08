"""Two agents, one H1-2 on flat ground: the legs track a velocity command, the arms a wrist pose each. MAPPO via skrl.

At every command event an env gets navigation (a velocity command, each wrist holding its rest pose) or an arm goal
(zero velocity, a wrist pose per arm fixed in the world; mdp.ArmTargetsCommand). Arm goals start once most episodes
survive (warm_start_timeout_share). Goals are standing-reachable wrist poses lowered by a depth curriculum; nothing
rewards a posture, so a crouch has to emerge from the legs' share of the arms' reward. The policies see the arm
command in the pelvis frame, moved by a learned pelvis-motion estimate as on the robot; rewards score the true target.

Joint targets are bounded inside the joint limits (as GOLEM's safety layer clips them) and by PD torque.
torso_joint belongs to neither agent: its actuator holds it at default.
"""

import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
from locomanipulation_game.tasks.manager_based.locomanipulation_game.common.reward_cfg import LowerRewardsCfg
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
from locomanipulation_game.tasks.manager_based.locomanipulation_game.mdp.rewards import GAIT_PERIOD

from . import mdp
from .golem_safety import GOLEM_TARGET_CLIP
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

LEGS_ACTION = "joint_pos"
ARMS_ACTION = "arm_pos"
# agent -> the action terms it drives. In possible_agents order the terms follow MarlActionsCfg's order: the env
# concatenates the agents' actions into the ActionManager's vector.
AGENT_ACTION_TERMS = {"legs": [LEGS_ACTION], "arms": [ARMS_ACTION]}

POLICY_DT = 0.02          # decimation 4 x sim dt 0.005; reward weights are per second
GOAL_BONUS = 5.0          # paid once per reached goal
# lambda: each agent's reward includes this share of the other's command-following terms
REWARD_SHARE = 0.5
ARM_GOAL_HIP_POS_SCALE = 0.2  # the legs' hip yaw/roll deviation term during arm goals, as a share of its weight
TRACKING_STD = 0.25       # velocity tracking kernel width, on the gait-cycle mean velocity
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
        pure_lateral_prob=0.2,
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
        joint_pos = _joints(mdp.joint_pos_rel, 0.01)
        joint_vel = _joints(mdp.joint_vel_rel, 1.5)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LegsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.applied_action, params={"action_name": LEGS_ACTION})

    @configclass
    class ArmsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.applied_action, params={"action_name": ARMS_ACTION})
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

    No term holds the pelvis height or the knees: base_height and stand_still are gone. hip_pos (hip yaw and roll at
    their defaults) and arm_goal_yaw_rate, as in run R, hold the pelvis's yaw against twisting; hip_pos is scaled to
    ARM_GOAL_HIP_POS_SCALE during arm goals. action_rate and self_collision count the legs only.
    feet_swing_clearance replaces feet_swing_height, which a dragging foot never pays. Velocity tracking scores the
    gait-cycle mean velocity, so stepping in place to turn doesn't pay for its sway.
    """

    base_height = None
    stand_still = None
    feet_swing_height = None
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-1.0, params={"arm_command_name": ARM_COMMAND}
    )
    action_beyond_clip = RewTerm(
        func=mdp.action_beyond_clip,
        weight=-0.02,
        params={"agent": "legs", "action_name": LEGS_ACTION, "clip": CLIP_ACTIONS},
    )
    feet_swing_clearance = RewTerm(
        func=mdp.feet_swing_clearance,
        weight=-25.0,
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
        # scored on the pelvis velocity averaged over one gait cycle (mdp.track_velocity_avg_exp)
        for term, component in ((self.track_lin_vel_xy, "lin_xy"), (self.track_ang_vel_z, "ang_z")):
            term.func = mdp.track_velocity_avg_exp
            term.params = {"command_name": "base_velocity", "std": TRACKING_STD, "component": component,
                           "window_s": GAIT_PERIOD}
        self.action_rate.func = mdp.action_term_rate_l2
        self.action_rate.params = {"action_name": LEGS_ACTION}
        self.self_collision.func = mdp.self_contacts_involving
        self.self_collision.params = {**_SELF_CONTACT_PARAMS, "own_links": LEGS_OWN_LINKS}
        self.hip_pos.func = mdp.joint_deviation_l2_modal
        self.hip_pos.params = {**self.hip_pos.params, "arm_command_name": ARM_COMMAND,
                               "arm_goal_scale": ARM_GOAL_HIP_POS_SCALE}


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
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.01, params={"action_name": ARMS_ACTION})
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
            setattr(self.legs, f"arms_{name}", term.replace(weight=REWARD_SHARE * term.weight))
        for name in LEG_TRACKING_TERMS:
            term = getattr(self.legs, name)
            setattr(self.arms, f"legs_{name}", term.replace(weight=REWARD_SHARE * term.weight))


@configclass
class MarlEventCfg(EventCfg):
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
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: MarlCurriculumCfg = MarlCurriculumCfg()
    estimator: PelvisEstimatorCfg = PelvisEstimatorCfg(odometry="learned")
    # while training, episodes are navigation only until this share of episode ends are time-outs rather than falls
    # (LocoManipMarlEnv._update_warm_start); then arm goals start
    warm_start_timeout_share: float = 0.8
    agent_action_terms: dict[str, list[str]] = AGENT_ACTION_TERMS
    clip_actions: float = CLIP_ACTIONS
    # added after the rewards are floored at 0 on terminating (not timed-out) steps
    termination_penalty: float = -5.0
    # soft limits as a share of the hard range, over the asset's 0.9: the knee and ankle pitch may use more of their
    # range before dof_pos_limits charges
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


"""
One agent for the whole body (the comparison for the two-agent game).
"""


@configclass
class WholeBodyObservationsCfg:
    @configclass
    class WholeCfg(MarlObservationsCfg.ProprioCfg):
        """Both agents' observations: the shared vector once, then the legs' and the arms' own terms."""

        leg_actions = ObsTerm(func=mdp.applied_action, params={"action_name": LEGS_ACTION})
        arm_actions = ObsTerm(func=mdp.applied_action, params={"action_name": ARMS_ACTION})
        wrist_poses = ObsTerm(func=mdp.body_pose_in_root_xyzw, params={"asset_cfg": _WRISTS},
                              noise=Unoise(n_min=-0.005, n_max=0.005))
        wrist_errors = ObsTerm(func=mdp.arm_target_error_in_root, params={"command_name": ARM_COMMAND},
                               noise=Unoise(n_min=-0.005, n_max=0.005))

    whole: WholeCfg = WholeCfg()
    critic: MarlObservationsCfg.CriticCfg = MarlObservationsCfg.CriticCfg()
    odometry: MarlObservationsCfg.OdometryCfg = MarlObservationsCfg.OdometryCfg()


@configclass
class WholeRewardsCfg:
    """Filled by WholeBodyRewardsCfg."""


@configclass
class WholeBodyRewardsCfg:
    """r = r(T^goal) + r(v^cmd) + r_U^shaping + r_L^shaping: every term of both agents' own rewards, once, and none of
    the shared copies. Logged as Episode_Reward/whole/legs_<term> and Episode_Reward/whole/arms_<term>."""

    whole: WholeRewardsCfg = WholeRewardsCfg()

    def __post_init__(self):
        for prefix, rewards in (("legs", LegsRewardsCfg()), ("arms", ArmsRewardsCfg())):
            for name, term in vars(rewards).items():
                if not isinstance(term, RewTerm):
                    continue
                if term.params.get("agent") is not None:
                    term = term.replace(params={**term.params, "agent": "whole"})
                setattr(self.whole, f"{prefix}_{name}", term)


@configclass
class LocoManipWholeBodyEnvCfg(LocoManipMarlEnvCfg):
    """The two-agent task with one agent driving all 26 joints through both action terms."""

    possible_agents = ["whole"]
    action_spaces = {"whole": 26}
    observation_spaces = {"whole": 1}
    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()
    agent_action_terms: dict[str, list[str]] = {"whole": [LEGS_ACTION, ARMS_ACTION]}
