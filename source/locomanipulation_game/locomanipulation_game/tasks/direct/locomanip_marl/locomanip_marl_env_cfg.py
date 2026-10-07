"""Two agents, one H1-2: the legs track (vx, vy, yaw rate), the arms track a wrist pose each.

Trained together with IPPO (skrl). The scene, events, terminations and terrain
curriculum are the IBR game's, imported from the manager-based task, and the
legs start from the IBR legs reward set. What is new here:

  * commands: at every command event an env gets a navigation goal (velocity
    command, arms at rest) OR an arm goal (zero velocity, one wrist pose per
    arm fixed in the world), never both. Arm goals have a reach curriculum
    (mdp.ArmTargetsCommand): they start in front of the robot, widen to the
    whole standing workspace, then drop up to 25 cm below it so the legs must
    crouch. A reached goal is replaced on the spot; episodes don't end on it.
  * the arm command the policies see is always in the pelvis frame: exact at
    the event, then moved each step by the pelvis motion a learned estimator
    reads from the IMU and joint states (odometry.py), as on the robot.
    Rewards score the true world point.
  * actions:  both agents' joint targets in one ActionManager (legs first)
  * observations: one group per agent, a privileged `critic` group that
    becomes the env state and feeds both critics, and the estimator's
    `odometry` group
  * rewards: one reward set per agent, each with 0.1 x the other's
    command-following terms. The legs' base-height target drops with the goal.

torso_joint belongs to neither agent: its actuator holds it at default, as in
the IBR rounds.
"""

import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
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
    CurriculumCfg,
    TerrainSceneCfg,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.legs_r0_env_cfg import (
    CommandsCfg,
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

# agent -> the action term it drives. possible_agents order must match the
# term order in MarlActionsCfg: the env concatenates actions in that order.
AGENT_ACTION_TERMS = {"legs": "joint_pos", "arms": "arm_pos"}

POLICY_DT = 0.02          # decimation 4 x sim dt 0.005; reward weights are per second
EE_POS_STD_COARSE = 0.25  # m
EE_POS_STD_FINE = 0.05    # m
EE_QUAT_STD = 0.5         # rad
GOAL_BONUS = 5.0          # paid once per reached goal
# Each agent's reward includes this fraction of the other's command-following
# terms: the legs get the arms' tracking and goal bonus, the arms the legs' velocity tracking.
# 0.5 for the legs: lowering the pelvis for a low target pays mostly through
# the arms' tracking, and at 0.1 that was ~11% of the legs' reward.
LEGS_SHARE_OF_ARMS = 0.5
ARMS_SHARE_OF_LEGS = 0.1
# Legs gait shaping during arm goals, as a fraction of its IBR weight (see LegsRewardsCfg).
# Run 10 trained with these: falls doubled early on (the legs' std climbed
# 0.58 -> 0.70 with the damping relaxed) but it converged stable, at 0.07-0.13
# falls per env-minute. The legs' std is now capped at 0.6 (skrl config,
# log_std_bounds) against that early excursion.
ARM_GOAL_SHAPING_SCALE = {"lin_vel_z": 0.0, "ang_vel_xy": 0.5, "hip_pos": 0.2}
CLIP_ACTIONS = 10.0       # the env clamps every raw policy action to +-this (IBR's rsl_rl clip_actions)


@configclass
class MarlCommandsCfg(CommandsCfg):
    # base_velocity: the IBR rounds' command, made modal in __post_init__.
    # It must stay first: arm_targets zeroes and resamples it.
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
        # an arm goal's give-up time: one both wrists reach is replaced at once
        resampling_time_range=(4.0, 6.0),
        debug_vis=True,
    )

    def __post_init__(self):
        # same ranges and timing as IBR, but zero while an arm goal is active
        ibr = self.base_velocity
        self.base_velocity = mdp.ModalVelocityCommandCfg(
            asset_name=ibr.asset_name,
            resampling_time_range=ibr.resampling_time_range,
            rel_standing_envs=ibr.rel_standing_envs,
            heading_command=ibr.heading_command,
            debug_vis=ibr.debug_vis,
            ranges=ibr.ranges,
            arm_command_name=ARM_COMMAND,
        )


@configclass
class MarlActionsCfg:
    # Named joint_pos so the IBR reward ankle_action_rate_l2 finds the legs'
    # term. It must stay first: that reward indexes the whole action vector by
    # the term's joint order.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    # Reaching needs larger joint excursions than walking.
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class MarlObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        """What both actors see: the IBR policy observation plus the arm goal."""

        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        arm_goal = ObsTerm(func=mdp.arm_goal_active, params={"command_name": ARM_COMMAND})
        # (x, y, z, qx, qy, qz, qw) per wrist, left then right, in the pelvis
        # frame: the command as the robot would hold it, moved by odometry
        ee_targets = ObsTerm(func=mdp.arm_targets_in_root_xyzw, params={"command_name": ARM_COMMAND})
        # Zeros: crouching should emerge from low targets, not be commanded.
        # Kept as an input so checkpoints from before still load.
        height_drop = ObsTerm(
            func=mdp.arm_target_height_drop, params={"command_name": ARM_COMMAND, "visible": False}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        gait_phase = ObsTerm(func=mdp.gait_phase_sin, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LegsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.last_action, params={"action_name": AGENT_ACTION_TERMS["legs"]})

    @configclass
    class ArmsCfg(ProprioCfg):
        actions = ObsTerm(func=mdp.last_action, params={"action_name": AGENT_ACTION_TERMS["arms"]})

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged, noise-free: the env state. Each critic sees its actor's obs plus this."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        ee_poses = ObsTerm(
            func=mdp.body_pose_in_root_xyzw,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_BODY, RIGHT_EE_BODY], preserve_order=True)},
        )
        # the rewards score the true world point; the actors only see the believed one
        true_ee_targets = ObsTerm(func=mdp.true_arm_targets_in_root_xyzw, params={"command_name": ARM_COMMAND})
        # how low the pelvis is, and how far the goal was lowered: what a crouch is worth
        pelvis_height = ObsTerm(func=mdp.pelvis_height_above_ground)
        target_drop = ObsTerm(func=mdp.arm_target_height_drop, params={"command_name": ARM_COMMAND, "visible": True})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OdometryCfg(ObsGroup):
        """The pelvis-motion estimator's measurements. Not a policy input.

        Sensor-level noise, not the actors' domain-randomization noise: at
        50 Hz a walking foot moves ~1 cm per step, and the actors' +-0.01 rad
        joint noise alone is that big once it goes through the leg (run 2:
        0.28 m/s error, ~0.25 m drift per goal). Assumed sensors: absolute
        joint encoders (~1e-3 rad), differentiated joint velocity, a MEMS IMU,
        motor-current torque (+-2 Nm). Each term keeps the last
        ODOMETRY_HISTORY frames, which end at the step's end: the estimate
        sees t and t+1 and a little before, to filter noise. Leg torques say
        which foot carries the robot.
        """

        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.001, n_max=0.001),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )
        leg_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
            history_length=ODOMETRY_HISTORY, flatten_history_dim=True,
        )

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
    """The IBR legs reward, with five terms changed for the shared body.

    action_rate and self_collision would otherwise see the arms; base_height
    and stand_still apply during navigation only, so nothing holds the legs
    at standing height during an arm goal and a crouch can emerge when low
    targets reward it. track_ang_vel_z is weighted up: yaw tracking earned
    0.48 of its 1.0 in run 5, against 1.65 of 2.0 for xy.

    During arm goals (ARM_GOAL_SHAPING_SCALE) lin_vel_z is off, ang_vel_xy
    halved and hip_pos at 0.2: run 9 measured them as the gait shaping that
    resists the crouch. Kept: torques, dof_pos_limits, feet_contact_forces,
    contact_no_vel, self_collision and the smoothness terms protect the
    hardware, and in a deep crouch they mostly charged falls and hard drops;
    the zero-command velocity terms and arm_goal_yaw_rate ask for standing
    still without turning, which a crouch shouldn't need to break.
    """

    # Heading held through arm goals (see mdp.yaw_rate_l2_during_arm_goal); on
    # top of track_ang_vel_z, which already asks for zero yaw then.
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-1.0, params={"arm_command_name": ARM_COMMAND}
    )
    # Raw leg actions past the clip (see mdp.action_beyond_clip). Per second:
    # an action of 15 costs 0.1, 80 costs 1.4, 400 costs 7.8, against a legs
    # total of ~+6. Run 12 at agent_100800: none in steady state (knee p99 6.6),
    # but in an episode's first 0.5 s ankle pitch passes the clip 5-6% of steps
    # and single actions reach ~100.
    action_beyond_clip = RewTerm(
        func=mdp.action_beyond_clip, weight=-0.02, params={"agent": "legs", "clip": CLIP_ACTIONS}
    )

    def __post_init__(self):
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
        self.track_ang_vel_z.weight = 1.5
        for name, func in (
            ("lin_vel_z", mdp.lin_vel_z_l2_modal),
            ("ang_vel_xy", mdp.ang_vel_xy_l2_modal),
            ("hip_pos", mdp.joint_deviation_l2_modal),
        ):
            term = getattr(self, name)
            term.func = func
            term.params = {
                **term.params,
                "arm_command_name": ARM_COMMAND,
                "arm_goal_scale": ARM_GOAL_SHAPING_SCALE[name],
            }


def _arm_tracking_terms(arm: int) -> dict[str, RewTerm]:
    params = {"command_name": ARM_COMMAND, "arm": arm}
    return {
        "pos": RewTerm(func=mdp.arm_target_pos_exp, weight=1.0, params={**params, "std": EE_POS_STD_COARSE}),
        "pos_fine": RewTerm(func=mdp.arm_target_pos_exp, weight=1.0, params={**params, "std": EE_POS_STD_FINE}),
        "quat": RewTerm(func=mdp.arm_target_quat_exp, weight=1.0, params={**params, "std": EE_QUAT_STD}),
    }


_LEFT = _arm_tracking_terms(0)
_RIGHT = _arm_tracking_terms(1)


@configclass
class ArmsRewardsCfg:
    # --- task: exp kernels, so the per-step total stays positive ---
    left_ee_pos = _LEFT["pos"]
    left_ee_pos_fine = _LEFT["pos_fine"]
    left_ee_quat = _LEFT["quat"]
    right_ee_pos = _RIGHT["pos"]
    right_ee_pos_fine = _RIGHT["pos_fine"]
    right_ee_quat = _RIGHT["quat"]
    # weights are per second (the manager multiplies by dt), so this is GOAL_BONUS per goal
    goal_reached = RewTerm(
        func=mdp.arm_goal_reached, weight=GOAL_BONUS / POLICY_DT, params={"command_name": ARM_COMMAND}
    )

    # --- shared survival: the legs' alive bonus (the fall penalty is the env's termination_penalty) ---
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # --- don't turn the robot while reaching (mdp.yaw_rate_l2_during_arm_goal) ---
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-2.0, params={"arm_command_name": ARM_COMMAND}
    )

    # --- effort and smoothness (arm joints only), the legs' weights ---
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    action_rate = RewTerm(
        func=mdp.action_term_rate_l2, weight=-0.01, params={"action_name": AGENT_ACTION_TERMS["arms"]}
    )

    # --- safety ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    self_collision = RewTerm(
        func=mdp.self_contacts_involving,
        weight=-1.0,
        params={**_SELF_CONTACT_PARAMS, "own_links": ARMS_OWN_LINKS},
    )


# Each agent's command-following terms, which the other agent shares in. Not
# effort or safety terms: an agent can't control the other's joints.
ARM_TRACKING_TERMS = [
    "left_ee_pos", "left_ee_pos_fine", "left_ee_quat",
    "right_ee_pos", "right_ee_pos_fine", "right_ee_quat",
    "goal_reached",
]
LEG_TRACKING_TERMS = ["track_lin_vel_xy", "track_ang_vel_z"]


@configclass
class MarlRewardsCfg:
    legs: LegsRewardsCfg = LegsRewardsCfg()
    arms: ArmsRewardsCfg = ArmsRewardsCfg()

    def __post_init__(self):
        # Copies at the share of the original weight, logged as
        # Episode_Reward/legs/arms_<term> and Episode_Reward/arms/legs_<term>.
        for name in ARM_TRACKING_TERMS:
            term: RewTerm = getattr(self.arms, name)
            setattr(self.legs, f"arms_{name}", term.replace(weight=LEGS_SHARE_OF_ARMS * term.weight))
        for name in LEG_TRACKING_TERMS:
            term = getattr(self.legs, name)
            setattr(self.arms, f"legs_{name}", term.replace(weight=ARMS_SHARE_OF_LEGS * term.weight))


@configclass
class MarlCurriculumCfg(CurriculumCfg):
    # terrain_levels: on how well navigation commands were followed, not on
    # distance from the origin (see mdp.terrain_levels_tracking)
    arm_target_levels = CurrTerm(func=mdp.arm_target_levels, params={"command_name": ARM_COMMAND})

    def __post_init__(self):
        self.terrain_levels = CurrTerm(func=mdp.terrain_levels_tracking, params={"command_name": "base_velocity"})


@configclass
class LocoManipMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 4  # policy 50 Hz, physics 200 Hz
    episode_length_s = 20.0
    possible_agents = ["legs", "arms"]
    # Placeholders: the env replaces all three with the managers' sizes at init.
    action_spaces = {"legs": 12, "arms": 14}
    observation_spaces = {"legs": 1, "arms": 1}
    state_space = 1

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    events: EventCfg = EventCfg()

    # manager configs, built by LocoManipMarlEnv
    commands: MarlCommandsCfg = MarlCommandsCfg()
    actions: MarlActionsCfg = MarlActionsCfg()
    observations: MarlObservationsCfg = MarlObservationsCfg()
    rewards: MarlRewardsCfg = MarlRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: MarlCurriculumCfg = MarlCurriculumCfg()
    # pelvis odometry that keeps the arm command on its world point
    estimator: PelvisEstimatorCfg = PelvisEstimatorCfg()
    # GOLEM's safety-layer e-stops, checked at every physics substep (golem_safety.py); None: not checked
    golem_estop: GolemEstopCfg | None = None
    # Joint targets reach the robot this many physics substeps late, drawn per env and episode from [lo, hi]
    # (5 ms each): the deploy loop's latency. The agents observe their actions undelayed. None: no delay.
    action_delay_substeps: tuple[int, int] | None = None

    # agent -> the action term it drives (see AGENT_ACTION_TERMS)
    agent_action_terms: dict[str, str] = AGENT_ACTION_TERMS
    # The env state (both critics' input under MAPPO): both actors'
    # observations + the privileged critic group, so each critic can see what
    # the other agent saw and did (its last action is in its observation).
    state_includes_agent_obs: bool = True
    # Same as the IBR rounds' rsl_rl clip_actions.
    clip_actions: float = CLIP_ACTIONS
    # Per-agent floor on the summed step reward, like PositiveRewardRLEnv.
    # None disables it.
    reward_clip_min: dict[str, float | None] = {"legs": 0.0, "arms": 0.0}
    # Whether the floor also applies during arm goals. Yes for both: the legs'
    # reward should never be negative, which is what the floor is for (user,
    # 2026-10-05). Runs 10-13 had it off for the legs during arm goals (going
    # down into a crouch measured -2.0/s in run 9, and the floor hid that
    # cost); flat run D then showed the price: early on the legs netted about
    # -0.3/s alive, falling cost about as much as living, and they never
    # learned to stand.
    reward_clip_during_arm_goals: dict[str, bool] = {"legs": True, "arms": True}
    # Added after the clip on terminating (not timed-out) steps, so the floor
    # can't cancel it. The legs' -5 (IBR and ALMI had 0) prices the falls the
    # freer crouch allows: run 9 fell 1.2-2.2 times per env-minute.
    termination_penalty: dict[str, float] = {"legs": -5.0, "arms": -5.0}
    # Per-joint soft limit factors over the asset's 0.9, applied once at init
    # (nothing here rewrites joint limits later). The knee (hard -0.12..2.19)
    # and ankle pitch (hard -0.90..0.52, the CL_Assets URDF, MJCF and USD
    # agree) bound a feet-flat squat: the pelvis goes 0.376 m down at 0.9,
    # 0.395 m at 0.95, so at 0.9 dof_pos_limits charges the deepest squat.
    soft_joint_pos_limit_factors: dict[str, float] = {".*_knee_joint": 0.95, ".*_ankle_pitch_joint": 0.95}

    def __post_init__(self):
        # Contacts every physics step (the phase-contact and swing-height terms
        # depend on it); height scan and IMU once per policy step.
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.decimation * self.sim.dt
        # The 25 filtered self-contact sensors: once per policy step, not every
        # physics step. Env stepping was 91 ms/step (run 5) and these were the
        # IBR prime suspect for slowdown. self_contacts_involving reads the
        # latest reading; a contact shorter than a policy step can be missed.
        for name in SELF_CONTACT_SENSOR_NAMES:
            sensor = getattr(self.scene, name)
            sensor.update_period = self.decimation * self.sim.dt
            sensor.history_length = 1
        if abs(self.decimation * self.sim.dt - POLICY_DT) > 1e-9:
            raise ValueError(f"POLICY_DT ({POLICY_DT}) must equal decimation * sim.dt: the goal bonus is sized by it.")
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)


@configclass
class LocoManipMarlStandEnvCfg(LocoManipMarlEnvCfg):
    """Stand and reach: navigation paused, every goal is an arm goal.

    The experiment for an emergent crouch: the legs only have to keep the
    robot up while the arms chase targets that the drop curriculum lowers
    below the standing workspace.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.arm_targets.arm_goal_prob = 1.0


@configclass
class LocoManipMarlFlatCurriculumEnvCfg(LocoManipMarlEnvCfg):
    """Flat world, fresh policies: walking and crouch-reaching, 50/50 from the start.

    The target behaviour is walk -> stop -> crouch-reach -> stand -> walk on,
    never walking and reaching at once. Each goal event is an arm goal with
    probability 0.5, else navigation with the arms holding the
    zero-joint-angle wrist pose (rest_pose_b) as their end-effector command;
    the reach curriculum (spread, then the 0.5 m drop) runs as in the stand
    task. Each mode switch first settles: 0.75 s stopped before an arm goal,
    1.5 s to stand up before walking.

    Flat run C used ArmTargetsCommand's walking gate instead (navigation only
    until 4 of 5 segments covered 80% of their commanded path): it walked well
    (2% falls, 0.19 m/s error) but covered 69% of the path on average, so only
    1.3% of envs passed by step 65k, and the arms' std had collapsed to its
    floor holding the rest pose. The user chose 50/50 from step 0 instead.

    Also, for this task only:
      * the flat generator (same tile layout as the terrain world, plane only),
        no terrain curriculum;
      * joint targets bounded to the hard limits +- 0.4 rad (no rate caps: flat
        run A had them, the user reverted them for the restart);
      * the policies observe the applied (bounded) action, not the raw one;
      * the IMU updates every physics step: at the policy rate its finite
        difference divided by the physics dt and read lin_acc 4x too large;
      * calmer resets: run 12 had ~2/3 of its falls in an episode's first second.
    """

    def __post_init__(self):
        super().__post_init__()
        # flat world
        self.scene.terrain.terrain_generator = TERRAINS_FLAT_CFG
        self.curriculum.terrain_levels = None
        # IMU at the physics rate (the base class sets the policy rate)
        self.scene.imu.update_period = self.sim.dt
        # calmer resets: on the ground, near still, joints near default
        self.events.reset_base.params = {
            **self.events.reset_base.params,
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        }
        self.events.reset_joints.params = {
            **self.events.reset_joints.params, "position_range": (-0.05, 0.05), "velocity_range": (-0.1, 0.1)
        }
        # walking and arm goals 50/50 from the start, settling at each switch
        arm = self.commands.arm_targets
        arm.walk_gate = False
        arm.arm_goal_prob = 0.5
        arm.settle_to_arm_s = 0.75
        arm.settle_to_nav_s = 1.5
        # bounded joint targets; the policies see the applied action
        self.actions.joint_pos = mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=LOWER_JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
        )
        self.actions.arm_pos = mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
        )
        self.observations.legs.actions.func = mdp.applied_action
        self.observations.arms.actions.func = mdp.applied_action
        # Staying up must pay. Flat run D (legs' reward then not floored during arm goals) netted about
        # -0.3/s alive early on, so the -5 fall cost about as much as living, and after 13.8k steps every
        # episode still fell within ~1.5 s (run C, walk-only, stood by 4k). With the floor now on in both
        # modes, this keeps a margin for living: a constant alive reward only moves the stay-up-vs-fall
        # trade (a 20 s episode is worth +20).
        self.rewards.legs.alive.weight = 1.0


@configclass
class LocoManipMarlFlatIKEnvCfg(LocoManipMarlFlatCurriculumEnvCfg):
    """The flat curriculum with arms that can see their error and reach through IK.

    Run F walked well but its arms stopped at ~12 cm and reached ~15% of goals
    (5 cm / 0.35 rad), so no env ever left reach level 0 and the targets were
    never lowered: no crouch. Three changes, on top of run F's setup:

      * the arms observe their wrist poses and each wrist's error to the
        command, in the pelvis frame (before, only the critic had them);
      * the arm action is one damped-least-squares IK step toward the command
        plus the policy's residual (mdp.IKResidualArmAction, 0.2 rad per unit);
      * the reach curriculum's axes move independently: spread on the reach
        rate as before, drop when the mean closest approach over the last 5
        goals is <= 12 cm (back above 18 cm), so lowered targets don't wait on
        5 cm precision; a lowered target the arms can't reach leaves an error
        only the legs can close.

    Observation sizes: legs 96 (unchanged), arms 98 + 14 + 12 = 124.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_pos = mdp.IKResidualArmActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            scale=0.2,
            use_default_offset=False,
            preserve_order=True,
            body_names=[LEFT_EE_BODY, RIGHT_EE_BODY],
            arm_joint_names=[LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES],
            command_name=ARM_COMMAND,
        )
        # appended after the existing arm terms (the manager reads the group's attributes in order)
        self.observations.arms.wrist_poses = ObsTerm(
            func=mdp.body_pose_in_root_xyzw,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[LEFT_EE_BODY, RIGHT_EE_BODY], preserve_order=True)},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        self.observations.arms.wrist_errors = ObsTerm(
            func=mdp.arm_target_error_in_root,
            params={"command_name": ARM_COMMAND},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        self.commands.arm_targets.drop_promote_error = 0.12
        self.commands.arm_targets.drop_demote_error = 0.18


@configclass
class LocoManipMarlFlatIK2EnvCfg(LocoManipMarlFlatIKEnvCfg):
    """The IK task with the arm goals redrawn: in front, evenly spread, low goals built in a real squat, held.

    Run G (the IK task) crouched and walked, but its sampler starved the
    standing workspace: at drop level 10, 96% of goals were lowered standing
    targets from the bottom 10% of the table, so 3% of targets were above
    1.2 m and the wide standing reach regressed (spread 8, no drop: 36%
    reached, 1.18 falls/env-min). The arms also darted from goal to goal:
    each was replaced 0.2 s after it was reached. Changes, all in the arm
    command (mdp.ArmTargetsCommand):

      * no targets less than 0.1 m forward of the pelvis (47% of the table
        was behind it);
      * every table balanced over 10 cm cells (random joint angles put 29% of
        targets in the outstretched band beside the shoulder, 2.3% in front
        of the chest);
      * half the arm goals from the whole standing table at every drop level,
        half from squat tables: targets reachable collision-free with the legs
        in a feet-flat squat at that depth, both wrists from one depth;
      * goals last 4 s and stay after they are reached (held 1 s inside 5 cm /
        0.35 rad), so staying on target keeps paying;
      * spread moves on standing goals, drop on low goals.

    Observation and action sizes are the IK task's: legs 96, arms 124.
    """

    def __post_init__(self):
        super().__post_init__()
        arm = self.commands.arm_targets
        arm.table_file = "arm_target_tables_v2.pt"
        arm.min_target_x = 0.1
        arm.balance_cell = 0.1
        arm.build_size = 300_000
        arm.squat_tables = True
        arm.foot_body_names = FOOT_LINK_NAMES
        arm.low_goal_prob = 0.5
        arm.judge_axes_separately = True
        arm.resample_on_reach = False
        arm.reach_hold_s = 1.0
        arm.resampling_time_range = (4.0, 4.0)


@configclass
class LocoManipMarlFlatIK3EnvCfg(LocoManipMarlFlatIK2EnvCfg):
    """The IK2 task with arm commands moved by leg kinematics, and the arms' residual low-passed.

    Run H (IK2) held each goal for 4 s, and over that time the learned
    estimator's small per-step errors (0.03 m/s, 0.03 rad/s) moved the arm
    command 10-15 cm off its world point. With the command moved by the true
    pelvis motion instead, run H's agent_91200 reached 94% of goals at
    spread 4 / drop 10 (86% on the estimate) and 64% at 8 / 5 (49%).

    Here the attitude gives the rotation and a planted foot the translation
    (LocoManipMarlEnv._leg_odometry): the ankle's position in the pelvis frame
    is forward kinematics of the joint encoders, and while the foot stays put
    its change is the pelvis's motion. Passively, on run H's agent_91200 at
    spread 4 / drop 10, a target drifted 0.9 cm over a 4 s goal this way (90th
    percentile 2.8 cm), against 3.7 cm (5.6 cm) with the learned estimator.
    During arm goals a foot is planted on 99.5% of steps; the learned
    estimator still trains and covers the rest. With the estimate this close,
    arm goals follow it from the start (no teacher-forcing warmup or ramp);
    the drift gate still holds it back while its drift over whole goals is
    above 5 cm.

    The arms' residual is low-passed at 3 Hz (mdp.IKResidualArmAction): while
    holding a goal, run H's residual chattered at ~12 Hz, every arm joint
    reversing 21-30 times a second (0.4-0.9 rad/s at the median, wrist
    0.32 m/s); with the residual off, IK alone reversed 5-10 times (wrist
    0.10 m/s) but sagged to a 3.7 cm error instead of 1.9 cm.

    Observation and action sizes are the IK task's: legs 96, arms 124.
    """

    def __post_init__(self):
        super().__post_init__()
        self.estimator.odometry = "legs"
        self.estimator.foot_body_names = FOOT_LINK_NAMES
        self.estimator.warmup_steps = 0
        self.estimator.ramp_steps = 0
        self.actions.arm_pos.residual_cutoff_hz = 3.0


@configclass
class LocoManipMarlFlatGolemEnvCfg(LocoManipMarlFlatIK3EnvCfg):
    """The IK3 task under GOLEM's safety layer: targets clipped as it clips them, and its e-stops end the episode.

    GOLEM's h12_safety_layer (relax_safety_split, the preset the real launch uses) clips position targets to the
    URDF range shrunk by 0.001 rad and e-stops the robot when a joint reaches its position limit, its velocity limit
    or its torque limit (golem_safety.py has the table). The sim has the same limits as hard physical stops, so
    run I worked right at them: under GOLEM it would trip 1-62 times per robot-minute on torque (shoulder yaw,
    elbows, ankle pitch), 4-15 on position (ankle roll and pitch, shoulder yaw, elbows) and 0.1-0.4 on velocity
    (knees, shoulder pitch), and 19% of its walking ankle-roll targets and ~20% of its arm-goal wrist-pitch
    targets lay outside GOLEM's clip.

    Here both action terms bound their targets to GOLEM's clip (hard limits shrunk by 0.001; the sim's shoulder
    roll limit is already the stricter one), and the golem_estop termination ends the episode like a fall (the
    -5 termination penalty, a level down for an arm goal) when any joint crosses GOLEM's e-stop tightened by
    GolemEstopCfg's margin: 90% of the velocity and torque thresholds, 0.02 rad inside the position ones.

    A squat needs no target past the knee limit: holding it takes extension torque, so the knee target sits below
    the knee angle (run I's knee targets left GOLEM's clip on 0.01% of steps).

    Observation and action sizes are the IK task's (legs 96, arms 124), so IK3 checkpoints load.
    """

    def __post_init__(self):
        super().__post_init__()
        self.golem_estop = GolemEstopCfg()
        self.terminations.golem_estop = DoneTerm(func=mdp.golem_estop)
        self.actions.joint_pos.target_margin = -GOLEM_TARGET_CLIP
        self.actions.arm_pos.target_margin = -GOLEM_TARGET_CLIP


@configclass
class LocoManipMarlFlatGolem2EnvCfg(LocoManipMarlFlatGolemEnvCfg):
    """The Golem task with each joint target also bounded by torque: the PD torque it asks for at the measured
    state stays within 85% of the joint's effort limit (mdp.actions.torque_bounds).

    Run I under the Golem task tripped GOLEM's e-stops 28.5 times per robot-minute, 80% of them on torque:
    shoulder yaw asked for up to 4.4x its 18 Nm through the arms' residual, and the IK step plus residual can
    put a target 0.3 rad or more from the joint. Run J, trained against the e-stop termination alone, cut the
    per-step trip rate by ~40% in 15k steps, but every e-stop during an arm goal cost a reach level and both
    levels fell from ~6 / ~8 to 0. With the bound, run I trips 15 times per robot-minute instead of 28.5 (torque
    7 / 3 in settles / arm goals instead of 61 / 35); the rest, mostly ankle pitch at its stop in deep squats,
    is left to the termination. Keeping targets 0.03 rad inside the limits instead of GOLEM's 0.001 changed
    nothing (15.25): loads, not targets, put those joints on their stops. 0.85 sits below the termination's
    0.9; the deploy controller applies the same bound at the measured state.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.torque_headroom = 0.85
        self.actions.arm_pos.torque_headroom = 0.85


BODY_JOINTS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint", ".*_shoulder_.*_joint",
               ".*_elbow_joint", ".*_wrist_.*_joint"]
"""The 27 motor joints (not the gripper hinges)."""


@configclass
class LocoManipMarlFlatGolem3EnvCfg(LocoManipMarlFlatGolem2EnvCfg):
    """The Golem2 task with the joints' passive dynamics randomized per robot, toward GOLEM's MuJoCo model.

    GOLEM's RoboCasa robot (CL_Assets robosuite_assets h1_2/robot.xml) gives every joint damping 10, armature
    0.1 and frictionloss 0.2, on top of the PD the policy commands; our asset has armature 0.01 and only the
    PD's damping (2-4). Run K at 14.4k steps, deployed through GOLEM into RoboCasa, lost its balance holding a
    standing reach in open floor and tripped the ankle e-stops in all three games (14-56 s engaged); run I the
    same (13-17 s). Here each robot gets, once at startup, extra passive damping uniform in [0, 10] N m s/rad
    and an armature uniform in [0.01, 0.12] kg m^2 on all 27 motor joints. The extra damping is passive: the
    torque bound and the e-stop monitor keep using the motor's own damping (default_joint_damping).
    """

    def __post_init__(self):
        super().__post_init__()
        joints = SceneEntityCfg("robot", joint_names=BODY_JOINTS)
        self.events.passive_damping = EventTerm(
            func=mdp.randomize_actuator_gains, mode="startup",
            params={"asset_cfg": joints, "damping_distribution_params": (0.0, 10.0), "operation": "add"},
        )
        self.events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters, mode="startup",
            params={"asset_cfg": joints, "armature_distribution_params": (0.01, 0.12), "operation": "abs"},
        )


@configclass
class LocoManipMarlFlatGolem4EnvCfg(LocoManipMarlFlatGolem3EnvCfg):
    """The Golem3 task with GOLEM's RoboCasa joint dynamics, the deploy loop's latency, sharper velocity tracking
    and wider e-stop margins on the joints behind RoboCasa's e-stops.

    GOLEM's sim-to-sim test (tests/locomanipulation_game/sim2sim_walk.py: the deploy controller on RoboCasa's
    compiled MuJoCo scene, without ROS) found, for run M on a fixed walking schedule:
    - RoboCasa's passive joint damping (10 on every joint) is what stops backward walking: path covered 0.02 with
      it, 1.17 without, against Isaac's 1.05-1.10 at nominal joints; contact stiffness, floor friction, the
      friction cone, the PD integration, the timestep and 10-20 ms of delay barely change it. In Isaac the same
      damping and armature 0.1 cut backward walking to 0.37 (run M saw damping only up to 10, rarely there).
      Here every robot gets damping uniform in [8, 12] and armature in [0.08, 0.12], around RoboCasa's.
    - Sideways steps and turning in place are not learned even in Isaac (path covered about 0): with the
      tracking kernel's std at 0.5, standing still under a 0.25 m/s sideways command earns 78% of the tracking
      reward (53% for a 0.4 rad/s turn). At std 0.25 it earns 37% (8%).
    - 20 ms of target delay added a fall and 12 e-stop onsets in MuJoCo (Isaac: 10-20 ms multiplied falls plus
      e-stops by 9-20). Here targets land 0-4 substeps (0-20 ms) late, drawn per episode.
    - Ankle pitch reached its position e-stop 9 times in deep MuJoCo squats, and ankle roll and knee extension
      caused most of RoboCasa's e-stops: their margins go from 0.02 to 0.05 rad (the knee's only at the
      extension limit, so the squat keeps its depth).

    Observation and action sizes are unchanged, so Golem3 checkpoints load. The deploy settings are Golem2's.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.passive_damping.params["damping_distribution_params"] = (8.0, 12.0)
        self.events.joint_armature.params["armature_distribution_params"] = (0.08, 0.12)
        self.action_delay_substeps = (0, 4)
        self.golem_estop = GolemEstopCfg(joint_position_margins={
            ".*_ankle_roll_joint": (0.05, 0.05),
            ".*_ankle_pitch_joint": (0.05, 0.05),
            ".*_knee_joint": (0.05, 0.02),
        })
        for term in (self.rewards.legs.track_lin_vel_xy, self.rewards.legs.track_ang_vel_z,
                     self.rewards.arms.legs_track_lin_vel_xy, self.rewards.arms.legs_track_ang_vel_z):
            term.params["std"] = 0.25


@configclass
class LocoManipMarlFlatGolem5EnvCfg(LocoManipMarlFlatGolem4EnvCfg):
    """The Golem4 task with a swing-height term that a dragging foot can't escape.

    Every run up to M walks with a shuffle: in Isaac and in GOLEM's MuJoCo test alike, the ankle rises about
    1 cm while walking (standing height 0.045 m, walking peak 0.055 m) and the feet stay loaded through most
    of their swing. feet_swing_height charges a foot only while it is off the ground, so dragging it costs
    nothing and a small lift costs more than none; its total was -0.0002 per second at the end of run M.
    Here it is replaced by feet_swing_clearance: during a commanded walk, each foot in its swing phase of
    the gait clock pays for being below a reference that rises 5.5 cm above its standing height and back
    (peak 0.10 m, against the old target's 0.08). At weight -10 a full shuffle costs about 0.5 per second
    of walking; a step that follows the reference costs nothing.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.legs.feet_swing_height = None
        self.rewards.legs.feet_swing_clearance = RewTerm(
            func=mdp.feet_swing_clearance,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True),
                "rest_height": 0.045,
                "lift_height": 0.055,
                "command_name": "base_velocity",
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class LocoManipMarlFlatGolem6EnvCfg(LocoManipMarlFlatGolem5EnvCfg):
    """The Golem5 task with turns in place and sideways walks drawn on purpose.

    Velocity commands are uniform over forward, sideways and yaw, so a turn with almost no linear velocity
    (|v| < 0.1 m/s, |yaw rate| > 0.25 rad/s) is 1.7% of commands and a near-pure sideways walk 1.4%. Run O at
    agent_52800 walks forward and backward at 85-90% of the command in Isaac, but on a 0.4 rad/s turn in place
    it stands still (0.01 rad/s, both feet loaded) and on 0.25 m/s sideways it reaches 0.01-0.09 m/s, though
    turning in place would earn about 1.2 more per second of track_ang_vel_z. Here 20% of the moving commands
    are replaced by a turn in place (|yaw rate| 0.2-0.5 rad/s) and 10% by a sideways walk (|vy| 0.15-0.3 m/s).
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.pure_turn_prob = 0.2
        self.commands.base_velocity.pure_lateral_prob = 0.1
