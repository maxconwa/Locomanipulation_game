"""Two agents, one H1-2 on rough ground: the legs track a velocity command, the arms a wrist pose each. MAPPO via skrl.

At every command event an env gets navigation (a velocity command, each wrist holding its rest pose) or an arm goal
(zero velocity, a wrist pose per arm fixed in the world; mdp.ArmTargetsCommand). Arm goals start once most episodes
survive (warm_start_timeout_share). Goal heights are uniform between a fixed z_max and a z_min that a depth
curriculum lowers; nothing rewards a pelvis height, so a crouch has to emerge from the legs' share of the arms' reward. The policies see the arm
command in the pelvis frame, moved by a learned pelvis-motion estimate as on the robot (trained beside the policies
on the motion during arm goals); rewards score the true target, with fine kernels that narrow as the error falls
(ArmTargetsCommand's kernel curriculum).

Joint targets are clipped 0.02 rad inside GOLEM's edges (its URDF command clip) and bounded by PD torque; GOLEM's e-stops
are checked and logged (golem_safety.py). Each agent pays for its measured joints nearing GOLEM's e-stops
(mdp.golem_limits), which the clip alone can't prevent; the arms also for their speed and effort.
torso_joint belongs to neither agent: its actuator holds it at default.

As ALMI-Open trains its H1-2 policies: explicit PD (the torque computed every physics step and clipped at the
effort limit, IdealPDActuatorCfg, not PhysX's implicit drive; at 200 Hz, where ALMI-Open's runs at 500), joint targets 0-2 policy steps late
(action_delay_range), fixed observation scales in place of running normalization (observation_scales), its friction,
mass, centre-of-mass and push randomization (base_cfg.EventCfg), and its rough terrain with a terrain-level curriculum.
Unlike ALMI-Open, the pushes ramp up per env on a push-level curriculum, and both an env's push level and its terrain
row go down after a fall and up after an episode that tracked its velocity commands (mdp.push_levels,
mdp.terrain_levels_tracking); the terrain not by distance walked.
"""

import math

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
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

from . import mdp
from .base_cfg import (
    SELF_CONTACT_LINK_NAMES,
    SELF_CONTACT_SENSOR_NAMES,
    EventCfg,
    LowerRewardsCfg,
    TerminationsCfg,
    TerrainSceneCfg,
)
from .golem_safety import GOLEM_TARGET_CLIP, GolemEstopCfg, position_envelope
from .odometry import PelvisEstimatorCfg

LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[7:]
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"
ARM_COMMAND = "arm_targets"
TERRAIN_MAX_TRACKING_ERROR = 0.4  # m/s, an episode's mean navigation velocity error for a terrain row or push level up
PUSH_LEVELS = 10  # the push curriculum's levels: level L pushes at up to L / PUSH_LEVELS x MAX_PUSH_SPEED
MAX_PUSH_SPEED = 1.0  # m/s, ALMI-Open's fixed push
ODOMETRY_HISTORY = 25  # frames in the estimator's window (0.5 s); every run up to golemlim_zrange_200k used 4

# Each agent is charged for the self-contact pairs that include one of its links.
LEGS_OWN_LINKS = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES
ARMS_OWN_LINKS = ARM_LINK_NAMES + FINGER_LINK_NAMES

LEGS_ACTION = "joint_pos"
ARMS_ACTION = "arm_pos"
# agent -> the action terms it drives. In possible_agents order the terms follow MarlActionsCfg's order: the env
# concatenates the agents' actions into the ActionManager's vector.
AGENT_ACTION_TERMS = {"legs": [LEGS_ACTION], "arms": [ARMS_ACTION]}

# physics 200 Hz, as every run up to feetodom_knee_200k; almitrain_200k to lstm_run ran ALMI-Open's 500 Hz (0.002, 10)
SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = 0.02          # DECIMATION x SIM_DT; reward weights are per second
# Each env's joint targets reach the robot this many policy steps late, drawn at every reset (ALMI-Open's 0-2)
ACTION_DELAY_RANGE = (0, 2)
# Fixed input scales, ALMI-Open's (legged_gym obs_scales and commands_scale) where it has the term: skrl's scalers are
# frozen at mean 0 and std 1/scale (train.py), in place of running normalization. Terms not listed are scaled by 1.
OBSERVATION_SCALES = {
    "base_ang_vel": 0.25,
    "joint_vel": 0.05,
    "velocity_commands": (2.0, 2.0, 0.25),  # lin x, lin y, yaw rate
    "base_lin_vel": 2.0,                    # critic
    "base_lin_acc": 0.1,                    # ALMI's policies don't see it; gravity -> about 1
}
# Kept implicit: the grippers are locked open by stiff gains that an explicit PD at SIM_DT would not hold stably.
IMPLICIT_ACTUATORS = ("grippers",)
GOAL_BONUS = 5.0          # paid once per reached goal
# lambda: each agent's reward includes this share of the other's command-following terms
REWARD_SHARE = 0.5
ARM_GOAL_HIP_POS_SCALE = 0.2  # the legs' hip yaw/roll deviation term during arm goals, as a share of its weight
CLIP_ACTIONS = 10.0       # the env clamps every raw policy action to +-this
TORQUE_HEADROOM = 0.85    # targets ask for at most this share of a joint's effort limit (the e-stop trips at 0.9)
KNEE_VALGUS_MARGIN = 0.02  # m: each knee may sit this far inside its hip-ankle line before knee_valgus charges
# mdp.golem_limits: the measured joints are charged within GOLEM_POSITION_MARGIN (rad) of GOLEM's position e-stops,
# and for |dq| or the motor torque beyond GOLEM_RATE_MARGIN of its velocity or torque e-stop
GOLEM_POSITION_MARGIN = 0.03
# the knees get a wider one: golemlim_zrange_200k's deep squats still pressed the right knee past its 2.19 rad e-stop
# in 1.2% of arm-goal substeps, the most of any joint
GOLEM_KNEE_POSITION_MARGIN = 0.08
# and the ankle rolls: Explicit500 on the real robot was e-stopped turning in place, its right ankle roll pushed by
# the robot's weight 21 mm past a target at the clip to -0.2623 rad, past GOLEM's -0.2617 (the 10-16 walk bag); in
# Isaac its |ankle roll| p99 is 0.23 rad. Charged from 0.06 rad inside, targets clipped 0.04 inside (GolemEstopCfg).
GOLEM_ANKLE_ROLL_POSITION_MARGIN = 0.06
GOLEM_ANKLE_ROLL_TARGET_MARGIN = 0.04
GOLEM_RATE_MARGIN = 0.8
# The tracking-kernel curriculum (ArmTargetsCommand._judge_kernels): the fine wrist-position and the wrist-rotation
# kernels' stds per stage, narrowed as the arm goals' errors fall. Stage 0 is wider than the paper runs' fixed 0.05 m,
# so the fine term still has a slope at their ~9 cm closest approach.
FINE_POS_STDS = (0.10, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012)  # m
ROT_STDS = (0.5, 0.4, 0.3, 0.22, 0.16, 0.12, 0.08)               # rad
# The MID-360 IMU (base_cfg scene.imu at livox_imu_link). Its gyro is low-passed against the head's 20 Hz resonance
# (the headshake), as GOLEM's deploy does; its mount and accelerometer bias are drawn per env at each reset.
GYRO_LOWPASS_HZ = 8.0
IMU_MOUNT_PITCH_DEG = 1.5
IMU_MOUNT_ROLL_DEG = 0.75
IMU_MOUNT_YAW_DEG = 0.75
IMU_ACC_BIAS_G = 0.01
# The velocity-tracking kernels' curriculum (ModalVelocityCommand._judge_kernels), the same way: stage 0 is
# unitree_rl_gym's std 0.5, under which standing still against a 0.4 m/s command keeps 53% of the linear reward
# (Tune3Track2 and Carry walked at a third of the command or less); narrowed as the navigation errors fall.
LIN_VEL_STDS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)  # m/s
ANG_VEL_STDS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)  # rad/s
# The yaw tracking term and its kernel's judge read the yaw rate averaged over one gait cycle: on the instantaneous
# rate the step's twist (about +-0.45 rad/s RMS) is as large as a turn command, and no run up to SmoothYawNavH turned
# the commanded way much above chance (eval_ablation). Explicit500 tracked on the cycle mean too. 0: instantaneous.
YAW_TRACKING_WINDOW_S = 0.8  # s, legs_rewards.GAIT_PERIOD
# ... and the twist inside the cycle is charged apart (mdp.yaw_twist_l2): about -0.2 per second at SmoothYawNavH's
# 0.42 rad/s RMS, -1.4 at YawMean's 1.2
YAW_TWIST_WEIGHT = -1.0
# knee_valgus's bodies: hip, knee, ankle per leg, left then right
VALGUS_LINK_NAMES = [f"{side}_{link}_link" for side in ("left", "right") for link in ("hip_roll", "knee", "ankle_pitch")]


@configclass
class MarlCommandsCfg:
    # first: arm_targets zeroes and resamples it
    base_velocity = mdp.ModalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # unitree_rl_gym's; Explicit500 and every run before it: 2.5-10 s
        rel_standing_envs=0.05,
        heading_command=False,  # a yaw rate, like a joystick
        debug_vis=True,
        ranges=mdp.ModalVelocityCommandCfg.Ranges(lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)),
        arm_command_name=ARM_COMMAND,
        # uniform sampling almost never draws a turn in place or a pure sideways walk
        pure_turn_prob=0.2,
        pure_turn_speed=(0.2, 0.5),
        pure_lateral_prob=0.2,
        pure_lateral_speed=(0.15, 0.3),
        lin_vel_stds=LIN_VEL_STDS,
        ang_vel_stds=ANG_VEL_STDS,
        yaw_window_s=YAW_TRACKING_WINDOW_S,
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
        fine_pos_stds=FINE_POS_STDS,
        rot_stds=ROT_STDS,
        debug_vis=True,
    )


@configclass
class MarlActionsCfg:
    # Named joint_pos so mdp.ankle_action_rate_l2 (legs_rewards.py) finds the legs' term, and first: that reward
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

        # the accelerometer with its drawn bias, the gyro low-passed as the deploy does, noise after (the env's
        # randomize_imu_mount and gyro_lowpass); projected gravity raw
        base_lin_acc = _imu(mdp.imu_lin_acc_biased, 0.5)
        base_ang_vel = _imu(mdp.imu_ang_vel_lowpass, 0.2)
        projected_gravity = _imu(mdp.imu_projected_gravity, 0.05)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        arm_goal = ObsTerm(func=mdp.arm_goal_active, params={"command_name": ARM_COMMAND})
        ee_targets = ObsTerm(func=mdp.arm_targets_in_root_xyzw, params={"command_name": ARM_COMMAND})
        joint_pos = _joints(mdp.joint_pos_rel, 0.01)
        joint_vel = _joints(mdp.joint_vel_rel, 1.5)
        gait_phase = ObsTerm(func=mdp.gait_phase_sin, params={"command_name": "base_velocity"})
        # with the sin, each leg's phase without ambiguity (the legs' sins alone are negatives of each other)
        gait_phase_cos = ObsTerm(func=mdp.gait_phase_cos, params={"command_name": "base_velocity"})

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

        base_lin_acc = _imu(mdp.imu_lin_acc_biased, 0.05, **_HISTORY)
        base_ang_vel = _imu(mdp.imu_ang_vel_lowpass, 0.02, **_HISTORY)
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


def _golem(joint_names: list[str], quantity: str, weight: float, margin: float = 0.0,
           joint_margins: dict[str, float] | None = None) -> RewTerm:
    """mdp.golem_limits on joint_names (the agent's own joints)."""
    return RewTerm(func=mdp.golem_limits, weight=weight, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True),
        "quantity": quantity, "margin": margin, "joint_margins": joint_margins})


@configclass
class LegsRewardsCfg(LowerRewardsCfg):
    """ALMI's legs reward (base_cfg.LowerRewardsCfg), changed for the shared body.

    Velocity tracking, alive, ang_vel_xy and base_height are unitree_rl_gym's H1-2 terms (per step, std 0.5; alive
    0.15; ang_vel_xy -0.05; pelvis 1.0 m above the ground at -10), with the tracking weights doubled from theirs (lin
    2.0, yaw 1.0; Tune3Track2: at their 1.0 and 0.5 the robot stood rather than walked under the -20 fall penalty, and
    the arm goals' gate stayed shut to 23k steps instead of 17k). base_height counts only while the
    robot learns to walk, the navigation-only warm start (LocoManipMarlEnv._set_walk_stage), so it never holds the
    pelvis against the arm goals' crouch. Explicit500 and every run before it: no base height, tracking with std 0.25 on
    the gait-cycle mean velocity (lin 2.0, yaw 1.5), alive 1.0, ang_vel_xy -0.5.
    No term holds the knees' bend: stand_still is gone. hip_pos (hip yaw and
    roll at their defaults) and arm_goal_yaw_rate, as in run R, hold the pelvis's yaw against twisting; hip_pos is
    scaled to ARM_GOAL_HIP_POS_SCALE during arm goals. knee_valgus charges the knees falling inward, which paper05
    crouched on (hip yaw turned in to its limit, knees 13 cm apart over ankles 35 cm apart; agent_scripts/
    knee_bow_test.py). action_rate and self_collision count the legs only.
    feet_swing_clearance replaces feet_swing_height, which a dragging foot never pays. golem_* charge the measured
    legs near GOLEM's e-stops: the deep squat rested hip yaw and ankle pitch on their stops.
    """

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
            "lift_height": 0.055,  # Livox500's 0.075 (at -25) and Lift55x2's -50 weight both crouched
            "command_name": "base_velocity",
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    knee_valgus = RewTerm(
        func=mdp.knee_valgus,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=VALGUS_LINK_NAMES, preserve_order=True),
                "margin": KNEE_VALGUS_MARGIN},
    )
    golem_position = _golem(LOWER_JOINT_NAMES, "position", -40.0, GOLEM_POSITION_MARGIN,
                            {".*_knee_joint": GOLEM_KNEE_POSITION_MARGIN,
                             ".*_ankle_roll_joint": GOLEM_ANKLE_ROLL_POSITION_MARGIN})
    golem_velocity = _golem(LOWER_JOINT_NAMES, "velocity", -10.0, GOLEM_RATE_MARGIN)
    golem_torque = _golem(LOWER_JOINT_NAMES, "torque", -10.0, GOLEM_RATE_MARGIN)
    yaw_twist = RewTerm(func=mdp.yaw_twist_l2, weight=YAW_TWIST_WEIGHT, params={"window_s": YAW_TRACKING_WINDOW_S})

    def __post_init__(self):
        # LowerRewardsCfg's per-step kernels at twice unitree_rl_gym's H1-2 weights, their stds annealed from its 0.5 by
        # the velocity command's kernel curriculum (LIN_VEL_STDS, ANG_VEL_STDS); the arms' shared copies follow
        self.track_lin_vel_xy.weight = 2.0
        self.track_lin_vel_xy.func = mdp.track_lin_vel_xy_exp_annealed
        self.track_lin_vel_xy.params = {"command_name": "base_velocity"}
        self.track_ang_vel_z.weight = 1.0
        self.track_ang_vel_z.func = mdp.track_ang_vel_z_avg_exp_annealed
        self.track_ang_vel_z.params = {"command_name": "base_velocity", "component": "ang_z",
                                       "window_s": YAW_TRACKING_WINDOW_S}
        self.ang_vel_xy.weight = -0.05
        self.action_rate.func = mdp.action_term_rate_l2
        self.action_rate.params = {"action_name": LEGS_ACTION}
        self.self_collision.func = mdp.self_contacts_involving
        self.self_collision.params = {**_SELF_CONTACT_PARAMS, "own_links": LEGS_OWN_LINKS}
        # outside arm goals only; LocoManipMarlEnv._set_walk_stage turns it off at the end of the warm start unless
        # base_height_during_navigation
        self.base_height.func = mdp.base_height_l2_navigation
        self.base_height.params = {**self.base_height.params, "arm_command_name": ARM_COMMAND}
        self.hip_pos.func = mdp.joint_deviation_l2_modal
        self.hip_pos.params = {**self.hip_pos.params, "arm_command_name": ARM_COMMAND,
                               "arm_goal_scale": ARM_GOAL_HIP_POS_SCALE}


def _arm_tracking_annealed(arm: int, func) -> RewTerm:
    """A tracking term whose std the arm command's kernel curriculum sets (FINE_POS_STDS, ROT_STDS)."""
    return RewTerm(func=func, weight=1.0, params={"command_name": ARM_COMMAND, "arm": arm})


def _arm_tracking(arm: int, func, std: float) -> RewTerm:
    return RewTerm(func=func, weight=1.0, params={"command_name": ARM_COMMAND, "arm": arm, "std": std})


def _arm_joints(func, weight: float) -> RewTerm:
    return RewTerm(func=func, weight=weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)})


@configclass
class ArmsRewardsCfg:
    # exp kernels: the per-step total stays positive. The coarse position kernel is fixed (a slope from afar); the fine
    # position and the rotation kernels narrow with the arm command's kernel curriculum.
    left_ee_pos = _arm_tracking(0, mdp.arm_target_pos_exp, 0.25)
    left_ee_pos_fine = _arm_tracking_annealed(0, mdp.arm_target_pos_exp_annealed)
    left_ee_quat = _arm_tracking_annealed(0, mdp.arm_target_quat_exp_annealed)
    right_ee_pos = _arm_tracking(1, mdp.arm_target_pos_exp, 0.25)
    right_ee_pos_fine = _arm_tracking_annealed(1, mdp.arm_target_pos_exp_annealed)
    right_ee_quat = _arm_tracking_annealed(1, mdp.arm_target_quat_exp_annealed)
    # weights are per second (the manager multiplies by dt), so this is GOAL_BONUS per goal
    goal_reached = RewTerm(func=mdp.arm_goal_reached, weight=GOAL_BONUS / POLICY_DT, params={"command_name": ARM_COMMAND})
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    arm_goal_yaw_rate = RewTerm(
        func=mdp.yaw_rate_l2_during_arm_goal, weight=-2.0, params={"arm_command_name": ARM_COMMAND}
    )
    # speed and effort as shares of GOLEM's velocity and torque e-stops, squared; then the measured arms near them
    effort = _golem(ARM_JOINT_NAMES, "torque_l2", -1.0)
    speed = _golem(ARM_JOINT_NAMES, "velocity_l2", -2.0)
    golem_position = _golem(ARM_JOINT_NAMES, "position", -20.0, GOLEM_POSITION_MARGIN)
    golem_velocity = _golem(ARM_JOINT_NAMES, "velocity", -10.0, GOLEM_RATE_MARGIN)
    golem_torque = _golem(ARM_JOINT_NAMES, "torque", -10.0, GOLEM_RATE_MARGIN)
    dof_acc = _arm_joints(mdp.joint_acc_l2, -2.5e-7)
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.01, params={"action_name": ARMS_ACTION})
    # arm joint targets changing per policy step during arm goals, against the reach jitter (Livox500: -20, with
    # action_rate -0.1); off since Tune3Track2, whose settled wrists still moved ~0.4 m/s (eval_ablation)
    target_rate = RewTerm(func=mdp.arm_joint_target_rate_l2, weight=0.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
        "arm_command_name": ARM_COMMAND})
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
    # ALMI-Open's pushes (base_cfg.EventCfg), scaled per env by the push curriculum
    push_robot = EventTerm(
        func=mdp.push_by_level,
        mode="interval",
        interval_range_s=(10.0, 10.0),
        params={"num_levels": PUSH_LEVELS, "max_speed": MAX_PUSH_SPEED, "fixed": False},
    )
    # the MID-360 chip's mount, beyond what ed592de measured, and its accelerometer's horizontal bias, per env at reset
    imu_mount = EventTerm(func=mdp.randomize_imu_mount, mode="reset", params={
        "roll_deg": IMU_MOUNT_ROLL_DEG, "pitch_deg": IMU_MOUNT_PITCH_DEG, "yaw_deg": IMU_MOUNT_YAW_DEG,
        "acc_bias_g": IMU_ACC_BIAS_G})
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
    # the deepest crouches lower the pelvis about 25 cm; kneeling lowers it about 60
    # unitree_rl_gym's: pelvis contact, |roll| > 0.8 or |pitch| > 1.0 rad. Explicit500 and every run before it:
    # mdp.fell_or_knelt (tilt over 1.0 rad; the pelvis-height check was off in their launch overrides)
    fell = DoneTerm(func=mdp.pelvis_contact_or_tilt, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "max_roll": 0.8, "max_pitch": 1.0})
    golem_estop = DoneTerm(func=mdp.golem_estop)


@configclass
class MarlCurriculumCfg:
    # down a row after a fall, up one after a time-out that tracked the velocity commands (not ALMI-Open's
    # terrain_levels_vel, which demoted almitrain_lstm_200k's robots to the easiest row as they tracked better)
    # gated: rows above 0 only after navigation is robust and the crouch has emerged, one more every steps_per_row
    # steps (Explicit500); not gated, as every run before it
    terrain_levels = CurrTerm(func=mdp.gated_terrain_levels, params={
        "command_name": "base_velocity", "max_error": TERRAIN_MAX_TRACKING_ERROR, "arm_command_name": ARM_COMMAND,
        "gated": False, "max_fall_share": 0.02, "min_tracked_share": 0.7, "min_depth_level": 4.0,
        "min_crouch_ratio": 0.75, "steps_per_row": 8000,
    })
    # as terrain_levels: down after a fall, up after a time-out that tracked the velocity commands
    push_levels = CurrTerm(func=mdp.push_levels, params={
        "num_levels": PUSH_LEVELS, "max_speed": MAX_PUSH_SPEED, "command_name": "base_velocity",
        "max_error": TERRAIN_MAX_TRACKING_ERROR,
    })
    arm_target_levels = CurrTerm(func=mdp.arm_target_levels, params={"command_name": ARM_COMMAND})
    velocity_kernels = CurrTerm(func=mdp.velocity_kernels, params={"command_name": "base_velocity"})


def explicit_pd(actuators: dict) -> dict:
    """The actuators as IdealPDActuatorCfg (torque = kp (target - q) - kd qd, clipped at the effort limit every
    physics step), with the same gains and limits; those in IMPLICIT_ACTUATORS stay as they are."""
    out = {}
    for name, act in actuators.items():
        if name in IMPLICIT_ACTUATORS or not isinstance(act, ImplicitActuatorCfg):
            out[name] = act
            continue
        out[name] = IdealPDActuatorCfg(
            joint_names_expr=act.joint_names_expr,
            effort_limit=act.effort_limit_sim,
            effort_limit_sim=act.effort_limit_sim,  # the torque bounds read it (mdp.actions.torque_bounds)
            velocity_limit=act.velocity_limit_sim,
            velocity_limit_sim=act.velocity_limit_sim,
            stiffness=act.stiffness,
            damping=act.damping,
            armature=act.armature,
            friction=act.friction,
        )
    return out


@configclass
class LocoManipMarlEnvCfg(DirectMARLEnvCfg):
    decimation = DECIMATION  # policy 50 Hz, physics 200 Hz
    episode_length_s = 20.0
    possible_agents = ["legs", "arms"]
    # placeholders: the env sets all three from the managers
    action_spaces = {"legs": 12, "arms": 14}
    observation_spaces = {"legs": 1, "arms": 1}
    state_space = 1

    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, render_interval=decimation)
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    events: MarlEventCfg = MarlEventCfg()
    commands: MarlCommandsCfg = MarlCommandsCfg()
    actions: MarlActionsCfg = MarlActionsCfg()
    observations: MarlObservationsCfg = MarlObservationsCfg()
    rewards: MarlRewardsCfg = MarlRewardsCfg()
    terminations: MarlTerminationsCfg = MarlTerminationsCfg()
    # GOLEM's e-stops. Every joint target is clipped 0.02 rad inside GOLEM's edges (the ankle rolls 0.04), its
    # 0.001 rad URDF command clip (__post_init__ fills the action terms' joint_limits from it); trips are logged under
    # Safety/ and cost golem_estop_penalty on each trip's first step, in every mode (walking included), without ending
    # the episode. Ending it (golem_estop_ends_episode, from the warm start's end) stalled Livox500's first try: 80% of
    # episodes ended on a trip from 5k to 30k steps and the arm depth curriculum stayed at level 0. Explicit500 and
    # every run before it: logged only, no penalty. None: no e-stop checks and no clip past the URDF one.
    golem_estop: GolemEstopCfg | None = GolemEstopCfg(
        position_margin=0.02,
        joint_position_margins={".*_ankle_roll_joint": (GOLEM_ANKLE_ROLL_TARGET_MARGIN, GOLEM_ANKLE_ROLL_TARGET_MARGIN)},
    )
    golem_estop_ends_episode: bool = False
    # added to both agents' rewards after the floor on the step a trip starts (from the end of the warm start), as
    # termination_penalty on a fall's. -20 (Livox500's second try) made falling the cheaper way out: about 1.8 trips per
    # 1000 steps cost -36 an episode against -20 for a fall that ends it, and 90% of episodes ended in falls from 8k to
    # 20k steps. -5 kept UnitreeRew-style runs (alive 0.15) falling in ~91% of episodes for 15k steps after the arm goals
    # started: trips at ~0.02 per step cost more than the floored reward paid; -1 (Tune2Trip1, Tune3Track2) let them
    # recover in ~10k
    golem_estop_penalty: float = -1.0
    # Hz: the policy's gyro low-pass (imu_filter.GyroLowpass, lmg_deploy's from 556a9e7; export.yaml imu.gyro_lowpass_hz
    # tells the deploy to apply it). None: the raw rate, as every run before the 10-19 MID-360 update.
    gyro_lowpass_hz: float | None = GYRO_LOWPASS_HZ
    curriculum: MarlCurriculumCfg = MarlCurriculumCfg()
    # the estimate moves the arm command, while training once the drift gate opens; the estimator fits the pelvis
    # motion during arm goals only; the vertical motion comes from the feet and the tilt from gravity instead
    estimator: PelvisEstimatorCfg = PelvisEstimatorCfg(
        odometry="learned",
        train_on_arm_goals_only=True,
        vertical_from_feet=True,
        tilt_from_gravity=True,
        foot_body_names=FOOT_LINK_NAMES,
    )
    # while training, episodes are navigation only (the robot learning to walk, with base_height) until the robot walks:
    # over the ending episodes (exponential means, warm_start_episode_decay per episode) falls under
    # warm_start_max_fall_share and the share that tracked the velocity commands (a time-out with a mean navigation
    # error under warm_start_max_tracking_error, as the terrain curriculum promotes) over warm_start_min_tracked_share
    # (LocoManipMarlEnv._update_warm_start); then arm goals start and base_height stops. warm_start_max_fall_share None:
    # until warm_start_timeout_share of the episode ends are time-outs (0.8, Explicit500 and every run before it)
    warm_start_timeout_share: float = 0.8
    warm_start_max_fall_share: float | None = 0.02
    warm_start_min_tracked_share: float = 0.7
    warm_start_max_tracking_error: float = TERRAIN_MAX_TRACKING_ERROR
    # and the walking episodes' velocity cos-sim to the command (nav_cos_vel_xy) at least this: standing scores ~0, and
    # UnitreeRew walked at 0.5 by 12k and 0.63 by 20k (UnitreeFall20's gate, without it, opened at 4.6k on a robot that
    # stood, and it never walked)
    warm_start_min_cos: float = 0.5
    # keep the legs' base_height term after the warm start, while navigating (never during arm goals): Smooth's
    # walking pelvis sank 3 -> 11 cm once the goals started and nothing held it
    base_height_during_navigation: bool = False
    warm_start_episode_decay: float = 0.999
    agent_action_terms: dict[str, list[str]] = AGENT_ACTION_TERMS
    clip_actions: float = CLIP_ACTIONS
    # added after the rewards are floored at 0 on terminating (not timed-out) steps, i.e. falls. Every run up to
    # golemlim_zrange_200k used -5, then -20. UnitreeRew's 0 (unitree_rl_gym's) made falling the way out of a reach:
    # the floored reward during arm goals (~0.01-0.03 per step) was under the -5 trip penalty's mean cost (~0.02 per
    # step), and a fall ends both, so the robot fell backwards in ~92% of episodes from the arm goals' start.
    termination_penalty: float = -20.0
    # soft limits as a share of the hard range, over the asset's 0.9: the knee and ankle pitch may use more of their
    # range before dof_pos_limits charges
    soft_joint_pos_limit_factors: dict[str, float] = {".*_knee_joint": 0.95, ".*_ankle_pitch_joint": 0.95}
    # None: targets apply on the step they are computed (every run up to feetodom_knee_200k)
    action_delay_range: tuple[int, int] | None = ACTION_DELAY_RANGE
    # True: action_delay_range counts physics substeps (finer steps); False: policy steps, as ALMI-Open
    action_delay_substeps: bool = False
    # None: train.py leaves skrl's running normalization (every run up to feetodom_knee_200k)
    observation_scales: dict | None = OBSERVATION_SCALES
    # False: PhysX's implicit drive, as every run up to feetodom_knee_200k (LocoManipMarlEnv puts the asset's
    # actuators back before building the scene)
    explicit_pd: bool = True

    def sync_physics_rate(self):
        """The sensor periods and render interval for sim.dt and decimation. __post_init__ runs before the command
        line's overrides, so train.py and the env call this again: 'env.sim.dt=0.002 env.decimation=10' is enough."""
        self.sim.render_interval = self.decimation
        # contacts every physics step (the gait-phase terms and leg odometry read them), the IMU too (its
        # finite-difference lin_acc divides by the physics dt), the height scan once per policy step
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_forces.history_length = self.decimation  # the last policy step's substeps, no older
        self.scene.imu.update_period = self.sim.dt
        # the self-contact sensors once per policy step: they are slow; a contact shorter than that can be missed
        for name in SELF_CONTACT_SENSOR_NAMES:
            sensor = getattr(self.scene, name)
            sensor.update_period = self.decimation * self.sim.dt
            sensor.history_length = 1

    def __post_init__(self):
        self.sync_physics_rate()
        # explicit PD (every run up to feetodom_knee_200k: PhysX's implicit drive)
        self.scene.robot = self.scene.robot.replace(actuators=explicit_pd(self.scene.robot.actuators))
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # every joint target inside GOLEM's position e-stop range (written here so params/env.yaml records it)
        if self.golem_estop is not None:
            envelope = {name: list(bounds) for name, bounds in position_envelope(self.golem_estop).items()}
            self.actions.joint_pos.joint_limits = envelope
            self.actions.arm_pos.joint_limits = envelope


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
