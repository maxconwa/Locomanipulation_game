"""Two agents, one H1-2: the legs walk on a velocity command as Unitree's sport mode does, the arms track a wrist pose
each. MAPPO via skrl.

At every command event an env gets navigation (a velocity command, each wrist holding its rest pose) or an arm goal
(zero velocity, a wrist pose per arm fixed in the world; mdp.ArmTargetsCommand). Arm goals start once the robot walks
like sport mode (LocoManipMarlEnv._sport_gate). The policies see the arm command in the pelvis frame, moved by the
pelvis's motion as the robot's FAST-LIO odometry moves it; rewards score the world target, with kernels that narrow
as the errors fall.

Sport mode (assets/h1_2.py, docs/experiments/sportmode_ablations.md): its PD gains on every joint with action scales
matched to them, its stance (knee 0.70 rad) held by a posture term, its command ranges (vx 0.55, vy 0.54 m/s, wz 0.97
rad/s), its 0.5 s command low-pass and 0.70 s stride, stepping on any command over 0.03. The policy runs at 100 Hz
over 500 Hz physics with an explicit PD (the torque computed every physics step and clipped at the effort limit) and
0-20 ms of action delay. Joint targets are clipped inside GOLEM's position e-stop range and bounded by PD torque;
GOLEM's e-stops are checked and charged (golem_safety.py). Domain randomization: motor gains and strength (drawn at
reset and every 20-30 s), friction, torso mass, the IMU's mount and bias, observation noise and pushes on a
curriculum. Episodes have drawn lengths (exponential, mean 90 s), so no network can learn when its episode ends.
"""

import math
import os

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
    ANKLE_JOINT_NAMES,
    ARM_JOINT_NAMES,
    ARM_LINK_NAMES,
    ARMS_RESIDUAL_SCALE,
    COLLISION_LINK_NAMES,
    FINGER_LINK_NAMES,
    FOOT_LINK_NAMES,
    HIP_YAW_ROLL_JOINT_NAMES,
    KNEE_LINK_NAMES,
    LEGS_ACTION_SCALE,
    LOWER_JOINT_NAMES,
    LOWER_LINK_NAMES,
    PELVIS_LINK_NAME,
    STANCE_PELVIS_HEIGHT,
    TABLE_PELVIS_HEIGHT,
)

from . import mdp
from .golem_safety import GOLEM_TARGET_CLIP, GolemEstopCfg, position_envelope
from .scene_cfg import SELF_CONTACT_LINK_NAMES, SELF_CONTACT_SENSOR_NAMES, TerrainSceneCfg

LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[7:]
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"
ARM_COMMAND = "arm_targets"
TERRAIN_MAX_TRACKING_ERROR = 0.4  # m/s, an episode's mean navigation velocity error for a terrain row or push level up
PUSH_LEVELS = 10  # the push curriculum's levels: level L pushes at up to L / PUSH_LEVELS x MAX_PUSH_SPEED
MAX_PUSH_SPEED = 1.0  # m/s, ALMI-Open's fixed push

# Each agent is charged for the self-contact pairs that include one of its links.
LEGS_OWN_LINKS = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES
ARMS_OWN_LINKS = ARM_LINK_NAMES + FINGER_LINK_NAMES

LEGS_ACTION = "joint_pos"
ARMS_ACTION = "arm_pos"
# agent -> the action terms it drives. In possible_agents order the terms follow MarlActionsCfg's order: the env
# concatenates the agents' actions into the ActionManager's vector.
AGENT_ACTION_TERMS = {"legs": [LEGS_ACTION], "arms": [ARMS_ACTION]}

SIM_DT = 0.002            # physics 500 Hz, as ALMI-Open
DECIMATION = 5            # policy 100 Hz, as sport mode
# Each env's joint targets reach the robot this many physics substeps late (0-20 ms), drawn at every reset
ACTION_DELAY_RANGE = (0, 10)
GOAL_BONUS = 5.0          # paid once per reached goal (sync_derived sets the per-second weight)
# The arm goals' tables (ArmTargetsCommandCfg.table_file), built at the table stance (TABLE_PELVIS_HEIGHT)
ARM_TARGET_TABLE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                                 "arm_target_tables_v3.pt"))
# lambda: the legs' reward includes this share of the arms' command-following terms (the arms take none of the
# legs': their cycle-mean yaw term diluted the arms' own)
REWARD_SHARE_TO_LEGS = 0.5
ARM_GOAL_HIP_POS_SCALE = 0.2  # the legs' hip yaw/roll deviation term during arm goals, as a share of its weight
CLIP_ACTIONS = 10.0       # the env clamps every raw policy action to +-this
TORQUE_HEADROOM = 0.85    # targets ask for at most this share of a joint's effort limit (the e-stop trips at 0.9)
KNEE_VALGUS_MARGIN = 0.02  # m: each knee may sit this far inside its hip-ankle line before knee_valgus charges
# mdp.golem_limits: the measured joints are charged within GOLEM_POSITION_MARGIN (rad) of GOLEM's position e-stops,
# and for |dq| or the motor torque beyond GOLEM_RATE_MARGIN of its velocity or torque e-stop
GOLEM_POSITION_MARGIN = 0.03
# the knees get a wider one: deep squats pressed the right knee past its 2.19 rad e-stop
GOLEM_KNEE_POSITION_MARGIN = 0.08
# and the ankle rolls: an earlier policy was e-stopped on the robot turning in place, its right ankle roll pushed by
# the robot's weight 21 mm past a target at the clip. Charged from 0.06 rad inside, targets clipped 0.04 inside.
GOLEM_ANKLE_ROLL_POSITION_MARGIN = 0.06
GOLEM_ANKLE_ROLL_TARGET_MARGIN = 0.04
GOLEM_RATE_MARGIN = 0.8
# The arm-goal kernel curriculum (ArmTargetsCommand._judge_kernels): the fine wrist-position and the wrist-rotation
# kernels' stds per stage, narrowed as the arm goals' errors fall.
FINE_POS_STDS = (0.10, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012)  # m
ROT_STDS = (0.5, 0.4, 0.3, 0.22, 0.16, 0.12, 0.08)               # rad
# The MID-360 IMU (scene.imu at livox_imu_link). Its gyro is low-passed against the head's 20 Hz resonance, as GOLEM's
# deploy does; its mount and accelerometer bias are drawn per env at each reset.
GYRO_LOWPASS_HZ = 8.0
IMU_MOUNT_PITCH_DEG = 1.5
IMU_MOUNT_ROLL_DEG = 0.75
IMU_MOUNT_YAW_DEG = 0.75
IMU_ACC_BIAS_G = 0.01
# The velocity-tracking kernel curriculum (ModalVelocityCommand._judge_kernels): stage 0 is unitree_rl_gym's std 0.5,
# under which standing still against a 0.4 m/s command keeps 53% of the linear reward; narrowed as the errors fall.
LIN_VEL_STDS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)  # m/s
# the yaw kernel starts wide, so the gait's twist still leaves it a slope
ANG_VEL_STDS = (1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2)  # rad/s
# knee_valgus's bodies: hip, knee, ankle per leg, left then right
VALGUS_LINK_NAMES = [f"{side}_{link}_link" for side in ("left", "right") for link in ("hip_roll", "knee", "ankle_pitch")]
# posture's joints: the sagittal ones, which set the stance height (hip yaw and roll have hip_pos)
POSTURE_JOINT_NAMES = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ("hip_pitch", "knee", "ankle_pitch")]
# the motors' gain and strength draws (mdp.randomize_actuator_scales)
ACTUATOR_SCALES = {"gain_range": (0.8, 1.2), "strength_range": (0.85, 1.15)}


@configclass
class MarlCommandsCfg:
    # first: arm_targets zeroes and resamples it
    base_velocity = mdp.ModalVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # in training the navigation events resample it (nav_time_range)
        rel_standing_envs=0.05,
        heading_command=False,  # a yaw rate, like a joystick
        debug_vis=True,
        # sport mode's full stick
        ranges=mdp.ModalVelocityCommandCfg.Ranges(lin_vel_x=(-0.55, 0.55), lin_vel_y=(-0.54, 0.54),
                                                  ang_vel_z=(-0.97, 0.97)),
        arm_command_name=ARM_COMMAND,
        # uniform sampling almost never draws a turn in place, a pure sideways walk or a command near the gate
        pure_turn_prob=0.2,
        pure_lateral_prob=0.2,
        small_command_prob=0.2,
        lin_vel_stds=LIN_VEL_STDS,
        ang_vel_stds=ANG_VEL_STDS,
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
        standing_height=TABLE_PELVIS_HEIGHT,
        resampling_time_range=(4.0, 4.0),
        fine_pos_stds=FINE_POS_STDS,
        rot_stds=ROT_STDS,
        table_file=ARM_TARGET_TABLE,
        debug_vis=True,
    )


@configclass
class MarlActionsCfg:
    # Named joint_pos so mdp.ankle_action_rate_l2 finds the legs' term, and first: that reward indexes the whole
    # action vector by the term's joint order.
    joint_pos = mdp.BoundedJointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_JOINT_NAMES,
        scale=LEGS_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        target_margin=-GOLEM_TARGET_CLIP,
        torque_headroom=TORQUE_HEADROOM,
    )
    arm_pos = mdp.IKResidualArmActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=ARMS_RESIDUAL_SCALE,
        use_default_offset=False,
        preserve_order=True,
        body_names=[LEFT_EE_BODY, RIGHT_EE_BODY],
        arm_joint_names=[LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES],
        command_name=ARM_COMMAND,
        residual_cutoff_hz=3.0,
        target_margin=-GOLEM_TARGET_CLIP,
        torque_headroom=TORQUE_HEADROOM,
    )


def _imu(func, noise: float) -> ObsTerm:
    return ObsTerm(func=func, params={"asset_cfg": SceneEntityCfg("imu")}, noise=Unoise(n_min=-noise, n_max=noise))


def _joints(func, noise: float) -> ObsTerm:
    return ObsTerm(
        func=func,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
        noise=Unoise(n_min=-noise, n_max=noise),
    )


_WRISTS = SceneEntityCfg("robot", body_names=[LEFT_EE_BODY, RIGHT_EE_BODY], preserve_order=True)


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

    legs: LegsCfg = LegsCfg()
    arms: ArmsCfg = ArmsCfg()
    critic: CriticCfg = CriticCfg()


_SELF_CONTACT_PARAMS = {
    "sensor_names": SELF_CONTACT_SENSOR_NAMES,
    "sensor_link_names": SELF_CONTACT_LINK_NAMES,
    "filter_link_names": COLLISION_LINK_NAMES,
    "threshold": 0.1,
}
_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES)
_FEET = SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES)
_LEGS = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)


def _golem(joint_names: list[str], quantity: str, weight: float, margin: float = 0.0,
           joint_margins: dict[str, float] | None = None) -> RewTerm:
    """mdp.golem_limits on joint_names (the agent's own joints)."""
    return RewTerm(func=mdp.golem_limits, weight=weight, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names, preserve_order=True),
        "quantity": quantity, "margin": margin, "joint_margins": joint_margins})


@configclass
class LegsRewardsCfg:
    """The legs' reward: ALMI-Open's h1_2_lower terms (gait clock, feet, effort and smoothness), unitree_rl_gym's
    H1-2 tracking (per step, on the gait-cycle mean velocity, kernels narrowed by the command's curriculum, at twice
    its weights) and alive term, and the task's own: sport mode's stance (base_height and posture outside arm goals,
    whose crouch is free), a steady torso, the swing clearance, the knees and hips, GOLEM's limits, and the yaw rate
    and hip deviation during arm goals. Weights are per second (the manager multiplies by dt)."""

    # --- task
    track_lin_vel_xy = RewTerm(func=mdp.track_velocity_cycle_exp, weight=2.0,
                               params={"command_name": "base_velocity", "component": "lin_xy"})
    track_ang_vel_z = RewTerm(func=mdp.track_velocity_cycle_exp, weight=1.5,
                              params={"command_name": "base_velocity", "component": "ang_z"})
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    # --- posture: sport mode's torso is steady (roll/pitch sd 0.83/0.75 deg while stepping) and its stance tall.
    # orientation -3 (ALMI-Open's -0.25 charged a 3.4 deg sway about 0.001 per second; at -10 the first sport-mode run
    # fell in 40% of its episodes at 24k steps' worth of time, against 11% before)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(func=mdp.base_height_l2_navigation, weight=-10.0, params={
        "target_height": STANCE_PELVIS_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner"),
        "arm_command_name": ARM_COMMAND})
    # --- gait (ALMI-Open)
    contact = RewTerm(func=mdp.contact_matches_phase, weight=0.18,
                      params={"sensor_cfg": _FEET_CONTACT, "command_name": "base_velocity"})
    contact_no_vel = RewTerm(func=mdp.contact_no_vel, weight=-0.2, params={"sensor_cfg": _FEET_CONTACT, "asset_cfg": _FEET})
    feet_distance = RewTerm(func=mdp.body_pair_distance, weight=1.0,
                            params={"asset_cfg": _FEET, "min_dist": 0.3, "max_dist": 0.6})
    knee_distance = RewTerm(func=mdp.body_pair_distance, weight=0.2, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=KNEE_LINK_NAMES), "min_dist": 0.3, "max_dist": 0.3})
    # hip yaw and roll at their defaults, scaled down during arm goals: they hold the pelvis's yaw against twisting
    hip_pos = RewTerm(func=mdp.joint_deviation_l2_modal, weight=-1.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_ROLL_JOINT_NAMES), "arm_command_name": ARM_COMMAND,
        "arm_goal_scale": ARM_GOAL_HIP_POS_SCALE})
    stance_base_vel = RewTerm(func=mdp.stance_base_vel, weight=-1.0, params={"command_name": "base_velocity"})
    # --- effort and smoothness. The rate terms are on raw actions: x4 for 100 Hz (a smooth trajectory's per-step
    # change halves) and x the mean squared ratio of the matched scales to the earlier ones (0.62), so a joint-target
    # change costs what it did
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": _LEGS})
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": _LEGS})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _LEGS})
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.025, params={"action_name": LEGS_ACTION})
    ankle_torque = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)})
    ankle_action_rate = RewTerm(func=mdp.ankle_action_rate_l2, weight=-0.05,
                                params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINT_NAMES)})
    # --- safety
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _LEGS})
    feet_contact_forces = RewTerm(func=mdp.feet_contact_forces, weight=-0.01,
                                  params={"sensor_cfg": _FEET_CONTACT, "max_force": 1400.0, "window": 3})
    self_collision = RewTerm(func=mdp.self_contacts_involving, weight=-1.0,
                             params={**_SELF_CONTACT_PARAMS, "own_links": LEGS_OWN_LINKS})
    # --- the task's own
    arm_goal_yaw_rate = RewTerm(func=mdp.yaw_rate_l2_during_arm_goal, weight=-1.0,
                                params={"arm_command_name": ARM_COMMAND})
    action_beyond_clip = RewTerm(func=mdp.action_beyond_clip, weight=-0.02,
                                 params={"agent": "legs", "action_name": LEGS_ACTION, "clip": CLIP_ACTIONS})
    # a dragging foot pays; sport mode lifts its feet 6 cm
    feet_swing_clearance = RewTerm(func=mdp.feet_swing_clearance, weight=-25.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True),
        "rest_height": 0.045,  # the ankle's standing height
        "lift_height": 0.06,
        "command_name": "base_velocity",
        "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
    })
    knee_valgus = RewTerm(func=mdp.knee_valgus, weight=-10.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=VALGUS_LINK_NAMES, preserve_order=True),
        "margin": KNEE_VALGUS_MARGIN})
    golem_position = _golem(LOWER_JOINT_NAMES, "position", -40.0, GOLEM_POSITION_MARGIN,
                            {".*_knee_joint": GOLEM_KNEE_POSITION_MARGIN,
                             ".*_ankle_roll_joint": GOLEM_ANKLE_ROLL_POSITION_MARGIN})
    golem_velocity = _golem(LOWER_JOINT_NAMES, "velocity", -10.0, GOLEM_RATE_MARGIN)
    golem_torque = _golem(LOWER_JOINT_NAMES, "torque", -10.0, GOLEM_RATE_MARGIN)
    yaw_twist = RewTerm(func=mdp.yaw_twist_l2, weight=-1.0)
    # sport mode's stance: the sagittal joints' gait-cycle mean at the default pose, outside arm goals
    posture = RewTerm(func=mdp.joint_mean_deviation_l2, weight=-2.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=POSTURE_JOINT_NAMES, preserve_order=True),
        "arm_command_name": ARM_COMMAND})


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
    # GOAL_BONUS per goal (sync_derived sets the per-second weight)
    goal_reached = RewTerm(func=mdp.arm_goal_reached, weight=GOAL_BONUS, params={"command_name": ARM_COMMAND})
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
    # the residual's and the arm targets' rates, against reach jitter: -0.1 and -20 at 50 Hz halved the settled
    # wrists' jitter. The residual's -0.2 at 100 Hz: noise (an untrained policy's actions) costs the same per step
    # at any rate, a smooth motion 4x less per step; -0.89 (x4, x2.2 for the matched scales) charged the arms'
    # exploration about 8 per second, their reward sat on its floor of 0, and they never learned to hold the rest pose
    action_rate = RewTerm(func=mdp.action_term_rate_l2, weight=-0.2, params={"action_name": ARMS_ACTION})
    target_rate = RewTerm(func=mdp.arm_joint_target_rate_l2, weight=-80.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
        "arm_command_name": ARM_COMMAND})
    dof_pos_limits = _arm_joints(mdp.joint_pos_limits, -5.0)
    self_collision = RewTerm(
        func=mdp.self_contacts_involving, weight=-1.0, params={**_SELF_CONTACT_PARAMS, "own_links": ARMS_OWN_LINKS}
    )


# The arms' command-following terms, which the legs share in.
ARM_TRACKING_TERMS = ["left_ee_pos", "left_ee_pos_fine", "left_ee_quat",
                      "right_ee_pos", "right_ee_pos_fine", "right_ee_quat", "goal_reached"]


@configclass
class MarlRewardsCfg:
    legs: LegsRewardsCfg = LegsRewardsCfg()
    arms: ArmsRewardsCfg = ArmsRewardsCfg()

    def __post_init__(self):
        # logged as Episode_Reward/legs/arms_<term>
        for name in ARM_TRACKING_TERMS:
            term: RewTerm = getattr(self.arms, name)
            setattr(self.legs, f"arms_{name}", term.replace(weight=REWARD_SHARE_TO_LEGS * term.weight))


@configclass
class MarlEventCfg:
    # friction (ALMI-Open's range), torso mass
    physics_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.1, 1.25),
        "dynamic_friction_range": (0.1, 1.25),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
        "make_consistent": True,  # dynamic <= static
    })
    add_torso_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), "mass_distribution_params": (-2.0, 4.0),
        "operation": "add"})
    # resets on the ground, near still, joints near default
    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                       "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-math.pi, math.pi)},
        "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                           "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
    })
    reset_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True),
        "position_range": (-0.05, 0.05), "velocity_range": (-0.1, 0.1)})
    # ALMI-Open's pushes, scaled per env by the push curriculum
    push_robot = EventTerm(func=mdp.push_by_level, mode="interval", interval_range_s=(10.0, 10.0),
                           params={"num_levels": PUSH_LEVELS, "max_speed": MAX_PUSH_SPEED})
    # the MID-360 chip's mount, beyond what CL_Assets measured, and its accelerometer's horizontal bias
    imu_mount = EventTerm(func=mdp.randomize_imu_mount, mode="reset", params={
        "roll_deg": IMU_MOUNT_ROLL_DEG, "pitch_deg": IMU_MOUNT_PITCH_DEG, "yaw_deg": IMU_MOUNT_YAW_DEG,
        "acc_bias_g": IMU_ACC_BIAS_G})
    # the motors' kp/kd and strength per env and joint at each reset, and again every 20-30 s (sync_derived gives the
    # redraw actuator_scales' params, so an override of those reaches both)
    actuator_scales = EventTerm(func=mdp.randomize_actuator_scales, mode="reset", params=dict(ACTUATOR_SCALES))
    actuator_scales_redraw = EventTerm(func=mdp.randomize_actuator_scales, mode="interval",
                                       interval_range_s=(20.0, 30.0), params=dict(ACTUATOR_SCALES))
    # frames without an inertial in the URDF, which the USD import gave 1 kg each
    massless_frames = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["head_camera_link", "imu_link", "livox_link", "logo_link"]),
                "mass_distribution_params": (0.01, 0.01), "operation": "abs"},
    )


@configclass
class MarlTerminationsCfg:
    # each episode's own drawn length (episode_length_mean_s), at most episode_length_s
    time_out = DoneTerm(func=mdp.episode_limit_reached, time_out=True)
    # unitree_rl_gym's: pelvis contact, |roll| > 0.8 or |pitch| > 1.0 rad
    fell = DoneTerm(func=mdp.pelvis_contact_or_tilt, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "max_roll": 0.8, "max_pitch": 1.0})
    terrain_out_of_bounds = DoneTerm(func=mdp.terrain_out_of_bounds, params={"distance_buffer": 3.0}, time_out=True)
    golem_estop = DoneTerm(func=mdp.golem_estop)


@configclass
class MarlCurriculumCfg:
    # down a row after a fall, up one after a time-out that tracked the velocity commands, rows above 0 only once
    # navigation is robust and the crouch has emerged
    terrain_levels = CurrTerm(func=mdp.gated_terrain_levels, params={
        "command_name": "base_velocity", "max_error": TERRAIN_MAX_TRACKING_ERROR, "arm_command_name": ARM_COMMAND,
        "max_fall_share": 0.02, "min_tracked_share": 0.7, "min_depth_level": 4.0, "min_crouch_ratio": 0.75,
        "steps_per_row": 16000,
    })
    # down after a fall, up per 20 s of a time-out that tracked the velocity commands
    push_levels = CurrTerm(func=mdp.push_levels, params={
        "num_levels": PUSH_LEVELS, "max_speed": MAX_PUSH_SPEED, "command_name": "base_velocity",
        "max_error": TERRAIN_MAX_TRACKING_ERROR, "seconds_per_level": 20.0,
    })
    arm_target_levels = CurrTerm(func=mdp.arm_target_levels, params={"command_name": ARM_COMMAND})
    velocity_kernels = CurrTerm(func=mdp.velocity_kernels, params={"command_name": "base_velocity"})


@configclass
class LocoManipMarlEnvCfg(DirectMARLEnvCfg):
    decimation = DECIMATION
    episode_length_s = 300.0
    """s: the longest episode; each one's length is drawn (episode_length_mean_s)."""
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
    curriculum: MarlCurriculumCfg = MarlCurriculumCfg()
    # GOLEM's e-stops. Every joint target is clipped 0.02 rad inside GOLEM's edges (the ankle rolls 0.04); trips are
    # logged under Safety/ and cost golem_estop_penalty on each trip's first step once the arm goals have started,
    # without ending the episode (ending it stalled training: most episodes ended on a trip)
    golem_estop: GolemEstopCfg = GolemEstopCfg(
        position_margin=0.02,
        joint_position_margins={".*_ankle_roll_joint": (GOLEM_ANKLE_ROLL_TARGET_MARGIN, GOLEM_ANKLE_ROLL_TARGET_MARGIN)},
    )
    golem_estop_penalty: float = -1.0
    gyro_lowpass_hz: float = GYRO_LOWPASS_HZ
    """Hz: the policy's gyro low-pass (imu_filter.GyroLowpass, as lmg_deploy filters /livox/imu)."""

    episode_length_mean_s: float = 90.0
    """s: each episode's length is drawn at its reset from an exponential with this mean, at least
    episode_length_min_s and at most episode_length_s (LocoManipMarlEnv.episode_limit). Memoryless, so no policy or
    value can learn when its episode will end (the robot's never does). 0: every episode episode_length_s."""
    episode_length_min_s: float = 2.0

    # -- training
    training: bool = True
    """Training: the warm start holds the arm goals back until the robot walks, and the curriculum state is saved
    under <log_dir>/curriculum. False (evaluation): arm goals from the start."""
    state_checkpoint_path: str | None = None
    """A saved curriculum state to start from (the scripts fill it in from an skrl checkpoint)."""
    state_save_every: int = 9600
    """Policy steps between curriculum-state saves: the skrl checkpoint interval."""
    # While training, episodes are navigation only until the robot walks like sport mode: the share of ending
    # episodes that fell (exponential mean, warm_start_episode_decay per episode) at most warm_start_max_fall_share
    # and the sport-mode metrics on training's own navigation (ModalVelocityCommand.sport_metrics) within these
    # thresholds, warm_start_sport_checks checks in a row warm_start_sport_every steps apart; or at
    # warm_start_deadline_steps (0: none). <log_dir>/warm_start_gate.yaml overrides any warm_start_* field while
    # training (LocoManipMarlEnv._read_gate_overrides).
    warm_start_max_fall_share: float = 0.08
    warm_start_episode_decay: float = 0.999
    warm_start_min_turn_sense: float = 0.95
    warm_start_speed_ratio_range: tuple[float, float] = (0.85, 1.15)
    warm_start_turn_ratio_range: tuple[float, float] = (0.75, 1.25)
    warm_start_min_sport_cos: float = 0.9
    warm_start_max_ang_err: float = 0.1
    warm_start_max_lin_err: float = 0.12
    warm_start_sport_every: int = 100
    warm_start_sport_checks: int = 3
    warm_start_deadline_steps: int = 240000

    agent_action_terms: dict[str, list[str]] = AGENT_ACTION_TERMS
    clip_actions: float = CLIP_ACTIONS
    # added after the rewards are floored at 0 on terminating (not timed-out) steps, i.e. falls: with less, falling
    # was the cheap way out of a hard reach
    termination_penalty: float = -20.0
    # soft limits as a share of the hard range, over the asset's 0.9: the knee and ankle pitch may use more of their
    # range before dof_pos_limits charges
    soft_joint_pos_limit_factors: dict[str, float] = {".*_knee_joint": 0.95, ".*_ankle_pitch_joint": 0.95}
    action_delay_range: tuple[int, int] | None = ACTION_DELAY_RANGE
    """Physics substeps each env's joint targets reach the robot late, drawn at each reset; None: on time."""
    gait_period: float = 0.7
    """s per full stride of the gait clock (mdp.leg_phase), sport mode's at every speed; the gait-cycle means of the
    tracking terms, the posture term and the velocity command's metrics span one."""
    zero_cmd_eps: float = 0.03
    """The stand/walk gate: a command whose norm over (vx, vy, wz) is under this counts as standing (gait phase 0,
    stance_base_vel); sport mode steps on any input."""

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def sync_derived(self):
        """What follows from other fields, for their values after the command line's overrides (__post_init__ runs
        before them, so train.py and the env call this again): the physics-rate settings, the goal bonus per second,
        and the motors' redraw params (actuator_scales')."""
        self.sync_physics_rate()
        self.rewards.arms.goal_reached.weight = GOAL_BONUS / self.policy_dt
        self.rewards.legs.arms_goal_reached.weight = REWARD_SHARE_TO_LEGS * GOAL_BONUS / self.policy_dt
        self.events.actuator_scales_redraw.params = dict(self.events.actuator_scales.params)

    def sync_physics_rate(self):
        """The sensor periods and render interval for sim.dt and decimation."""
        self.sim.render_interval = self.decimation
        # contacts every physics step (the gait-phase terms read them), the IMU too (its finite-difference lin_acc
        # divides by the physics dt), the height scan once per policy step
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
        self.sync_derived()
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # every joint target inside GOLEM's position e-stop range (written here so params/env.yaml records it)
        envelope = {name: list(bounds) for name, bounds in position_envelope(self.golem_estop).items()}
        self.actions.joint_pos.joint_limits = envelope
        self.actions.arm_pos.joint_limits = envelope
