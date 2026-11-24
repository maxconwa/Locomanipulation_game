"""The tracking env ``amass_s1`` trained in, as an Isaac Lab manager-based env cfg.

Transcribed from h12_train (this repository's own code, branch ``supermimic``, commit b175f1c),
``src/h12_train/tasks.py``: ``h12_flat_tracking_env_cfg`` then ``apply_gencoord`` with ``h12_sensor_actor_group``,
over mjlab's ``make_tracking_env_cfg`` (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0,
``src/mjlab/tasks/tracking/tracking_env_cfg.py``) and SuperMimic's ``gencoord_env_cfg`` groups (commit 7e94b07,
Apache-2.0, ``src/supermimic/tasks/env_cfgs.py``). Anchor ``pelvis``; flat ground; events foot friction, torso COM,
encoder bias (no pushes) plus two start-up fixes of main's USD (the massless frames, h12_train's joint ranges); terminations time-out, motion end, anchor height, anchor
tilt; the actor group without corruption (the learner adds the noise).
"""

import copy
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from locomanipulation_game.tasks.locomanip_marl.joint_noise import ENCODER_OFFSET_RAD, JOINT_POS_NOISE, joint_vel_noise
from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import ACTION_DELAY_RANGE, MarlEventCfg

from pretraining.constants import (
    ACTION_SCALE, ANCHOR_BODY, DECIMATION, IMU_BODY, JOINT_NAMES, JOINT_RANGES, SIM_DT, TARGET_LIMITS, TRACKED_BODIES,
)
from pretraining.plants import MAIN
from pretraining.robot import H1_2_PRETRAIN_CFG, MASSLESS_FRAME_FIX, robot_cfg
from pretraining.tasks import events, observations, rewards, terminations
from pretraining.tasks.actions import BiasedJointPositionActionCfg
from pretraining.tasks.commands import MotionCommandCfg
from pretraining.tasks.delay import TargetDelayCfg

DEFAULT_MOTION_DIR = str(Path(__file__).resolve().parents[3] / "data" / "motions" / "h12_amass")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")  # the bodies of h12_train's FOOT_GEOMS sole capsules

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

_JOINTS = SceneEntityCfg("robot", joint_names=list(JOINT_NAMES), preserve_order=True)
_IMU = SceneEntityCfg("robot", body_names=[IMU_BODY])
_MOTION = {"command_name": "motion"}


@configclass
class TrackSceneCfg(InteractiveSceneCfg):
    # MuJoCo's foot geoms take priority, so the contact's friction is the foot's: multiply against 1 here
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
        ),
    )
    robot: ArticulationCfg = H1_2_PRETRAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class CommandsCfg:
    motion = MotionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file=DEFAULT_MOTION_DIR,
        anchor_body_name=ANCHOR_BODY,
        body_names=TRACKED_BODIES,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    # targets shifted by -encoder bias when applied, as mjlab's
    joint_pos = BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(JOINT_NAMES),
        scale=dict(ACTION_SCALE),
        use_default_offset=True,
        # SAFETY: targets are clipped to TARGET_LIMITS (the range golem_safety clips them to) before the encoder-bias
        # shift, so the applied ones can sit up to the bias (<= 0.01 rad) past them, as mjlab's.
        clip=dict(TARGET_LIMITS),
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class ActorCfg(ObsGroup):
        """h12_sensor_actor_group: [projected_gravity(3), base_ang_vel(3), joint_pos(27), joint_vel(27), actions(27)]
        from the torso IMU and the biased encoders."""

        projected_gravity = ObsTerm(func=observations.body_projected_gravity, params={"asset_cfg": _IMU})
        base_ang_vel = ObsTerm(func=observations.body_ang_vel, params={"asset_cfg": _IMU})
        joint_pos = ObsTerm(func=observations.joint_pos_rel_biased, params={"asset_cfg": _JOINTS})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """make_tracking_env_cfg's critic group."""

        command = ObsTerm(func=mdp.generated_commands, params=_MOTION)
        motion_anchor_pos_b = ObsTerm(func=observations.motion_anchor_pos_b, params=_MOTION)
        motion_anchor_ori_b = ObsTerm(func=observations.motion_anchor_ori_b, params=_MOTION)
        body_pos = ObsTerm(func=observations.robot_body_pos_b, params=_MOTION)
        body_ori = ObsTerm(func=observations.robot_body_ori_b, params=_MOTION)
        base_lin_vel = ObsTerm(func=observations.root_lin_vel_b)
        base_ang_vel = ObsTerm(func=observations.root_ang_vel_b)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FullStateCfg(ObsGroup):
        """gencoord_env_cfg's full state: [root_pos - env origin, root_quat, q, root lin vel, root ang vel, dq]."""

        root_pos = ObsTerm(func=observations.root_pos)
        root_ori = ObsTerm(func=observations.root_ori)
        joint_pos = ObsTerm(func=observations.joint_pos, params={"asset_cfg": _JOINTS})
        root_lin_vel = ObsTerm(func=observations.root_lin_vel)
        root_ang_vel = ObsTerm(func=observations.root_ang_vel)
        joint_vel = ObsTerm(func=observations.joint_vel, params={"asset_cfg": _JOINTS})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class GoalCfg(ObsGroup):
        """_goal_obs_group: [command window, anchor rel_pos window (world model only), anchor rel_ori window]."""

        command = ObsTerm(func=observations.motion_command_window, params=_MOTION)
        motion_anchor_pos_b = ObsTerm(func=observations.goal_anchor_pos_b, params=_MOTION)
        motion_anchor_ori_b = ObsTerm(func=observations.goal_anchor_ori_b, params=_MOTION)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    full_state: FullStateCfg = FullStateCfg()
    goal: GoalCfg = GoalCfg()


@configclass
class EventCfg:
    foot_friction = EventTerm(
        func=events.foot_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(FEET)), "friction_range": (0.3, 1.2)},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    encoder_bias = EventTerm(func=events.sample_encoder_bias, mode="startup", params={"bias_range": (-0.01, 0.01)})
    massless_frames = MASSLESS_FRAME_FIX
    # h12_train's MJCF ranges (golem_safety's); main's USD has the shoulder rolls at -0.19 / 0.19 instead of
    # -0.38 / 0.38, which clipped the reference poses at reset
    joint_ranges = EventTerm(
        func=events.joint_ranges,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "ranges": dict(JOINT_RANGES)},
    )


@configclass
class RewardsCfg:
    motion_global_root_pos = RewTerm(
        func=rewards.motion_global_anchor_position_error_exp, weight=0.5, params={**_MOTION, "std": 0.3}
    )
    motion_global_root_ori = RewTerm(
        func=rewards.motion_global_anchor_orientation_error_exp, weight=0.5, params={**_MOTION, "std": 0.4}
    )
    motion_body_pos = RewTerm(
        func=rewards.motion_relative_body_position_error_exp, weight=1.0, params={**_MOTION, "std": 0.3}
    )
    motion_body_ori = RewTerm(
        func=rewards.motion_relative_body_orientation_error_exp, weight=1.0, params={**_MOTION, "std": 0.4}
    )
    motion_body_lin_vel = RewTerm(
        func=rewards.motion_global_body_linear_velocity_error_exp, weight=1.0, params={**_MOTION, "std": 1.0}
    )
    motion_body_ang_vel = RewTerm(
        func=rewards.motion_global_body_angular_velocity_error_exp, weight=1.0, params={**_MOTION, "std": 3.14}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": _JOINTS})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=terminations.motion_end, time_out=True, params=_MOTION)
    anchor_pos = DoneTerm(func=terminations.bad_anchor_pos_z_only, params={**_MOTION, "threshold": 0.25})
    anchor_ori = DoneTerm(
        func=terminations.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), **_MOTION, "threshold": 0.8},
    )


@configclass
class TrackEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrackSceneCfg = TrackSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 10.0
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.viewer.eye = (2.8, 2.8, 1.5)
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "torso_link"


@configclass
class TrackEnvCfg_PLAY(TrackEnvCfg):
    """h12_flat_tracking_env_cfg(play=True): no time-out, every reset at its clip's start, no pose or velocity
    perturbation."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = int(1e9)
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.sampling_mode = "start"


@configclass
class TrackMainEnvCfg(TrackEnvCfg):
    """The tracking env on main's plant (``plants.MAIN``): main's articulation (explicit PD on sport mode's gains,
    its default pose and stance height), 0.002 s x 5 (100 Hz, the clips resampled to one frame per step) and
    ``MAIN.action_scale`` about main's default pose. Rewards, terminations, observations and start-up events as
    ``TrackEnvCfg``'s."""

    plant: str = "main"
    """The ``plants`` name; the runner reads its gains for the estop terms and train.py records it in run.json."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = robot_cfg(MAIN).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = MAIN.decimation
        self.sim.dt = MAIN.sim_dt
        self.sim.render_interval = self.decimation
        self.actions.joint_pos.scale = dict(MAIN.action_scale)


@configclass
class MainFineTuneEventCfg(EventCfg):
    """Main's randomization (``MarlEventCfg``) in place of the tracking env's foot friction and encoder bias: friction
    on every body, torso mass, the motors' gains and strength per reset, and the encoders' offsets per reset."""

    foot_friction = None
    encoder_bias = EventTerm(
        func=events.sample_encoder_offsets, mode="reset", params={"half_widths": dict(ENCODER_OFFSET_RAD)}
    )
    physics_material = copy.deepcopy(MarlEventCfg().physics_material)
    add_torso_mass = copy.deepcopy(MarlEventCfg().add_torso_mass)
    actuator_scales = copy.deepcopy(MarlEventCfg().actuator_scales)


@configclass
class TrackMainFineTuneEnvCfg(TrackMainEnvCfg):
    """``TrackMainEnvCfg`` with main's domain randomization: its events (``MainFineTuneEventCfg``), its joint encoder
    noise on the actor's joint terms (``joint_noise``; the learner's own proprio noise comes on top) and its target
    delay (``ACTION_DELAY_RANGE`` substeps, ``TargetDelay``)."""

    events: MainFineTuneEventCfg = MainFineTuneEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos = TargetDelayCfg(**{
            **{k: getattr(self.actions.joint_pos, k) for k in ("asset_name", "joint_names", "scale",
                                                               "use_default_offset", "clip", "preserve_order")},
            "delay_range": tuple(ACTION_DELAY_RANGE),
        })
        actor = self.observations.actor
        actor.enable_corruption = True
        actor.joint_pos.noise = Unoise(n_min=-JOINT_POS_NOISE, n_max=JOINT_POS_NOISE)
        actor.joint_vel.noise = joint_vel_noise(list(JOINT_NAMES))
