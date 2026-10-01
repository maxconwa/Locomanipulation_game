"""Round 0: locomotion velocity tracking for the lower body, no adversary.

Reward structure follows ALMI-Open's h1_2_lower config. The upper body is held
at its default pose by its actuators, so no frozen policy is needed yet.

Two things are sized for later rounds rather than this one: the observation
covers all 27 body joints, and it includes a height scan. Both are constant
here, but a fixed observation space lets every later round warm-start from the
previous round's checkpoint instead of training from scratch.
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import LOWER_JOINT_NAMES, ALL_JOINTS_NAMES

from . import mdp
from .common.scenes import TerrainSceneCfg, CurriculumCfg
from .common.reward_cfg import LowerRewardsCfg
from .common.observations import ObservationsCfg


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 10.0),
        # Some zero-command envs are required, not optional: stand_still,
        # stance_base_vel and the gait clock's standing branch are all
        # inactive without them.
        rel_standing_envs=0.05,
        heading_command=False,   # theta is a yaw RATE, like a joystick
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7),   # ALMI ranges
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )

@configclass
class LowerActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
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


@configclass
class LocoManipulationLegsR0EnvCfg(ManagerBasedRLEnvCfg):
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: LowerActionsCfg = LowerActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: LowerRewardsCfg = LowerRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4          # policy 50 Hz, physics 200 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Contacts every physics step (the phase-contact and swing-height terms
        # depend on it); height scan once per policy step.
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (6.0, 6.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.scene.imu.update_period = self.decimation * self.sim.dt
        


