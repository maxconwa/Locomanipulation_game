"""Round 0: flat-ground velocity tracking for the lower body, no adversary.

Reward structure follows ALMI-Open's h1_2_lower config. The upper body is held
at its default pose by its actuators, so no frozen policy is needed yet.

Two things are sized for later rounds rather than this one: the observation
covers all 27 body joints, and it includes a height scan. Both are constant
here, but a fixed observation space lets every later round warm-start from the
previous round's checkpoint instead of training from scratch.
"""

import math

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import (
    BODY_JOINTS,
    H1_2_MAGPIE_CFG,
    LOWER_BODY_JOINTS,
)

from . import mdp

# --- body-name patterns (verified against the check_h1_2.py body list) ---
FEET = ".*_ankle_roll_link"
KNEES = ".*_knee_link"
TRUNK = ["pelvis", "torso_link"]

# --- joint-name patterns ---
LEG_ONLY = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
ANKLE_ONLY = [".*_ankle_.*_joint"]
HIP_YAW_ROLL = [".*_hip_yaw_joint", ".*_hip_roll_joint"]

# --- ALMI reward parameters (h1_2_lower_config.py, class rewards) ---
BASE_HEIGHT_TARGET = 0.95
FEET_SWING_HEIGHT = 0.08
MIN_DIST = 0.3
MAX_DIST = 0.6
MAX_CONTACT_FORCE = 700.0
TRACKING_STD = 0.5      # ALMI tracking_sigma = 0.25 = std^2

# Indices into the 12-D action vector, valid because the action term uses
# preserve_order=True with LOWER_BODY_JOINTS ordered hip_yaw, hip_pitch,
# hip_roll, knee, ankle_pitch, ankle_roll (left then right). Matches ALMI's
# [4, 5, 10, 11]. VERIFY against env.action_manager before trusting.
ANKLE_ACTION_IDS = [4, 5, 10, 11]




@configclass
class LocoMotionSceneCfg(InteractiveSceneCfg):
    # Must be named `terrain`: the terrain-level curriculum looks for
    # env.scene.terrain. Round 0 is a plane; later rounds switch
    # terrain_type to "generator" and add a TerrainGeneratorCfg.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    robot: ArticulationCfg = H1_2_MAGPIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # All bodies: the reward terms need feet, knees, pelvis and torso.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )



@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # Some zero-command envs are required, not optional: stand_still,
        # stance_base_vel and the gait clock's standing branch are all
        # inactive without them.
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
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
    # q_target = default_joint_pos + 0.25 * action, legs only (12 joints).
    # Arms, torso and the locked grippers are held at default by their actuators.
    # preserve_order keeps the action vector in LOWER_BODY_JOINTS order rather
    # than PhysX's interleaved order, which is what makes ANKLE_ACTION_IDS and
    # the deployment index map predictable.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )




@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Not measurable on hardware: replace with a state estimate or drop
        # before deployment.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # All 27 body joints, not just the 12 legs: keeps the obs space fixed
        # across IBR rounds so later rounds can warm-start.
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
        # ALMI's gait clock: sin of each leg's phase. The policy needs this to
        # exploit the contact_matches_phase reward, which is otherwise
        # unlearnable -- it would be rewarded on a schedule it cannot observe.
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()




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
        func=mdp.reset_joints_by_offset,   # additive, unlike _by_scale
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
class LowerRewardsCfg:
    """ALMI-Open h1_2_lower reward structure.

    No `termination` term: ALMI's scale is -0.0, inherited and never
    overridden. Also no `feet_air_time`: they set it to 0 (with 10.0 commented
    out) because contact_matches_phase does that job, and running both would
    double-count gait timing with conflicting incentives.
    """

    # --- task ---
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

    # --- posture ---
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": BASE_HEIGHT_TARGET},
    )

    # --- gait shaping (the ALMI-specific part) ---
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
            "max_dist": MAX_DIST / 2,   # ALMI halves it for knees
        },
    )
    hip_pos = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_YAW_ROLL)},
    )

    # --- standing still ---
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

    # --- effort and smoothness (leg joints only: ALMI slices [:, :12]) ---
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    ankle_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_ONLY)},
    )
    ankle_action_rate = RewTerm(
        func=mdp.ankle_action_rate_l2,
        weight=-0.02,
        params={"action_ids": ANKLE_ACTION_IDS},
    )

    # --- safety ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_ONLY)},
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "max_force": MAX_CONTACT_FORCE,
        },
    )




@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ALMI terminates on pelvis contact only (terminate_after_contacts_on =
    # ["pelvis"]) and penalizes hip/knee contact instead. We include torso_link
    # too, since a torso on the ground is unambiguously a fall.
    fell = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TRUNK),
            "threshold": 1.0,
        },
    )


@configclass
class LocoManipulationLegsR0EnvCfg(ManagerBasedRLEnvCfg):
    scene: LocoMotionSceneCfg = LocoMotionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: LowerActionsCfg = LowerActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: LowerRewardsCfg = LowerRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4          # policy 50 Hz, physics 200 Hz
        self.episode_length_s = 60.0*5
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Contacts every physics step (the phase-contact and swing-height terms
        # depend on it); height scan once per policy step.
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (6.0, 6.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)


from isaaclab.envs import ManagerBasedRLEnv
import torch
class PositiveRewardRLEnv(ManagerBasedRLEnv):
    """Clips the summed per-step reward at zero.

    Equivalent to legged_gym's `only_positive_rewards = True`. Without it, a
    policy facing large early penalties learns that terminating quickly beats
    accumulating negative reward, so it falls over on purpose instead of
    learning to walk.

    Clips the TOTAL, not individual terms: the relative weighting between terms
    still shapes behaviour whenever the sum is positive. Per-term TensorBoard
    logging is unaffected, since the reward manager logs before this clamp.
    """

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        clipped = torch.clamp(reward, min=0.0)
        self._n = getattr(self, "_n", 0) + 1
        # if self._n % 20 == 0:
            # print(f"[clip] call {self._n}: raw {reward.mean().item():+.4f} -> {clipped.mean().item():+.4f}")
        self.reward_buf = clipped
        return obs, clipped, terminated, truncated, extras