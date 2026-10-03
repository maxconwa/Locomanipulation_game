"""Two agents, one H1-2: the legs track (vx, vy, yaw rate), the arms track a wrist pose each.

Trained together with IPPO (skrl). The scene, events, terminations, curriculum
and the legs' reward set are the IBR game's, imported from the manager-based
task, so the two setups stay comparable. What is new here:

  * commands: two ReachablePoseCommand terms, one per wrist, in the pelvis frame
  * actions:  both agents' joint targets in one ActionManager (legs first)
  * observations: one group per agent, plus a privileged `critic` group that
    becomes the env state and feeds both critics
  * rewards: one reward set per agent

torso_joint belongs to neither agent: its actuator holds it at default, as in
the IBR rounds.
"""

from isaaclab.envs import DirectMARLEnvCfg
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
    LOWER_JOINT_NAMES,
    LOWER_LINK_NAMES,
    PELVIS_LINK_NAME,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.common.reward_cfg import LowerRewardsCfg
from locomanipulation_game.tasks.manager_based.locomanipulation_game.common.scenes import (
    SELF_CONTACT_LINK_NAMES,
    SELF_CONTACT_SENSOR_NAMES,
    CurriculumCfg,
    TerrainSceneCfg,
)
from locomanipulation_game.tasks.manager_based.locomanipulation_game.legs_r0_env_cfg import (
    CommandsCfg,
    EventCfg,
    TerminationsCfg,
)

from . import mdp

LEFT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_JOINT_NAMES = ARM_JOINT_NAMES[7:]
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"

# Each agent is charged for the self-contact pairs that include one of its links.
LEGS_OWN_LINKS = [PELVIS_LINK_NAME] + LOWER_LINK_NAMES
ARMS_OWN_LINKS = ARM_LINK_NAMES + FINGER_LINK_NAMES

# agent -> the action term it drives. possible_agents order must match the
# term order in MarlActionsCfg: the env concatenates actions in that order.
AGENT_ACTION_TERMS = {"legs": "joint_pos", "arms": "arm_pos"}

EE_POS_STD_COARSE = 0.25  # m
EE_POS_STD_FINE = 0.05    # m
EE_QUAT_STD = 0.5         # rad


@configclass
class MarlCommandsCfg(CommandsCfg):
    # base_velocity comes from the IBR rounds unchanged
    left_ee_pose = mdp.ReachablePoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_EE_BODY,
        joint_names=LEFT_ARM_JOINT_NAMES,
        collision_body_names=["left_(shoulder|elbow|wrist)_.*", "lg_.*"],
        resampling_time_range=(3.0, 6.0),
        debug_vis=True,
    )
    right_ee_pose = mdp.ReachablePoseCommandCfg(
        asset_name="robot",
        body_name=RIGHT_EE_BODY,
        joint_names=RIGHT_ARM_JOINT_NAMES,
        collision_body_names=["right_(shoulder|elbow|wrist)_.*", "rg_.*"],
        resampling_time_range=(3.0, 6.0),
        debug_vis=True,
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
        """What both actors see: the IBR policy observation plus both wrist targets."""

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
        left_ee_target = ObsTerm(func=mdp.pose_command_xyzw, params={"command_name": "left_ee_pose"})
        right_ee_target = ObsTerm(func=mdp.pose_command_xyzw, params={"command_name": "right_ee_pose"})
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


@configclass
class LegsRewardsCfg(LowerRewardsCfg):
    """The IBR legs reward, with the two terms that would otherwise see the arms."""

    def __post_init__(self):
        # action_rate_l2 differences the whole action vector, arms included
        self.action_rate.func = mdp.action_term_rate_l2
        self.action_rate.params = {"action_name": AGENT_ACTION_TERMS["legs"]}
        self.self_collision.func = mdp.self_contacts_involving
        self.self_collision.params = {**_SELF_CONTACT_PARAMS, "own_links": LEGS_OWN_LINKS}


def _ee_terms(command_name: str, body_name: str) -> dict[str, RewTerm]:
    asset_cfg = SceneEntityCfg("robot", body_names=body_name)
    return {
        "pos": RewTerm(
            func=mdp.track_ee_pos_exp,
            weight=1.0,
            params={"command_name": command_name, "asset_cfg": asset_cfg, "std": EE_POS_STD_COARSE},
        ),
        "pos_fine": RewTerm(
            func=mdp.track_ee_pos_exp,
            weight=1.0,
            params={"command_name": command_name, "asset_cfg": asset_cfg, "std": EE_POS_STD_FINE},
        ),
        "quat": RewTerm(
            func=mdp.track_ee_quat_exp,
            weight=1.0,
            params={"command_name": command_name, "asset_cfg": asset_cfg, "std": EE_QUAT_STD},
        ),
    }


_LEFT = _ee_terms("left_ee_pose", LEFT_EE_BODY)
_RIGHT = _ee_terms("right_ee_pose", RIGHT_EE_BODY)


@configclass
class ArmsRewardsCfg:
    # --- task: exp kernels, so the per-step total stays positive ---
    left_ee_pos = _LEFT["pos"]
    left_ee_pos_fine = _LEFT["pos_fine"]
    left_ee_quat = _LEFT["quat"]
    right_ee_pos = _RIGHT["pos"]
    right_ee_pos_fine = _RIGHT["pos_fine"]
    right_ee_quat = _RIGHT["quat"]

    # --- shared survival: the legs' alive bonus (the fall penalty is the env's termination_penalty) ---
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

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


@configclass
class MarlRewardsCfg:
    legs: LegsRewardsCfg = LegsRewardsCfg()
    arms: ArmsRewardsCfg = ArmsRewardsCfg()


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
    curriculum: CurriculumCfg = CurriculumCfg()

    # agent -> the action term it drives (see AGENT_ACTION_TERMS)
    agent_action_terms: dict[str, str] = AGENT_ACTION_TERMS
    # Same as the IBR rounds' rsl_rl clip_actions.
    clip_actions: float = 10.0
    # Per-agent floor on the summed step reward, like PositiveRewardRLEnv.
    # None disables it.
    reward_clip_min: dict[str, float | None] = {"legs": 0.0, "arms": 0.0}
    # Added after the clip on terminating (not timed-out) steps, so the floor
    # can't cancel it. The legs keep IBR's zero (ALMI has no terminal penalty).
    termination_penalty: dict[str, float] = {"legs": 0.0, "arms": -5.0}

    def __post_init__(self):
        # Contacts every physics step (the phase-contact and swing-height terms
        # depend on it); height scan and IMU once per policy step.
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
