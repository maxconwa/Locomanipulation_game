from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES, ALL_JOINTS_NAMES

from . import mdp
from .common.scenes import TerrainSceneCfg
from .common.reward_cfg import LowerRewardsCfg
from .legs_r0_env_cfg import (
    CommandsCfg,
    EventCfg,
    LocoManipulationLegsR0EnvCfg,
    TerminationsCfg,
)
LEGS_CFG = LocoManipulationLegsR0EnvCfg()


@configclass
class AdvActionsCfg:
    # 14. The frozen term is action_dim 0, so the whole agent vector lands here.
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    opponent = mdp.FrozenPolicyActionCfg(
        asset_name="robot",
        low_level_decimation=4,
        low_level_actions=LEGS_CFG.actions.joint_pos,
        low_level_observations=LEGS_CFG.observations.policy,
    )


@configclass
class AdvObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 390. No height_scan: the adversary attacks the body, not the ground.
        # gait_phase stays -- it says when a foot is about to leave the floor.
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=3,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=3,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=3,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=3,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=3,
            flatten_history_dim=True,
        )
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action, history_length=3, flatten_history_dim=True)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AdvRewardsCfg(LowerRewardsCfg):
    """-1 x the legs round's reward, plus the adversary's own effort costs."""
    def __post_init__(self):
        inherited = set(LowerRewardsCfg.__dataclass_fields__)
        for name, term in self.__dict__.items():
            if name in inherited and isinstance(term, RewTerm):
                term.weight = -term.weight
        self.action_rate.weight = 0.0
        self.ankle_action_rate.weight = 0.0



@configclass
class LocoManipulationUpperAdvRiEnvCfg(ManagerBasedRLEnvCfg):
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AdvObservationsCfg = AdvObservationsCfg()
    actions: AdvActionsCfg = AdvActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: AdvRewardsCfg = AdvRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.decimation * self.sim.dt
        # No curriculum term, so levels never move. None spreads envs across all
        # ten at init, matching the distribution the legs policy finished on.
        self.scene.terrain.max_init_terrain_level = None
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
