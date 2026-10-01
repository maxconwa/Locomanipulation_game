from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES

from . import mdp
from .common.scenes import TerrainSceneCfg
from .common.reward_cfg import LowerRewardsCfg
from .common.observations import ObservationsCfg
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
    observations: ObservationsCfg = ObservationsCfg()
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
