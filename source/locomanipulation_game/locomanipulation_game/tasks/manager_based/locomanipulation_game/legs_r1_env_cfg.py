"""Round 1: the legs retrained against the frozen upper-body adversary."""

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import LOWER_BODY_JOINTS

from . import mdp
from .common.scenes import CurriculumCfg, TerrainSceneCfg
from .legs_r0_env_cfg import (
    CommandsCfg,
    EventCfg,
    LowerRewardsCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from .upper_adv_r1_env_cfg import LocoManipulationUpperAdvR1EnvCfg

ADV_POLICY_PATH = os.environ.get(
    "ADV_POLICY_PATH", mdp.latest_export("locoManipulation_upper_adv_r1")
)

# Only arm_pos and observations.policy are read. The nested legs term inside
# AdvActionsCfg is never built, so no second policy is loaded.
ADV_CFG = LocoManipulationUpperAdvR1EnvCfg()


@configclass
class LegsR1ActionsCfg:
    # Named joint_pos, not legs_pos: ankle_action_rate_l2 resolves
    # get_term("joint_pos") and looks the ankles up in its _joint_names. With
    # the 12 leg joints under that name the r0 reward set works unmodified.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LOWER_BODY_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    arms = mdp.FrozenPolicyActionCfg(
        asset_name="robot",
        policy_path=ADV_POLICY_PATH,
        low_level_decimation=4,
        low_level_actions=ADV_CFG.actions.arm_pos,
        low_level_observations=ADV_CFG.observations.policy,
    )


@configclass
class LocoManipulationLegsR1EnvCfg(ManagerBasedRLEnvCfg):
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: LegsR1ActionsCfg = LegsR1ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: LowerRewardsCfg = LowerRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.decimation * self.sim.dt
        # Continuing from a policy that already cleared the curriculum, so start
        # spread across all ten levels rather than back at flat.
        self.scene.terrain.max_init_terrain_level = None
        self.viewer.eye = (6.0, 6.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
