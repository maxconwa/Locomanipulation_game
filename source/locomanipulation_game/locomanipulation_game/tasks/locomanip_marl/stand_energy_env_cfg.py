"""Standing in place on the least energy: LocoManip-StandEnergy-v0.

The walking task (locomanip_marl_env_cfg.py) with every velocity command zero (rel_standing_envs 1.0, so the gait
clock never runs) and the legs paid only for their electrical power: exp(-P / 100 W), P the mechanical power sum
|tau qd| over the leg joints plus each winding's copper loss, 200 W at the joint's effort limit and quadratic in the
torque (so a pose held still costs the torque it takes). No command following, no sport-mode stance or swing (no
tracking, sport_gait, foot lift, contact clock, feet_pass, feet/knee/pelvis-height bands), no effort terms (energy
replaces them) and no safety terms (GOLEM's limits, joint limits, contact forces, self-contact, knee valgus, the
action clip): only a steady torso (lin_vel_z, ang_vel_xy, orientation) and smooth actions (action rate, ankle action
rate). Falling is the walking task's (pelvis contact or tilt; no minimum height). The arms hold their rest pose by
IK alone: the arm goals' warm-start gate needs walking metrics, so it never opens, and their residual stays at 0.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import LOWER_JOINT_NAMES

from . import mdp
from .locomanip_marl_env_cfg import LegsRewardsCfg, LocoManipMarlEnvCfg, MarlRewardsCfg

# The walking task's legs terms this task keeps: the steady torso and the action rates
KEPT_LEGS_TERMS = ["lin_vel_z", "ang_vel_xy", "orientation", "action_rate", "ankle_action_rate"]


@configclass
class StandEnergyLegsRewardsCfg(LegsRewardsCfg):
    # the only pay: 2 per second at zero power. 200 W per joint at its effort limit is an assumption (no motor
    # constants for the H1-2 at hand): a joint at 30% of its limit heats at 18 W. LegsPass_s42 at 120k stood (MuJoCo,
    # zero command) at 57 W mechanical and 61 W of heat on this model, exp(-1.18) = 0.31
    energy = RewTerm(func=mdp.electrical_power_exp, weight=2.0, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True),
        "heat_at_limit": 200.0, "power_scale": 100.0})

    def __post_init__(self):
        for name, term in list(vars(self).items()):
            if isinstance(term, RewTerm) and name != "energy" and name not in KEPT_LEGS_TERMS:
                setattr(self, name, None)


@configclass
class StandEnergyRewardsCfg(MarlRewardsCfg):
    legs: StandEnergyLegsRewardsCfg = StandEnergyLegsRewardsCfg()


@configclass
class StandEnergyEnvCfg(LocoManipMarlEnvCfg):
    rewards: StandEnergyRewardsCfg = StandEnergyRewardsCfg()

    # configclass's post-init deep-copies every member that isn't a property of the instance's own class (its
    # __dict__), and would try to set the inherited one
    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        # every command zero: stand in place
        self.commands.base_velocity.rel_standing_envs = 1.0
        # no arm goals; the arms at their rest pose by IK alone
        self.commands.arm_targets.arm_goal_prob = 0.0
        self.actions.arm_pos.residual_gain = 0.0
