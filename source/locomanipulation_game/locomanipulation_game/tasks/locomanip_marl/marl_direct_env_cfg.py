"""The legs and the arms on one free-standing H1-2, each on its own task: LocoManip-Marl-Direct-v0.

The two agents coexist. Each has exactly the MDP of its own task, and nothing in either's task depends on the other's:
- the legs': LocoManip-LegsOnly-v0's (legs_only_env_cfg.py). The velocity command runs on its own timer, a fresh one
  every drawn 4-8 s as LegsOnly's navigation events draw it, and no arm goal zeroes it; the legs' observation, action
  and reward terms are LegsOnly's, none of them reading the arms' goals;
- the arms': LocoManip-ArmsOnly-v0's (arms_only_env_cfg.py): an arm goal every drawn 4-8 s from the first step, its
  trajectory, gravity offset, integral and residual, and ArmsOnly's observation and reward terms. The pelvis is free
  here, so a goal rides on it (ArmTargetsCommandCfg.pelvis_fixed_goals): the pose drawn is the pelvis-frame command
  for the whole goal, as it is on ArmsOnly's pinned pelvis, whatever the legs do.

On top of its own task's reward each agent gets a share of the other's task reward (lambda_legs, lambda_arms): the
legs lambda_legs times the arms' wrist tracking terms (ARM_TRACKING_TERMS, as the arms weigh them, the goal bonus
included), the arms lambda_arms times the legs' velocity tracking terms (LEG_TRACKING_TERMS). They are logged as
Episode_Reward/legs/arms_<term> and Episode_Reward/arms/legs_<term>. At 0 and 0 the two rewards are exactly LegsOnly's
and ArmsOnly's.

What the two share besides is the robot: the physics and a fall (it ends the episode and costs both
termination_penalty). A GOLEM e-stop trip costs only the agent whose joint tripped. Each actor's joint observation is
all 27 joints, and the critics' state is the whole robot's, as in both tasks.

Dropped from both: arm_goal_yaw_rate (the pelvis's yaw rate during an arm goal), always zero in LegsOnly (no arm goals)
and in ArmsOnly (a pinned pelvis); here it would charge both agents for the legs' turning on a velocity command.

The class extends ArmsOnlyEnvCfg for its arms' settings and their command-line variants (sync_derived), frees the
pelvis and gives the legs LegsOnly's task back.
"""

from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import LEGS_ACTION_SCALE

from .arms_only_env_cfg import ArmsOnlyArmsRewardsCfg, ArmsOnlyEnvCfg
from .legs_only_env_cfg import LegsOnlyLegsRewardsCfg, set_legs_task
from .locomanip_marl_env_cfg import MarlEventCfg, MarlRewardsCfg


# each agent's task terms, which the other shares in (lambda_legs, lambda_arms)
ARM_TRACKING_TERMS = [f"{side}_ee_{kind}" for side in ("left", "right") for kind in (
    "pos", "pos_fine", "quat", "pos_k20", "pos_k5", "pos_k2", "quat_k200", "quat_k50", "pos_linear")] + ["goal_reached"]
LEG_TRACKING_TERMS = ["track_lin_vel_xy", "track_ang_vel_z"]


@configclass
class MarlDirectRewardsCfg(MarlRewardsCfg):
    legs: LegsOnlyLegsRewardsCfg = LegsOnlyLegsRewardsCfg()
    arms: ArmsOnlyArmsRewardsCfg = ArmsOnlyArmsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.legs.arm_goal_yaw_rate = None
        self.arms.arm_goal_yaw_rate = None


@configclass
class MarlDirectEnvCfg(ArmsOnlyEnvCfg):
    rewards: MarlDirectRewardsCfg = MarlDirectRewardsCfg()

    lambda_legs: float = 0.5
    """lambda_L: the legs' reward includes this share of the arms' wrist tracking terms (ARM_TRACKING_TERMS)."""
    lambda_arms: float = 0.5
    """lambda_U: the arms' reward includes this share of the legs' velocity tracking terms (LEG_TRACKING_TERMS)."""

    def sync_derived(self):
        super().sync_derived()
        # after the arms' own weights are set (the goal bonus, ArmsOnly's variants)
        rewards = self.rewards
        for name in ARM_TRACKING_TERMS:
            term = getattr(rewards.arms, name)
            setattr(rewards.legs, f"arms_{name}", term.replace(weight=self.lambda_legs * term.weight))
        for name in LEG_TRACKING_TERMS:
            term = getattr(rewards.legs, name)
            setattr(rewards.arms, f"legs_{name}", term.replace(weight=self.lambda_arms * term.weight))

    def __post_init__(self):
        super().__post_init__()
        # the pelvis free, reset as the walking task's
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.events.reset_base.params["pose_range"] = dict(MarlEventCfg().reset_base.params["pose_range"])
        # -- the legs: LegsOnly's task
        self.actions.joint_pos.scale = dict(LEGS_ACTION_SCALE)
        set_legs_task(self)
        # the velocity command on its own timer, LegsOnly's navigation events'
        arm = self.commands.arm_targets
        self.commands.base_velocity.resampling_time_range = arm.nav_time_range
        # -- the arms: ArmsOnly's task (the super class), its goals on the free pelvis and apart from the legs' task
        arm.independent_of_legs = True
        arm.pelvis_fixed_goals = True
        self.sync_derived()
