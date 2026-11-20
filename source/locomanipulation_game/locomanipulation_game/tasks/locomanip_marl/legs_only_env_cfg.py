"""The legs alone walking on sport mode's velocity commands: LocoManip-LegsOnly-v0.

The walking task (locomanip_marl_env_cfg.py) with no arm goals and the arms held at their rest pose by IK alone
(the arms' residual 0), the legs' actor without the wrist targets (ee_targets: the rest pose here, always) or the
arm-goal flag (always 0 here; the critic's state drops it too), and on top of its legs terms:
- the feet and knee bands on the lateral component of each pair's offset in the pelvis heading frame instead of its
  horizontal distance, and a standing band on the feet's fore-aft offset, so the stance's width and stagger are
  scored apart. Against sport mode's bags (lateral in the levelled pelvis frame): feet lateral (0.28, 0.45), stepping
  p1 0.267, p99 0.441, standing 0.297-0.338; knees lateral, KNEE_DISTANCE_RANGE (0.286, 0.376), stepping p1 0.284,
  p99 0.394; feet fore-aft while standing is commanded |dx| <= 0.05, the bags' standing p99 0.047;
- a standing band on the feet's load split, 1 up to 60/40;
- the mechanical-power penalty on (energy, -1e-3 per W) and the pelvis-height band off (LegsEnergy200k_s42's recipe:
  it tracks sport mode's turns as well as with the band, in a deeper crouch).

Why each is there: with the total feet distance banded, the standing stance turned from side by side to one foot
ahead at a constant 0.40 m (LegsPass_s42, with and without a LAFAN1 style reward, ~120k steps) and walking lost its step-through
20-40k steps later; with the stance scored apart (LegsSquare_s42) it held to the end. With the style reward (an adversarial motion
prior on LAFAN1 walking, in every run up to LegsEnergyDefault200k_s43; removed 2026-11-19), the standing stance went lopsided by 160k (the pelvis rolled ~17 deg toward a bent left knee, the left foot carrying 64-66%
of the load), though every term but the style paid less for it. The load band fixed that: LegsBalanceAmp_s42 (200k
steps, this task under its earlier name LocoManip-LegsBalanceAmp-v0) stands at 48-53% and is GOLEM lmg_deploy's
policies/h12.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, KNEE_LINK_NAMES

from . import mdp
from .locomanip_marl_env_cfg import (FEET_DISTANCE_RANGE, KNEE_DISTANCE_RANGE, LegsRewardsCfg, LocoManipMarlEnvCfg,
                                     MarlRewardsCfg)

_FEET = SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True)
STANDING_STAGGER_MAX = 0.05  # m


@configclass
class LegsOnlyLegsRewardsCfg(LegsRewardsCfg):
    # the walking task's bands at the same ranges and weights, on the lateral width alone
    feet_distance = RewTerm(func=mdp.body_pair_offset_band, weight=0.5, params={
        "asset_cfg": _FEET, "axis": 1, "min_dist": FEET_DISTANCE_RANGE[0], "max_dist": FEET_DISTANCE_RANGE[1]})
    knee_distance = RewTerm(func=mdp.body_pair_offset_band, weight=0.5, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=KNEE_LINK_NAMES, preserve_order=True), "axis": 1,
        "min_dist": KNEE_DISTANCE_RANGE[0], "max_dist": KNEE_DISTANCE_RANGE[1]})
    # standing, the feet side by side: at most 1 per second, 0.37 of it at 6 cm of stagger, 0.05 at 8 cm
    feet_stagger_standing = RewTerm(func=mdp.body_pair_offset_band, weight=1.0, params={
        "asset_cfg": _FEET, "axis": 0, "min_dist": 0.0, "max_dist": STANDING_STAGGER_MAX,
        "command_name": "base_velocity"})
    # standing, the load split evenly: at most 1 per second, as feet_stagger_standing
    feet_load_balance_standing = RewTerm(func=mdp.feet_load_balance_band, weight=1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES, preserve_order=True),
        "max_imbalance": 0.1, "command_name": "base_velocity"})


def set_legs_task(cfg):
    """The legs' own settings on cfg, whose rewards.legs is a LegsOnlyLegsRewardsCfg (this task's, and
    LocoManip-Marl-Direct-v0's, marl_direct_env_cfg.py)."""
    # the wrist targets never leave the rest pose: the legs' actor doesn't see them (the critic's state keeps them)
    cfg.observations.legs.ee_targets = None
    # no task is an arm goal: the flag that says so is always 0 (the arms' actor keeps it: lmg_deploy's layout)
    cfg.observations.legs.arm_goal = None
    cfg.observations.critic.arm_goal = None
    # LegsEnergy200k_s42's recipe: no pelvis-height band, the legs' mechanical power penalised
    cfg.rewards.legs.base_height.weight = 0.0
    cfg.rewards.legs.energy.weight = -1e-3


@configclass
class LegsOnlyRewardsCfg(MarlRewardsCfg):
    legs: LegsOnlyLegsRewardsCfg = LegsOnlyLegsRewardsCfg()


@configclass
class LegsOnlyEnvCfg(LocoManipMarlEnvCfg):
    rewards: LegsOnlyRewardsCfg = LegsOnlyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # no arm goals; the arms at their rest pose by IK alone
        self.commands.arm_targets.arm_goal_prob = 0.0
        self.actions.arm_pos.residual_gain = 0.0
        set_legs_task(self)
