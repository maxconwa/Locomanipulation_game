"""The walking task with the stance's width and stagger scored apart: LocoManip-LegsSquare-v0.

LegsPass_s42's recipe (the walking task, legs only: no arm goals, the arms at their rest pose by IK) with the feet and
knee bands on the lateral component of each pair's offset in the pelvis heading frame instead of its horizontal
distance, and a standing band on the feet's fore-aft offset. LegsPass_s42 and LegsPassAmp_s42 both lost the
step-through after ~120k steps: the standing stance turned at a near-constant 0.40 m horizontal distance, inside
feet_distance's band, from side by side (feet 0.42 m apart laterally, 5 cm stagger) to one foot ahead (0.19 m and
35 cm by 140k), and walking followed 20-40k steps later (the feet passing in 100% of 0.7 s cycles at 0.4 m/s, 0% by
140k / 180k; the walking width 0.37 -> 0.20 m). Nothing scored the direction of the feet's offset, and standing
earns no gait terms.

The bands, against sport mode's bags (lateral in the levelled pelvis frame):
- feet lateral (0.28, 0.45): stepping p1 0.267, p99 0.441; standing 0.297-0.338.
- knees lateral, KNEE_DISTANCE_RANGE (0.286, 0.376): stepping p1 0.284, p99 0.394.
- feet fore-aft while standing is commanded, |dx| <= 0.05: the bags' standing p99 0.047.

LegsSquare_s42 (200k steps) held the square stance and the step-through to the end. LocoManip-LegsSquareAmp-v0
(LegsSquareAmpEnvCfg): the same plus LegsPassAmp_s42's style reward (amp_walk_env_cfg.AMP_STYLE, LAFAN1 discriminator,
weight 3.5, ramped in over 20k-60k steps).

LegsSquareAmp200k_s42 stood lopsided by 160k: the pelvis rolled ~17 deg toward a bent left knee (1.10 vs 0.43 rad),
the left foot carrying 64-66% of the load (43% at 20k), though in Isaac every term but the style paid ~1/s less for
it than for the 20k stance and the run's own discriminator scored it lower too. LocoManip-LegsBalanceAmp-v0
(LegsBalanceAmpEnvCfg): LegsSquareAmp plus a standing band on the feet's load split, 1 up to 60/40.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, KNEE_LINK_NAMES

from . import mdp
from .amp_walk_env_cfg import AMP_STYLE
from .locomanip_marl_env_cfg import (FEET_DISTANCE_RANGE, KNEE_DISTANCE_RANGE, LegsRewardsCfg, LocoManipMarlEnvCfg,
                                     MarlRewardsCfg)

_FEET = SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True)
STANDING_STAGGER_MAX = 0.05  # m


@configclass
class LegsSquareLegsRewardsCfg(LegsRewardsCfg):
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


@configclass
class LegsSquareRewardsCfg(MarlRewardsCfg):
    legs: LegsSquareLegsRewardsCfg = LegsSquareLegsRewardsCfg()


@configclass
class LegsSquareEnvCfg(LocoManipMarlEnvCfg):
    rewards: LegsSquareRewardsCfg = LegsSquareRewardsCfg()

    # configclass's post-init deep-copies every member that isn't a property of the instance's own class (its
    # __dict__), and would try to set the inherited one
    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        # LegsPass_s42's overrides: no arm goals, the arms at their rest pose by IK alone
        self.commands.arm_targets.arm_goal_prob = 0.0
        self.actions.arm_pos.residual_gain = 0.0


@configclass
class LegsSquareAmpLegsRewardsCfg(LegsSquareLegsRewardsCfg):
    amp_style = AMP_STYLE


@configclass
class LegsSquareAmpRewardsCfg(MarlRewardsCfg):
    legs: LegsSquareAmpLegsRewardsCfg = LegsSquareAmpLegsRewardsCfg()


@configclass
class LegsSquareAmpEnvCfg(LegsSquareEnvCfg):
    rewards: LegsSquareAmpRewardsCfg = LegsSquareAmpRewardsCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt


@configclass
class LegsBalanceAmpLegsRewardsCfg(LegsSquareAmpLegsRewardsCfg):
    # standing, the load split evenly: at most 1 per second, as feet_stagger_standing
    feet_load_balance_standing = RewTerm(func=mdp.feet_load_balance_band, weight=1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES, preserve_order=True),
        "max_imbalance": 0.1, "command_name": "base_velocity"})


@configclass
class LegsBalanceAmpRewardsCfg(MarlRewardsCfg):
    legs: LegsBalanceAmpLegsRewardsCfg = LegsBalanceAmpLegsRewardsCfg()


@configclass
class LegsBalanceAmpEnvCfg(LegsSquareEnvCfg):
    rewards: LegsBalanceAmpRewardsCfg = LegsBalanceAmpRewardsCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt
