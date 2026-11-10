"""The legs alone walking on sport mode's velocity commands: LocoManip-LegsOnly-v0.

The walking task (locomanip_marl_env_cfg.py) with no arm goals and the arms held at their rest pose by IK alone
(the arms' residual 0), and on top of its legs terms:
- the feet and knee bands on the lateral component of each pair's offset in the pelvis heading frame instead of its
  horizontal distance, and a standing band on the feet's fore-aft offset, so the stance's width and stagger are
  scored apart. Against sport mode's bags (lateral in the levelled pelvis frame): feet lateral (0.28, 0.45), stepping
  p1 0.267, p99 0.441, standing 0.297-0.338; knees lateral, KNEE_DISTANCE_RANGE (0.286, 0.376), stepping p1 0.284,
  p99 0.394; feet fore-aft while standing is commanded |dx| <= 0.05, the bags' standing p99 0.047;
- a style reward from an adversarial motion prior on LAFAN1's walking, running and sprinting retargeted to the H1-2
  (mdp.amp_style, a discriminator trained alongside the policy, ramped in over 20k-60k steps);
- a standing band on the feet's load split, 1 up to 60/40.

Why each is there: with the total feet distance banded, the standing stance turned from side by side to one foot
ahead at a constant 0.40 m (LegsPass_s42 and LegsPass with the style, ~120k steps) and walking lost its step-through
20-40k steps later; with the stance scored apart (LegsSquare_s42) it held to the end. With the style added, the
standing stance went lopsided by 160k (the pelvis rolled ~17 deg toward a bent left knee, the left foot carrying 64-66%
of the load), though every term but the style paid less for it. The load band fixed that: LegsBalanceAmp_s42 (200k
steps, this task under its earlier name LocoManip-LegsBalanceAmp-v0) stands at 48-53% and is GOLEM lmg_deploy's
policies/h12.

The style's reference is scripts/build_amp_reference.py on Unitree's LAFAN1 retargeting (CC BY-NC-ND 4.0, not
committed: build it into data/amp_lafan1_h1_2.npz). Training the gait by the style alone, without the gait clock
(earlier AmpWalk tasks), skated or shuffled in place; the style works here as a tie-breaker on the clock's gait.
"""

import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, KNEE_LINK_NAMES, LOWER_JOINT_NAMES

from . import mdp
from .locomanip_marl_env_cfg import (FEET_DISTANCE_RANGE, KNEE_DISTANCE_RANGE, LegsRewardsCfg, LocoManipMarlEnvCfg,
                                     MarlRewardsCfg)

_FEET = SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True)
STANDING_STAGGER_MAX = 0.05  # m

AMP_REFERENCE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                              "amp_lafan1_h1_2.npz"))
# the style reward at the tracking terms' combined weight (2.0 + 1.5 per second): 1 per step for a transition the
# discriminator takes for the reference's. The discriminator: lr 1e-5, 4 minibatch steps per rollout and a gradient
# penalty of 10 (5e-5 x 16 and 5 saturated it), judging transitions 3 steps (30 ms) apart. The style reward is off
# until 20k steps and at full weight from 60k: from the start, standing still out-earned walking and the policy never
# left it.
AMP_STYLE = RewTerm(func=mdp.amp_style, weight=3.5, params={
    "joint_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True),
    "feet_cfg": _FEET,
    "sensor_cfg": SceneEntityCfg("height_scanner"),
    "reference": AMP_REFERENCE,
    "hidden": (1024, 512), "lr": 1e-5, "weight_decay": 1e-4, "grad_penalty": 10.0, "logit_reg": 0.05,
    "update_every": 48, "num_updates": 4, "batch_size": 8192, "buffer_size": 400_000, "frame_stride": 3,
    "style_ramp": (20_000, 60_000)})


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
    amp_style = AMP_STYLE
    # standing, the load split evenly: at most 1 per second, as feet_stagger_standing
    feet_load_balance_standing = RewTerm(func=mdp.feet_load_balance_band, weight=1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES, preserve_order=True),
        "max_imbalance": 0.1, "command_name": "base_velocity"})


@configclass
class LegsOnlyRewardsCfg(MarlRewardsCfg):
    legs: LegsOnlyLegsRewardsCfg = LegsOnlyLegsRewardsCfg()


@configclass
class LegsOnlyEnvCfg(LocoManipMarlEnvCfg):
    rewards: LegsOnlyRewardsCfg = LegsOnlyRewardsCfg()

    # configclass's post-init deep-copies every member that isn't a property of the instance's own class (its
    # __dict__), and would try to set the inherited one
    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        # no arm goals; the arms at their rest pose by IK alone
        self.commands.arm_targets.arm_goal_prob = 0.0
        self.actions.arm_pos.residual_gain = 0.0
