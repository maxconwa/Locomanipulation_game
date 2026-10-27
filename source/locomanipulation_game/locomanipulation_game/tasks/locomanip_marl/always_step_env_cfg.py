"""Velocity following, stepping on any command: LocoManip-AlwaysStep-v0.

The walking task (locomanip_marl_env_cfg.py) with its full-stick velocity commands, a quarter of them zero
(rel_standing_envs 0.25; the walking task has 5%) and no small-command mode (the walking task's 20% of the moving
commands near the gate are drawn uniformly over the ranges instead), and its stand/walk gate: the clock runs, and the robot steps, on any
command whose norm reaches zero_cmd_eps (0.03), and stops with both legs in stance under it. The legs are paid only for
following the command (the walking task's tracking terms) and the clock: each swing foot lifted to sport mode's 6 cm
sine reference over its 0.70 s stride, the stance foot down and the leg joints' swing sport mode's (the walking task's
sport_gait; on foot height alone the policy reached 6 cm with the thigh vertical, the shin folded back and the toes
down), less the walking task's safety and smoothness costs (GOLEM's limits, effort, action rate, joint limits, contact
forces, self-contact, knee valgus, an upright torso), and a small reward for the feet and the knees inside their distance
bands (nothing outside). No alive, posture, hip or drift terms. The arms hold
their rest pose by IK alone (residual scale 0; their agent still trains, to no effect). Flat ground (row 0, no terrain
curriculum), no pushes; the motors', friction, mass and IMU randomization and the action delay stay.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES

from . import mdp
from .locomanip_marl_env_cfg import LegsRewardsCfg, LocoManipMarlEnvCfg, MarlRewardsCfg

# The walking task's legs terms this task drops: alive, the stance, the hips, the drift, foot slip and the yaw twist,
# the swing-height terms it replaces, and the arm-goal ones. The tracking, sport_gait and the feet and knee distance
# bands stay.
DROPPED_LEGS_TERMS = [
    "alive", "base_height", "contact_no_vel", "hip_yaw_mean", "hip_roll_pos", "stance_base_vel", "arm_goal_yaw_rate",
    "feet_swing_clearance", "feet_swing_height", "yaw_twist", "posture",
]


@configclass
class AlwaysStepLegsRewardsCfg(LegsRewardsCfg):
    """The gait terms; the tracking, sport_gait and the safety and smoothness costs are the walking task's
    (LegsRewardsCfg)."""

    # each swing foot at the 6 cm sine reference: at most 0.57 per second at weight 1 (two feet, 45% of the stride
    # each, the lift-share weighting's mean 0.64), 0.15 of that for dragged feet. At 4.0 (2.3/s) the tracking's 2.1/s
    # for shuffling won and the feet stayed down (StepWalk_s42); at 12.0 a full lift pays 6.9/s, twice the tracking's
    # maximum
    feet_swing_height_tracking = RewTerm(func=mdp.feet_swing_height_tracking_exp, weight=12.0, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True),
        "rest_height": 0.045,  # the ankle's standing height
        "lift_height": 0.06,
        "command_name": "base_velocity",
        "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
        "std": 0.03,
    })
    # each foot's contact at the clock's stance or swing: at most 2 per second at weight 1, 1.1 with both feet down
    contact = RewTerm(func=mdp.contact_matches_phase, weight=0.5,
                      params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES),
                              "command_name": "base_velocity"})

    def __post_init__(self):
        for name in DROPPED_LEGS_TERMS:
            setattr(self, name, None)
        # a small reward for the feet and the knees inside their bands, nothing outside (body_pair_distance_band: 1
        # inside, 0.37 at 1 cm out, 0.05 at 3 cm; the horizontal distance, stride included). Without them the stance
        # widened unchecked: at 140k the feet stood 0.98 m apart and the knees 0.83 m (AlwaysStep200k_s42). The feet's
        # upper bound is sport mode's p95 at speed (0.45 m) instead of the walking task's 0.6, which let 0.41-0.49 m
        # through at 60k; the knees keep the walking task's band, which sport mode's own steps leave 5% of the time
        for term in (self.feet_distance, self.knee_distance):
            term.func = mdp.body_pair_distance_band
            term.weight = 0.5
        self.feet_distance.params["max_dist"] = 0.45


@configclass
class AlwaysStepRewardsCfg(MarlRewardsCfg):
    legs: AlwaysStepLegsRewardsCfg = AlwaysStepLegsRewardsCfg()


@configclass
class AlwaysStepEnvCfg(LocoManipMarlEnvCfg):
    rewards: AlwaysStepRewardsCfg = AlwaysStepRewardsCfg()

    # configclass's post-init deep-copies every member that isn't a property of the instance's own class (its
    # __dict__), and would try to set the inherited one
    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        # a quarter of the commands zero, so standing still under the stand/walk gate is learned as well as stepping
        self.commands.base_velocity.rel_standing_envs = 0.25
        # no small-command mode: its share is drawn uniformly over the full ranges (of the moving commands, 20% turn
        # in place, 20% walk sideways, 60% uniform)
        self.commands.base_velocity.small_command_prob = 0.0
        # the arms at their rest pose by IK alone
        self.actions.arm_pos.scale = 0.0
        # flat ground, no pushes
        self.events.push_robot = None
        self.curriculum.terrain_levels = None
        self.curriculum.push_levels = None
