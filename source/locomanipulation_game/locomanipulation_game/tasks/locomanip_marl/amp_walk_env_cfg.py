"""Walking on a velocity command with an adversarial motion prior: LocoManip-AmpWalk-v0.

The walking task (locomanip_marl_env_cfg.py) with the gait made by imitation instead of a clock: the legs are paid
for following the command (the cycle-mean tracking terms, unchanged) and for moving like LAFAN1's walking, running
and sprinting retargeted to the H1-2 (mdp.amp_style, a discriminator trained alongside the policy; the reference
is in no observation), plus a light electrical-power term, smoothness (the action rates), a foot-slip cost
(feet_slide) and the safety terms (GOLEM's limits, joint limits, contact forces, self-contact, knee valgus, the
action clip). No gait clock, sport-mode
stance or swing, posture bands, steady-torso or effort terms. The legs actor observes only the IMU, the joints, its
last action and the (vx, vy, wz) command: no gait phase and no arm-goal inputs; the clock is out of every group.
The commands are sport mode's stick (unchanged). The arms hold their rest pose by IK alone (no goals, residual 0).

The reference: scripts/build_amp_reference.py on Unitree's LAFAN1 retargeting (h1_2/walk*, run*, sprint*, each
also mirrored; upright frames turning no faster than the command's 0.97 rad/s; 75 min at 100 Hz).

AmpWalk_s42 (the first version: 39-dim state without the feet's ground height and velocity, single 10 ms
transitions, discriminator lr 5e-5 x 16 updates per rollout, gradient penalty 5, no feet_slide, unmirrored reference)
saturated its discriminator by 15k steps (policy scored -0.85, style reward 0.13 per step at 100k) and skated: both
feet loaded 80% of the time, the loaded feet sliding at 16 cm/s, 13 touchdowns per second, a constant left turn.
AmpWalk2_s42 (all that fixed, the style at full weight from the start) stood in place: speed ratio 0.21, cos 0.28 at
60k, both feet loaded over 90% of the time. Hence the style's ramp. AmpWalk3_s42 (the ramp) shuffled in place to the
end: 0.26 m/s on a 0.5 m/s command at its best (140k), 0.02 m/s at 200k, both feet loaded 87% of the time, no turning.

LocoManip-AmpStyleOnly-v0 (AmpStyleOnlyEnvCfg): the same with the legs paid for the command and the style alone, AMP's
own task + style split: no energy, safety, smoothness or foot-slip terms. AmpStyleOnly_s42 tracked (cos 0.98, speed
ratio 0.97 at 140k) by skating: 0.53 m/s on a 0.5 m/s command, the loaded feet sliding 25 cm/s, 17.5 touchdowns a
second. Its discriminator gave a stepping gait (LegsPass_s42's) 0.57 per step and its own 0.14: the style favoured
stepping, but from a standing start the policy never found it.

LocoManip-AmpRsi-v0 (AmpRsiEnvCfg): the command and the style with the smoothness and safety terms back (no energy or
foot-slip terms), and AMP's reference state initialisation (mdp.reset_from_reference): half the resets start from a
LAFAN1 frame no faster than 0.8 m/s, mid-stride. AmpRsi_s42 skated all the same: 0.49 m/s on a 0.5 m/s command
at 200k, the loaded feet sliding 22 cm/s, 10.7 touchdowns a second; style reward 0.20.

LocoManip-AmpAirTime-v0 (AmpAirTimeEnvCfg): AmpRsi plus Isaac Lab's feet_air_time, paid at each touchdown: the swing's
air time less 0.3 s, while the command walks (over 0.1 m/s). A shuffle's 0.05 s swings at 11 touchdowns a second
cost about 2.8 per second; a 0.7 s gait's 0.3 s swings about nothing. That forgot the manager's dt: an event term
is scaled by the 10 ms step as a rate is, so each touchdown paid (air - 0.3) x 0.01 and the term came to -0.008 per
second. AmpAirTime_s42 skated as AmpRsi_s42 had (0.50 m/s on a 0.5 m/s command at 120k, 13.5 touchdowns a second); in
Isaac its feet were off the ground (under 1 N) 4% of the time.

LocoManip-LegsPassAmp-v0 (LegsPassAmpEnvCfg): the walking task unchanged, gait clock and every legs term (LegsPass_s42's
recipe: legs only, the arms at rest by IK), plus the same style reward and discriminator.
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, LOWER_JOINT_NAMES

_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAMES, preserve_order=True)
_FEET = SceneEntityCfg("robot", body_names=FOOT_LINK_NAMES, preserve_order=True)

from . import mdp
from .locomanip_marl_env_cfg import (LegsRewardsCfg, LocoManipMarlEnvCfg, MarlEventCfg, MarlObservationsCfg,
                                     MarlRewardsCfg)

AMP_REFERENCE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                                              "amp_lafan1_h1_2.npz"))
# The walking task's legs terms this task keeps: the command, smoothness and safety
KEPT_LEGS_TERMS = ["track_lin_vel_xy", "track_ang_vel_z", "action_rate", "ankle_action_rate", "dof_pos_limits",
                   "feet_contact_forces", "self_collision", "action_beyond_clip", "knee_valgus", "golem_position",
                   "golem_velocity", "golem_torque"]


# the style reward at the tracking terms' combined weight (2.0 + 1.5 per second): 1 per step for a transition the
# discriminator takes for the reference's
# discriminator: lr 1e-5, 4 minibatch steps per rollout and a gradient penalty of 10 (AmpWalk_s42's 5e-5 x 16 and
# 5 saturated it), judging transitions 3 steps (30 ms) apart. The style reward is off until 20k steps and at full
# weight from 60k: from the start (AmpWalk2_s42) standing still out-earned walking badly and the policy never left
# it (speed ratio 0.21 at 60k); on the task reward alone AmpWalk_s42 walked by 40k (cos 0.92)
AMP_STYLE = RewTerm(func=mdp.amp_style, weight=3.5, params={
    "joint_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True),
    "feet_cfg": _FEET,
    "sensor_cfg": SceneEntityCfg("height_scanner"),
    "reference": AMP_REFERENCE,
    "hidden": (1024, 512), "lr": 1e-5, "weight_decay": 1e-4, "grad_penalty": 10.0, "logit_reg": 0.05,
    "update_every": 48, "num_updates": 4, "batch_size": 8192, "buffer_size": 400_000, "frame_stride": 3,
    "style_ramp": (20_000, 60_000)})


@configclass
class AmpWalkLegsRewardsCfg(LegsRewardsCfg):
    amp_style = AMP_STYLE
    # each loaded foot's horizontal speed (m/s), summed: AmpWalk_s42 skated at about 0.3 m/s summed (-0.3 per second),
    # LegsPass_s42 stepped at about 0.12
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-1.0, params={"sensor_cfg": _FEET_CONTACT, "asset_cfg": _FEET})
    # light: a tie-breaker between reference-like gaits. StandEnergy's power model (mdp.electrical_power_exp) on a
    # walking scale, 300 W (100 W suits standing: walking draws several hundred and would sit on exp's flat tail)
    energy = RewTerm(func=mdp.electrical_power_exp, weight=0.5, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True),
        "heat_at_limit": 200.0, "power_scale": 300.0})

    def __post_init__(self):
        for name, term in list(vars(self).items()):
            if isinstance(term, RewTerm) and name not in ("amp_style", "energy", "feet_slide") \
                    and name not in KEPT_LEGS_TERMS:
                setattr(self, name, None)


@configclass
class AmpWalkRewardsCfg(MarlRewardsCfg):
    legs: AmpWalkLegsRewardsCfg = AmpWalkLegsRewardsCfg()


@configclass
class AmpStyleOnlyLegsRewardsCfg(AmpWalkLegsRewardsCfg):
    def __post_init__(self):
        for name, term in list(vars(self).items()):
            if isinstance(term, RewTerm) and name not in ("track_lin_vel_xy", "track_ang_vel_z", "amp_style"):
                setattr(self, name, None)


@configclass
class AmpStyleOnlyRewardsCfg(MarlRewardsCfg):
    legs: AmpStyleOnlyLegsRewardsCfg = AmpStyleOnlyLegsRewardsCfg()


@configclass
class AmpRsiLegsRewardsCfg(AmpWalkLegsRewardsCfg):
    def __post_init__(self):
        for name, term in list(vars(self).items()):
            if isinstance(term, RewTerm) and name != "amp_style" and name not in KEPT_LEGS_TERMS:
                setattr(self, name, None)


@configclass
class AmpRsiRewardsCfg(MarlRewardsCfg):
    legs: AmpRsiLegsRewardsCfg = AmpRsiLegsRewardsCfg()


@configclass
class AmpAirTimeLegsRewardsCfg(AmpRsiLegsRewardsCfg):
    # each touchdown pays its swing's air time less the threshold (negative for a shorter swing), while the command
    # walks: the per-swing signal the discriminator doesn't give (AmpRsi_s42 and the AMP runs before it shuffled)
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=1.0, params={
        "command_name": "base_velocity", "sensor_cfg": _FEET_CONTACT, "threshold": 0.3})

    def __post_init__(self):
        for name, term in list(vars(self).items()):
            if isinstance(term, RewTerm) and name not in ("amp_style", "feet_air_time") and name not in KEPT_LEGS_TERMS:
                setattr(self, name, None)


@configclass
class AmpAirTimeRewardsCfg(MarlRewardsCfg):
    legs: AmpAirTimeLegsRewardsCfg = AmpAirTimeLegsRewardsCfg()


@configclass
class AmpRsiEventCfg(MarlEventCfg):
    # after reset_base and reset_joints: half the resets mid-stride from LAFAN1 (frames at the command box's speeds and
    # a little past: 54% of the reference), the lower foot 1 cm above its standing height at worst
    reset_from_reference = EventTerm(func=mdp.reset_from_reference, mode="reset", params={
        "reference": AMP_REFERENCE, "probability": 0.5, "max_speed": 0.8, "clearance": 0.01,
        "joint_cfg": SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES, preserve_order=True)})


@configclass
class AmpWalkObservationsCfg(MarlObservationsCfg):
    def __post_init__(self):
        # no gait clock anywhere; the legs' only command is the velocity
        for group in (self.legs, self.arms, self.critic):
            group.gait_phase = None
            group.gait_phase_cos = None
        self.legs.arm_goal = None
        self.legs.ee_targets = None


@configclass
class AmpWalkEnvCfg(LocoManipMarlEnvCfg):
    rewards: AmpWalkRewardsCfg = AmpWalkRewardsCfg()
    observations: AmpWalkObservationsCfg = AmpWalkObservationsCfg()

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


@configclass
class AmpStyleOnlyEnvCfg(AmpWalkEnvCfg):
    rewards: AmpStyleOnlyRewardsCfg = AmpStyleOnlyRewardsCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt


@configclass
class AmpRsiEnvCfg(AmpWalkEnvCfg):
    rewards: AmpRsiRewardsCfg = AmpRsiRewardsCfg()
    events: AmpRsiEventCfg = AmpRsiEventCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt


@configclass
class AmpAirTimeEnvCfg(AmpRsiEnvCfg):
    rewards: AmpAirTimeRewardsCfg = AmpAirTimeRewardsCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt


@configclass
class LegsPassAmpLegsRewardsCfg(LegsRewardsCfg):
    amp_style = AMP_STYLE


@configclass
class LegsPassAmpRewardsCfg(MarlRewardsCfg):
    legs: LegsPassAmpLegsRewardsCfg = LegsPassAmpLegsRewardsCfg()


@configclass
class LegsPassAmpEnvCfg(LocoManipMarlEnvCfg):
    rewards: LegsPassAmpRewardsCfg = LegsPassAmpRewardsCfg()

    @property
    def policy_dt(self) -> float:
        return self.decimation * self.sim.dt

    def __post_init__(self):
        super().__post_init__()
        # LegsPass_s42's overrides: no arm goals, the arms at their rest pose by IK alone
        self.commands.arm_targets.arm_goal_prob = 0.0
        self.actions.arm_pos.residual_gain = 0.0
