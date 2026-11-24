"""The cooperative task on the stage-1 motion prior (PULSE): LocoManip-Marl-Pulse-v0.

LocoManip-Marl-Direct-v0 (marl_direct_env_cfg.py: the legs on a velocity command, the arms on wrist poses, each with
a share of the other's tracking, MAPPO via skrl) where each agent acts in the 32-wide latent of its frozen stage-1
actor from pretraining/ (pulse_stage1_run, default the GOLEM-aware fine-tune isaac_s1_golem_aware) and the env
decodes it into joint targets (mdp/pulse_actions.py), as h12_train's stage 2 did (amass_s2_pulse). The design and
its reasons: docs/superpowers/specs/2026-11-24-pulse-cooperative-design.md. Against MarlDirect:

- the decoders' contract, not main's: 50 Hz (decimation 10 over main's 500 Hz physics, so the 0-10 substep action
  delay stays 0-20 ms); target = pretraining's default pose + its ACTION_SCALE x the decoder's output, clipped to its
  TARGET_LIMITS within GOLEM's position envelope; the legs' decoder sets the 12 leg joints, the upper body's the torso
  and the 14 arm joints (the arms agent owns the torso); no IK, no residual; the grippers as main holds them;
- the robot: main's explicit PD (IdealPDActuatorCfg, so the gain and strength draws act on it) on 17 joints, implicit
  PD on the shoulder yaws, elbows and wrists (no gain or strength draws there; the action delay, the encoder offsets
  and GOLEM's torque check act on all 27), with pretraining's gains, effort limits and armature (pretraining.constants), its default
  pose and pelvis height as the articulation's (init_state), its joint ranges (h12_train's MJCF's, a start-up event
  as stage 1's; main's USD has the shoulder rolls at +-0.19). Stage 1 ran implicit PD at 5 ms: the plant differs a
  little, inside what the gain draws cover;
- main's randomisation and sensing unchanged: friction, torso mass, base and joint resets, the pre-pose resets (which
  name the articulation's default pose: pretraining's here, the hold the deploy ramps to), actuator scales, action
  delay, encoder offsets, joint noise, the lowstate IMU, the gyro low-pass (the actors'; the decoders read the gyro
  before it, as h12_train's deploy does);
- PULSE's latent prior (pulse_prior, fitted by pretraining/scripts/fit_prior.py): z = tanh(u + prior(proprio)), u the
  skrl actor's output, which starts near 0 (last layer x0.01, std 0.1, log std in [log 0.03, log 0.5]); clip_actions 5;
- the gait and stance shaping and the torso posture terms at weight 0 (kept, so the logs list them; PULSE_ZEROED_TERMS),
  so they do not compete with the motion prior; the terms that read a joint-space action term removed
  (PULSE_REMOVED_TERMS); action_rate on the latent; the rest as MarlDirect's;
- the observations as MarlDirect's, the agents' applied action the latent (the gait clock stays: harmless);
- skrl: main's agent yaml at discount 0.99 (its own comment's value at 50 Hz; agents.skrl_mappo_pulse_cfg).
"""

import copy
import math
import re

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from pretraining import constants as pretrain
from pretraining.tasks.events import joint_ranges

from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG, REPO_ROOT

from .golem_safety import GOLEM_LIMITS
from .locomanip_marl_env_cfg import DECIMATION as MAIN_DECIMATION
from .lowstate_imu import LowstateImuCfg
from .marl_direct_env_cfg import MarlDirectEnvCfg
from .mdp.pulse_actions import PulseLatentActionCfg, resolve

PULSE_DECIMATION = 10  # 50 Hz at main's 2 ms physics, the decoders' rate
PULSE_STAGE1_RUN = "data/pretraining/runs/isaac_s1_golem_aware"
PULSE_PRIOR = "prior.pt"  # relative: in the stage-1 run's directory (fit_prior.py's default --out)
PULSE_CLIP_ACTIONS = 5.0  # on the pre-tanh latent residual
PULSE_OUTPUT_INIT_SCALE = 0.01
PULSE_INITIAL_STD = 0.1
PULSE_LOG_STD_BOUNDS = (math.log(0.03), math.log(0.5))
LEGS_LATENT = "legs_latent"
ARMS_LATENT = "arms_latent"
PULSE_ZEROED_TERMS = (
    "feet_swing_height_tracking", "contact", "sport_gait", "feet_pass", "feet_distance", "knee_distance", "base_height",
    "knee_standing", "feet_stagger_standing", "feet_load_balance_standing", "orientation", "lin_vel_z", "ang_vel_xy")
"""The legs' terms at weight 0: the gait and stance shaping and the torso posture."""
PULSE_REMOVED_TERMS = ("ankle_action_rate", "engage_smooth", "target_near_bounds", "action_beyond_clip")
"""The legs' terms that read a joint-space action term (its joint columns of the action, its policy targets, its
bounds, its overshoot), which a latent does not have."""

_VELOCITY_LIMITS = {row[0]: row[3] for row in GOLEM_LIMITS}  # rad/s, the URDF's (main's actuators carry the same)


def _pd(pattern: str) -> IdealPDActuatorCfg:
    """One of pretraining's joint groups (constants.ACTUATOR_GROUPS) as main's explicit PD, its values per joint."""
    joints = [j for j in pretrain.JOINT_NAMES if re.fullmatch(pattern, j)]
    effort = {j: pretrain.EFFORT[j] for j in joints}
    velocity = {j: _VELOCITY_LIMITS[j] for j in joints}
    return IdealPDActuatorCfg(
        joint_names_expr=joints, effort_limit=effort, effort_limit_sim=effort, velocity_limit=velocity,
        velocity_limit_sim=velocity, stiffness={j: pretrain.KP[j] for j in joints},
        damping={j: pretrain.KD[j] for j in joints}, armature={j: pretrain.ARMATURE[j] for j in joints}, friction=0.0)


# The forearm joints (shoulder yaws, elbows, wrists): implicit PD, as stage 1 ran them. Explicit damping at the physics
# step is stable only while every eigenvalue of dt M^-1 Kd stays under 2, and rings above 1 (M the joint-space mass
# matrix, Kd the explicit joints' kd). With pretraining's kd (12 on the shoulder yaws, elbows and wrists) and armature,
# at dt 0.002 and pretraining's default pose (M: CL_Assets' h1_2_magpie.xml in MuJoCo, armature replaced), the
# diagonal kd dt / I alone is 4.39 on the wrist rolls, 1.42 / 1.24 on the wrist yaws / pitches; with the wrists
# implicit, the coupled shoulder-yaw / elbow mode is still at 1.28 (1.76 under actuator_scales' gain x1.2 x strength
# x1.15), and in the sim (zero latent, prior) the shoulder yaws rang up to ~1.5-2 rad/s within 2 s, with or without the
# randomisation, and tripped GOLEM's torque check on ~half the steps (all implicit: 0.6 rad/s, no trips). With the
# shoulder yaws and elbows implicit too the largest explicit eigenvalue is 0.50 (the ankle rolls; 0.69 under the
# draws); the elbows alone left explicit would be 0.69 (0.96).
# What this costs on these joints: actuator_scales (mdp.randomize_actuator_scales, IdealPDActuator only) draws no gains
# or strength for them, and GOLEM's torque check (golem_safety.GolemEstopMonitor._motor_torque) estimates their torque
# from the nominal gains, as it does for every joint (the action delay and encoder offsets still act: they go through
# the joint targets).
PULSE_IMPLICIT_GROUPS = ("shoulder_yaw", "elbows", "wrists")


def _implicit(pattern: str) -> ImplicitActuatorCfg:
    """One of pretraining's joint groups as stage 1's implicit PD (pretraining.robot), its values per joint."""
    joints = [j for j in pretrain.JOINT_NAMES if re.fullmatch(pattern, j)]
    return ImplicitActuatorCfg(
        joint_names_expr=joints, stiffness={j: pretrain.KP[j] for j in joints},
        damping={j: pretrain.KD[j] for j in joints}, effort_limit_sim={j: pretrain.EFFORT[j] for j in joints},
        velocity_limit_sim={j: _VELOCITY_LIMITS[j] for j in joints},
        armature={j: pretrain.ARMATURE[j] for j in joints}, friction=0.0)


H1_2_PULSE_CFG = ArticulationCfg(
    spawn=copy.deepcopy(H1_2_MAGPIE_CFG.spawn),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, pretrain.DEFAULT_ROOT_HEIGHT),
        joint_pos={**pretrain.DEFAULT_JOINT_POS, "[lr]g_.*_hinge_.*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=H1_2_MAGPIE_CFG.soft_joint_pos_limit_factor,
    actuators={**{name: (_implicit(p) if name in PULSE_IMPLICIT_GROUPS else _pd(p))
                  for name, p in pretrain.ACTUATOR_GROUPS.items()},
               "grippers": copy.deepcopy(H1_2_MAGPIE_CFG.actuators["grippers"])},
)
"""Main's H1-2 (USD, self-collision, grippers) with pretraining's actuators, default pose and pelvis height."""


@configclass
class MarlPulseActionsCfg:
    # legs first: the agents' actions are concatenated in possible_agents order
    legs_latent = PulseLatentActionCfg(asset_name="robot", agent="legs", stage1_run="")
    arms_latent = PulseLatentActionCfg(asset_name="robot", agent="arms", stage1_run="")


@configclass
class MarlPulseEnvCfg(MarlDirectEnvCfg):
    pulse_stage1_run: str = PULSE_STAGE1_RUN
    """The stage-1 run whose actors decode the latents (actors.pt, run.json); relative: to the repository root."""
    pulse_prior: str | None = PULSE_PRIOR
    """The latent prior (fit_prior.py's file); relative: to the stage-1 run's directory (the default is
    <pulse_stage1_run>/prior.pt). None or "": no prior, z = tanh(u)."""

    def sync_derived(self):
        super().sync_derived()
        # the parents' __post_init__ call this before this class's has replaced the actions
        if not isinstance(getattr(self.actions, LEGS_LATENT, None), PulseLatentActionCfg):
            return
        stage1 = resolve(self.pulse_stage1_run, str(REPO_ROOT))
        prior = resolve(self.pulse_prior, stage1)
        for term in (self.actions.legs_latent, self.actions.arms_latent):
            term.stage1_run, term.prior = stage1, prior
        # the IMU's vibration is per rad of leg target change per policy step, fitted at 100 Hz (decimation 5): the
        # same leg motion changes the targets decimation / 5 times as much per step here (from the field's default,
        # so a decimation override follows)
        self.lowstate_imu.vibration_per_target_rate = (
            LowstateImuCfg().vibration_per_target_rate * MAIN_DECIMATION / self.decimation)
        # after ArmsOnly's, which sets the arms' alone
        self.policy_initial_std = {agent: PULSE_INITIAL_STD for agent in self.possible_agents}
        self.policy_log_std_bounds = {agent: PULSE_LOG_STD_BOUNDS for agent in self.possible_agents}

    def __post_init__(self):
        super().__post_init__()
        # -- the decoders' rate and robot
        self.decimation = PULSE_DECIMATION
        robot = H1_2_PULSE_CFG.replace(prim_path=self.scene.robot.prim_path)
        robot.spawn = copy.deepcopy(robot.spawn)
        robot.spawn.articulation_props.fix_root_link = False  # MarlDirect's free pelvis
        self.scene.robot = robot
        # the arm goals' standing frame: the pelvis height of the stance the target tables are built at, which
        # ArmTargetsCommand builds them from (the default root state: pretraining's 0.993 m, not main's 0.9798 m
        # ArmsOnly set it to; with that, every goal sat 1.3 cm off its table pose in the pelvis frame)
        self.commands.arm_targets.standing_height = robot.init_state.pos[2]
        self.events.joint_ranges = EventTerm(func=joint_ranges, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot"), "ranges": dict(pretrain.JOINT_RANGES)})
        # -- the latent actions
        self.actions = MarlPulseActionsCfg()
        self.agent_action_terms = {"legs": [LEGS_LATENT], "arms": [ARMS_LATENT]}
        self.clip_actions = PULSE_CLIP_ACTIONS
        self.policy_output_init_scale = {agent: PULSE_OUTPUT_INIT_SCALE for agent in self.possible_agents}
        # -- the observations: the applied action is the latent
        obs = self.observations
        obs.legs.actions.params["action_name"] = LEGS_LATENT
        obs.arms.actions.params["action_name"] = ARMS_LATENT
        obs.critic.legs_actions.params["action_name"] = LEGS_LATENT
        obs.critic.arms_actions.params["action_name"] = ARMS_LATENT
        # -- the rewards
        legs, arms = self.rewards.legs, self.rewards.arms
        for name in PULSE_ZEROED_TERMS:
            getattr(legs, name).weight = 0.0
        for name in PULSE_REMOVED_TERMS:
            setattr(legs, name, None)
        legs.action_rate.params["action_name"] = LEGS_LATENT
        arms.action_rate.params["action_name"] = ARMS_LATENT
        self.sync_derived()

