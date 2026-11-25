"""The cooperative task on the stage-1 motion prior (PULSE): LocoManip-Marl-Pulse-v0.

LocoManip-Marl-Torso-v0 (marl_torso_env_cfg.py: Marl-Direct, the legs on a velocity command, the arms on wrist poses
and the torso joint, each with a share of the other's tracking, MAPPO via skrl) where each agent acts in the 32-wide
latent of its frozen stage-1 actor from pretraining/ (pulse_stage1_run, default main_s1_finetune: a stage-1 run on
main's plant) and the env decodes it into joint targets (mdp/pulse_actions.py), as h12_train's stage 2 did
(amass_s2_pulse). The design: docs/superpowers/specs/2026-11-24-fair-comparison-design.md (piece 3; the first
version, on pretraining's plant, docs/superpowers/specs/2026-11-24-pulse-cooperative-design.md). Against MarlTorso:

- main's plant throughout: its robot (explicit PD at sport mode's gains, armature, default pose and standing height),
  100 Hz, its joint ranges; the stage-1 run must have been trained on it (pretraining.plants.MAIN: run.json's "plant";
  _PulseShared refuses another), so the decoders' contract is the env's: target = main's default pose + the plant's
  action scale x the decoder's output, clipped to its target limits within GOLEM's position envelope; the legs'
  decoder sets the 12 leg joints, the upper body's the torso and the 14 arm joints (the arms agent owns the torso, as
  in MarlTorso, through the latent); no IK, no residual; the grippers as main holds them;
- MarlTorso's joint-space torso term (torso_pos), its observation and its rate reward absent (arms_drive_torso off:
  the upper-body decoder drives the torso), as the other joint-space terms;
- main's randomisation and sensing unchanged: friction, torso mass, base and joint resets, the pre-pose resets,
  actuator scales, action delay, encoder offsets, joint noise, the lowstate IMU, the gyro low-pass (the actors'; the
  decoders read the gyro before it, as h12_train's deploy does);
- PULSE's latent prior (pulse_prior, fitted by pretraining/scripts/fit_prior.py): z = tanh(u + prior(proprio)), u the
  skrl actor's output, which starts near 0 (last layer x0.01, std 0.1, log std in [log 0.03, log 0.5]); clip_actions 5;
- the gait and stance shaping and the torso posture terms at weight 0 (kept, so the logs list them; PULSE_ZEROED_TERMS),
  so they do not compete with the motion prior; the terms that read a joint-space action term removed
  (PULSE_REMOVED_TERMS); action_rate on the latent; the rest as MarlDirect's;
- the observations as MarlDirect's, the agents' applied action the latent (the gait clock stays: harmless);
- skrl: main's agent yaml as it is (agents.skrl_mappo_pulse_cfg; 100 Hz, so main's discount and lambda).
"""

import math

from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import REPO_ROOT

from .marl_torso_env_cfg import MarlTorsoEnvCfg
from .mdp.pulse_actions import PulseLatentActionCfg, resolve

PULSE_STAGE1_RUN = "data/pretraining/runs/main_s1_finetune"
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


@configclass
class MarlPulseActionsCfg:
    # legs first: the agents' actions are concatenated in possible_agents order
    legs_latent = PulseLatentActionCfg(asset_name="robot", agent="legs", stage1_run="")
    arms_latent = PulseLatentActionCfg(asset_name="robot", agent="arms", stage1_run="")


@configclass
class MarlPulseEnvCfg(MarlTorsoEnvCfg):
    arms_drive_torso: bool = False
    """Off: the arms' latent drives the torso through the upper-body decoder, so MarlTorso's joint-space torso term,
    its observation and its rate reward never exist here (sync_arms_drive_torso would add them on each sync_derived,
    and refuses: there is no arm_pos term)."""
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
        # after ArmsOnly's, which sets the arms' alone
        self.policy_initial_std = {agent: PULSE_INITIAL_STD for agent in self.possible_agents}
        self.policy_log_std_bounds = {agent: PULSE_LOG_STD_BOUNDS for agent in self.possible_agents}

    def __post_init__(self):
        super().__post_init__()
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

