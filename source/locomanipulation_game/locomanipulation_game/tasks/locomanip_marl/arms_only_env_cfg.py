"""The arms alone reaching standing arm goals, the pelvis pinned: LocoManip-ArmsOnly-v0.

The two-agent task (locomanip_marl_env_cfg.py) with only the arms acting and learning anything:
- the pelvis is pinned in the world at the stance height (fix_root_link), so nothing has to balance; the reset still
  places it (x, y, yaw drawn as the walking task's) but upright and at the stance height;
- the legs hold their default pose: their action scale is 0 and their reward terms are off, so the legs agent's
  actions do nothing (it is still trained, on the fall and GOLEM-trip penalties alone);
- every task is a standing arm goal (arm_goal_prob 1, the velocity command zero), each lasting a drawn 4-8 s
  (ARMS_GOAL_TIME_RANGE; 4 s up to ArmsOnly200k_s42), from the first step: the walking
  task's warm-start gate needs walking, which a pinned robot never does, so its deadline opens it at step 1 with no
  ramp, and the arms' residual is at full gain from then on;
- the arms' fresh actor starts with its last layer scaled by 0.01 and its bias zeroed (policy_output_init_scale), so
  the residual starts near 0 and the arms start as the IK alone. MainStep_s42 (10-27) ramped in an arms actor that had
  learned arbitrary offsets while its output had no effect, which threw the arms 0.4-0.48 m off their rest pose;
- the arms' exploration starts at std 0.2 (policy_initial_std), not the yaml's 1.0, so the noise doesn't swamp the IK;
- the arm joints' targets are clipped 0.1 rad inside GOLEM's edges, not 0.02: the goals are wrist poses at uniformly
  random joint angles, so the IK often drives a joint onto its clip, and at 0.02 rad gravity sagged it past GOLEM's
  position e-stop (the IK alone tripped in 94% of envs over 6 goals; at 0.1 rad, 12.5%, at the same reach: 119 vs
  124 mm mean end error, 45 vs 46% within 5 cm);
- the goals' standing frame is at the pinned pelvis's own height (0.9798 m), not the tables' stance (1.0024 m): every
  goal had sat 2.3 cm higher in the pelvis frame than the pose its table angles reach;
- no IK step: the goal tables keep each row's joint angles (built from angles ARMS_JOINT_RANGE_MARGIN inside the soft
  limits, into the run's own folder, ARMS_TABLE_FILE), each goal's seed is the nearest row's (the same lookup works
  on the robot for any commanded pose), and the arm targets move from their last value to the seed
  (mdp.IKPostureArmAction's trajectory, ARMS_IK). Alone (zero actions) on full table goals, each joint capped at
  0.05 rad per step: 92.8% reached, 29 mm, 0.12 rad; the old IK step 10.8%, 122 mm, 0.54 rad (an IK step from the
  measured angles, capped at 0.05 rad, can't hold a joint whose gravity sag needs more PD error than that: the
  shoulders stalled 0.3 rad short);
- the seed is solved for the goal itself from the nearest row's angles (ArmTargetsCommandCfg.seed_solve, the URDF's
  kinematics): ArmsOnlyV2_s42 (11-11) reached 95% of table goals but ~20% of off-table ones all run, the residual
  never closing the gap from the nearest row; solved, 94% of off-table goals are reachable and the IK alone put 96%
  of wrists within 5 cm in MuJoCo;
- each arm's joints move together, at most 2 rad/s and 10 rad/s^2 (trajectory_max_vel / _acc): per-joint caps of
  0.05 rad per step tripped GOLEM's shoulder torque e-stop in 46% of table goals in MuJoCo (the IK alone, the same
  with V2's policy); synchronized, none;
- a goal whose solved seed, or the straight joint-space path to it, overlaps cuRobo's self-collision spheres is drawn
  again (collision_check): V2's links overlapped in ~30% of goals in MuJoCo, mostly in transit; screened and
  synchronized, the IK alone overlapped in 16% of table and 9% of off-table goals;
- the action is the residual alone, 14 wide (posture_actions False): V2's posture half moved the seed, which the
  solve now sets and the collision check has screened;
- a tenth of the goals are the rest pose (rest_goal_prob), and the observations have no arm-goal flag: every task is
  a goal. V2's arms shook in rest (0.58 rad/s) when no goal was set; rest as a goal, 0.20;
- half the goals are moved off the table rows, a random 25-100% of the way from the rest pose (ARMS_OFFTABLE_*), so
  the seed is only the nearest row's and the residual has to close the gap, as for a commanded pose on the robot:
  the trajectory alone reaches 7.7% of goals a quarter of the way out (63 mm). A curriculum from near the rest pose
  outward (ArmTargetsCommandCfg.goal_scales) is left off: near goals are off the rows, the hardest here;
- the arms' std stays at or above ARMS_MIN_STD (policy_log_std_bounds): ArmsOnlyIK_s42 (11-10) went from 0.2 to the
  yaml's floor, 0.05, by 70k steps, and its reach stopped rising at 40k (33.5% reached at 40k, 33.3% at 60k).

Before the trajectory, the IK alone (residual 0, scripts/skrl/eval_arms.py) on these goals: 130 mm at the goal's
end, 49% of wrists within 5 cm, 12% of goals reached; ~9% of goals end with a hand on the pelvis (a "fell"); with the
trajectory, 2%. ArmsOnly_s42 (11-10, stopped at 109k) trained on a pinned pelvis before the IK's Jacobian indexing knew a fixed base: the IK didn't track
(632 mm), and the arms learned to park out of the penalties' way (764 mm at 100k).

The arms' actor sees the wrist targets, the joints, its last action and the wrists' poses and errors: not the IMU
(acceleration, gyro, gravity), the velocity command or the gait clock (ARMS_DROPPED_OBS), which a pinned, standing
robot holds constant or as noise (the critic's state keeps them). The rewards are the walking task's, and a reward for holding still on
the goal (ARMS_HOLD_STILL): ArmsOnlyV3_s42 (11-11) reached 98% / 88% of table / off-table goals at 20k but swayed 3-4 mm
RMS in the hold, its joints at 0.26 rad/s, where the trajectory alone holds without jitter; nothing paid stillness.
The reward didn't change it (ArmsOnlyV4_s42 / V5_s42, 11-11: at 60k the sway 4.7 mm and the joints at 0.34 rad/s, the IK
alone's 2.6 mm and 0.04 rad/s), so the policy's residual is low-passed at 1 Hz here (ARMS_IK residual_cutoff_hz; the
walking task's 3 Hz). That halved the joints' speed but not the sway: 4-8 s into a goal, with the arm settled,
ArmsOnlyV6_s42's residual still moved (0-10 Hz) and the wrist swayed 2.8-3.9 mm where the IK alone held 0.07 mm, so
the residual is latched once the wrists are on the goal (ARMS_IK freeze_residual_on_goal; mdp.IKPostureArmAction):
within 1 mm and 5 deg of the command, as a goal now counts as reached (ARMS_REACH_*).

The target also carries the arms' gravity torque over kp (ARMS_IK gravity_offset; mdp.ArmKinematics.gravity_torques,
the URDF's masses): the IK alone (11-12, pinned) held the wrists 26 mm off, 23 mm low, all of it the arm joints' sag
under gravity (shoulder pitch 2.1 deg), which the residual was left to cancel; with the offset, 2.8 mm in Isaac (the
actuator gain DR's spread) and 1.1 mm in MuJoCo. ArmsOnlyV7_s42 (no offset, 11-11/12) sat at 10-12 mm from 60k on, 0%
within 1 mm.

The residual is at half the walking task's scale (ARMS_RESIDUAL_SCALE_FACTOR) and the fine-position and rotation
kernels narrow on past the walking task's last stage, to 3 mm and 0.035 rad (ARMS_FINE_POS_STDS, ARMS_ROT_STDS):
ArmsOnlyV8_s42 (with the offset, 11-12) was on the last stage (12 mm, 0.08 rad) by 20k and held 7.1 / 10.6 mm on
table / off-table goals through 80k, worse than the IK alone with the offset (3.5 / 8.2 mm), its joints at 0.21 rad/s
in the hold (the IK alone's 0.04).

V10 (11-13): the target carries an integral of the measured wrists' pose error once the trajectory has stopped
(ARMS_IK integral_*; mdp.IKPostureArmAction), the residual is integrated rather than low-passed (arms_integrate), the
fixed 20 / 5 / 2 mm and 0.2 / 0.05 rad kernels and a capped linear term replace the annealed ones
(arms_precision_rewards), and a goal counts as reached within 2 mm / 5 deg (user 11-13; the residual's freeze stays
at 1 mm). ArmsOnlyV9_s42 sat at 5.6-6.3 / 9.5-9.8 mm from 60k to 240k. The 50k runs of 11-12/13 (Isaac, table /
off-table goals; V9 6.1 / 6.0 mm on table goals at 40k / 60k): the arms' observation noise at a tenth 7.7 / 11.0 mm
(and V9 at 220k evaluated with it 7.2 / 10.4); the integrated residual 5.4 / 9.1 mm, its joints at half V9's speed in
the hold; the fixed kernels 4.7 / 8.9 mm, its std rising to 0.48; the three together 6.3 / 9.8 mm; with the integral
under them (gain 0.02, clamp 0.05 rad, 5 s leak) 3.9 / 6.6 mm. The IK alone with the integral, no policy: 1.6 / 2.9 mm
at gain 0.01, clamp 0.3 rad, 60 s leak (16 / 21% of goals within 1 mm; the IK alone 3.5 / 8.2 mm, 0.6%); 2.4 / 5.1
mm with the 0.05 rad clamp, which saturated off the table. Every policy's std rose from its start (0.1-0.2) to
0.18-0.48 instead of falling; V10's arms train with the LCP-style smoothness penalty (agents yaml lipschitz_penalty).
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from locomanipulation_game.assets.h1_2 import CL_ASSETS_DIR

from . import mdp
from .locomanip_marl_env_cfg import FINE_POS_STDS, ROT_STDS, ArmsRewardsCfg, LocoManipMarlEnvCfg, MarlRewardsCfg

ARMS_OUTPUT_INIT_SCALE = 0.01
ARMS_INITIAL_STD = 0.2
ARM_JOINTS = ".*_(shoulder|elbow|wrist)_.*"
ARMS_TARGET_MARGIN = 0.1  # rad
ARMS_MIN_STD = 0.1
ARMS_RESIDUAL_SCALE_FACTOR = 0.5  # of the walking task's per-joint residual scale
# the walking task's kernel stages, then narrower ones
ARMS_FINE_POS_STDS = FINE_POS_STDS + (0.008, 0.005, 0.003)  # m
ARMS_ROT_STDS = ROT_STDS + (0.06, 0.045, 0.035)            # rad
ARMS_IK = {"rot_weight": 0.3, "null_gain": 0.05, "posture_scale": 0.5, "limit_margin": 0.05, "ik_passes": 3,
           "posture_from_command": True, "trajectory": True, "posture_actions": False,
           "trajectory_max_vel": 2.0, "trajectory_max_acc": 10.0, "residual_cutoff_hz": 1.0,
           "freeze_residual_on_goal": True, "freeze_pos_tol": 0.001, "freeze_rot_tol": math.radians(5.0),
           "gravity_offset": True, "integral_gain": 0.01, "integral_clamp": 0.3, "integral_leak_s": 60.0}
ARMS_JOINT_RANGE_MARGIN = 0.1  # rad
ARMS_TABLE_FILE = "arm_target_tables_armsonly_v2.pt"  # relative: built into the run's folder
ARMS_OFFTABLE_PROB = 0.5
# reached (the bonus, the reached flag and metric): both wrists held within these of the goal for reach_hold_s (user
# 11-13: 2 mm and 5 deg; 11-11: 1 mm, as the residual's freeze; the walking task's 5 cm / 0.35 rad)
ARMS_REACH_POS_TOL = 0.002  # m
ARMS_REACH_ROT_TOL = math.radians(5.0)
ARMS_OFFTABLE_SCALES = (0.25, 1.0)
ARMS_REST_GOAL_PROB = 0.1
ARMS_GOAL_TIME_RANGE = (4.0, 8.0)  # s: each goal's duration, drawn
# the walking task's shared observation terms the arms' actor doesn't get here
ARMS_DROPPED_OBS = ("base_lin_acc", "base_ang_vel", "projected_gravity", "velocity_commands", "gait_phase",
                    "gait_phase_cos")
ARMS_KINEMATICS_URDF = str(CL_ASSETS_DIR / "ros_assets" / "h1_2_magpie.urdf")
ARMS_KINEMATICS_SPHERES = str(CL_ASSETS_DIR / "curobo_assets" / "h1_2_magpie_curobo.yml")

# on the goal, each wrist moving at most ARMS_HOLD_STILL_TOL per policy step (0.1 mm: 1 cm/s at 100 Hz) is paid in full,
# e^-1 of it at ARMS_HOLD_STILL_SCALE beyond (mdp.arm_hold_still); at two tracking terms' weight. With a 0.2 mm scale
# (ArmsOnlyV4_s42, 11-11) the exploration noise alone left it at ~0 (0.03 of 2 per episode) and the sway unchanged at
# 40k; 2 mm (20 cm/s) still pays a slower hold more
ARMS_HOLD_STILL_TOL = 1e-4  # m per policy step
ARMS_HOLD_STILL_SCALE = 2e-3  # m per policy step
# on the goal for it: the walking task's reach tolerances, not ARMS_REACH_*
ARMS_HOLD_STILL_GATE = (0.05, 0.35)  # m, rad
ARMS_HOLD_STILL = RewTerm(func=mdp.arm_hold_still, weight=2.0, params={
    "command_name": "arm_targets", "tolerance": ARMS_HOLD_STILL_TOL, "scale": ARMS_HOLD_STILL_SCALE,
    "pos_tol": ARMS_HOLD_STILL_GATE[0], "rot_tol": ARMS_HOLD_STILL_GATE[1]})


def _fixed(func, arm: int, **params) -> RewTerm:
    """A tracking term off by default: ArmsOnlyEnvCfg.arms_precision_rewards weighs it."""
    return RewTerm(func=func, weight=0.0, params={"command_name": "arm_targets", "arm": arm, **params})


# arms_precision_rewards (ArmsOnlyEnvCfg): fixed position kernels (m) and rotation kernels (rad) in place of the
# annealed ones, weight 1 each, and the capped linear position term at ARMS_PRECISION_LINEAR per m
ARMS_PRECISION_POS_STDS = {"k20": 0.02, "k5": 0.005, "k2": 0.002}
ARMS_PRECISION_ROT_STDS = {"k200": 0.2, "k50": 0.05}
ARMS_PRECISION_LINEAR = -20.0
ARMS_PRECISION_LINEAR_CAP = 0.02  # m
ARMS_PRECISION_HOLD_TOLS = (0.005, 0.1)  # m, rad: hold_still's on-goal gate
ARMS_PRECISION_ACTION_RATE = -0.05
# arms_integrate (ArmsOnlyEnvCfg): the arms' exploration
ARMS_INTEGRATE_INITIAL_STD = 0.1
ARMS_INTEGRATE_MIN_STD = 0.03
# the arms' observation noise at arms_obs_noise_scale 1 (the walking task's). The measured wrist poses carry none
# (user 2026-11-19; the walking task's 0.005, as every run up to ArmsObs200k_s42)
ARMS_OBS_NOISE = {"wrist_errors": 0.005, "joint_pos": 0.01, "joint_vel": 1.5}


@configclass
class ArmsOnlyArmsRewardsCfg(ArmsRewardsCfg):
    hold_still = ARMS_HOLD_STILL
    left_ee_pos_k20 = _fixed(mdp.arm_target_pos_exp, 0, std=ARMS_PRECISION_POS_STDS["k20"])
    left_ee_pos_k5 = _fixed(mdp.arm_target_pos_exp, 0, std=ARMS_PRECISION_POS_STDS["k5"])
    left_ee_pos_k2 = _fixed(mdp.arm_target_pos_exp, 0, std=ARMS_PRECISION_POS_STDS["k2"])
    left_ee_quat_k200 = _fixed(mdp.arm_target_quat_exp, 0, std=ARMS_PRECISION_ROT_STDS["k200"])
    left_ee_quat_k50 = _fixed(mdp.arm_target_quat_exp, 0, std=ARMS_PRECISION_ROT_STDS["k50"])
    left_ee_pos_linear = _fixed(mdp.arm_target_pos_capped, 0, cap=ARMS_PRECISION_LINEAR_CAP)
    right_ee_pos_k20 = _fixed(mdp.arm_target_pos_exp, 1, std=ARMS_PRECISION_POS_STDS["k20"])
    right_ee_pos_k5 = _fixed(mdp.arm_target_pos_exp, 1, std=ARMS_PRECISION_POS_STDS["k5"])
    right_ee_pos_k2 = _fixed(mdp.arm_target_pos_exp, 1, std=ARMS_PRECISION_POS_STDS["k2"])
    right_ee_quat_k200 = _fixed(mdp.arm_target_quat_exp, 1, std=ARMS_PRECISION_ROT_STDS["k200"])
    right_ee_quat_k50 = _fixed(mdp.arm_target_quat_exp, 1, std=ARMS_PRECISION_ROT_STDS["k50"])
    right_ee_pos_linear = _fixed(mdp.arm_target_pos_capped, 1, cap=ARMS_PRECISION_LINEAR_CAP)


@configclass
class ArmsOnlyRewardsCfg(MarlRewardsCfg):
    arms: ArmsOnlyArmsRewardsCfg = ArmsOnlyArmsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # the legs' reward terms off (the env still charges them a fall and a GOLEM trip, which cannot move them here)
        for name, term in list(vars(self.legs).items()):
            if isinstance(term, RewTerm):
                setattr(self.legs, name, None)


@configclass
class ArmsOnlyEnvCfg(LocoManipMarlEnvCfg):
    rewards: ArmsOnlyRewardsCfg = ArmsOnlyRewardsCfg()

    # Variants for the command line (sync_derived applies them after the overrides; V9 had 1.0, False, False):
    arms_obs_noise_scale: float = 1.0
    """The arms' wrist-error and joint observation noise times this (ARMS_OBS_NOISE at 1)."""
    arms_integrate: bool = True
    """The residual integrated, not low-passed (IKPostureArmActionCfg.residual_integrate), the arms' std starting at
    ARMS_INTEGRATE_INITIAL_STD with a floor of ARMS_INTEGRATE_MIN_STD."""
    arms_precision_rewards: bool = True
    """Fixed kernels (ARMS_PRECISION_POS_STDS / _ROT_STDS) and the capped linear term in place of the annealed fine
    position and rotation kernels, hold_still gated at ARMS_PRECISION_HOLD_TOLS, action_rate at
    ARMS_PRECISION_ACTION_RATE."""

    def __post_init__(self):
        super().__post_init__()
        # the pelvis pinned, upright at the stance height
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        pose = self.events.reset_base.params["pose_range"]
        pose["z"], pose["roll"], pose["pitch"] = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        # the legs at their default pose
        self.actions.joint_pos.scale = 0.0
        # standing arm goals only, from the first step, at full residual
        self.commands.arm_targets.arm_goal_prob = 1.0
        self.commands.arm_targets.arm_goal_ramp_steps = 0
        self.warm_start_deadline_steps = 1
        self.policy_output_init_scale = {"arms": ARMS_OUTPUT_INIT_SCALE}
        # the arm joints' targets ARMS_TARGET_MARGIN inside GOLEM's edges, not 0.02 rad
        self.golem_estop.joint_position_margins[ARM_JOINTS] = (ARMS_TARGET_MARGIN, ARMS_TARGET_MARGIN)
        # the limit-aware, posture-seeking IK, the policy setting the posture too
        fields = {k: v for k, v in vars(self.actions.arm_pos).items() if k != "class_type"}
        self.actions.arm_pos = mdp.IKPostureArmActionCfg(**{**fields, **ARMS_IK})
        self.actions.arm_pos.scale = {k: v * ARMS_RESIDUAL_SCALE_FACTOR for k, v in self.actions.arm_pos.scale.items()}
        # goals from joint angles off the limits, nearest the rest pose first
        arm = self.commands.arm_targets
        arm.joint_range_margin = ARMS_JOINT_RANGE_MARGIN
        arm.table_joints = True
        # the goals' standing frame at the pinned pelvis's own height: at the tables' stance (1.0024 m) over the sport
        # stance it is pinned at (0.9798 m), every goal sat 2.3 cm higher in the pelvis frame than its table pose
        arm.standing_height = self.scene.robot.init_state.pos[2]
        arm.table_file = ARMS_TABLE_FILE
        arm.offtable_prob, arm.offtable_scale_range = ARMS_OFFTABLE_PROB, ARMS_OFFTABLE_SCALES
        # each goal's seed solved for the goal itself, goals whose seed or path collide drawn again, some rest goals
        arm.seed_solve, arm.collision_check, arm.rest_goal_prob = True, True, ARMS_REST_GOAL_PROB
        arm.reach_pos_tol, arm.reach_rot_tol = ARMS_REACH_POS_TOL, ARMS_REACH_ROT_TOL
        arm.resampling_time_range = ARMS_GOAL_TIME_RANGE
        arm.fine_pos_stds, arm.rot_stds = ARMS_FINE_POS_STDS, ARMS_ROT_STDS
        arm.kinematics_urdf, arm.kinematics_spheres = ARMS_KINEMATICS_URDF, ARMS_KINEMATICS_SPHERES
        # every task is an arm goal: the observations drop the flag that says so
        for group in (self.observations.arms, self.observations.legs, self.observations.critic):
            group.arm_goal = None
        for name in ARMS_DROPPED_OBS:
            setattr(self.observations.arms, name, None)
        self.observations.arms.wrist_poses.noise = None
        self.sync_derived()

    def sync_derived(self):
        super().sync_derived()
        if not isinstance(self.actions.arm_pos, mdp.IKPostureArmActionCfg):
            return  # the base class's __post_init__, before this one's
        for name, noise in ARMS_OBS_NOISE.items():
            term = getattr(self.observations.arms, name)
            term.noise.n_min, term.noise.n_max = -noise * self.arms_obs_noise_scale, noise * self.arms_obs_noise_scale
        self.actions.arm_pos.residual_integrate = self.arms_integrate
        initial_std, min_std = ((ARMS_INTEGRATE_INITIAL_STD, ARMS_INTEGRATE_MIN_STD) if self.arms_integrate
                                else (ARMS_INITIAL_STD, ARMS_MIN_STD))
        self.policy_initial_std = {"arms": initial_std}
        self.policy_log_std_bounds = {"arms": (math.log(min_std), 0.0)}
        arms, on = self.rewards.arms, self.arms_precision_rewards
        for side in ("left", "right"):
            getattr(arms, f"{side}_ee_pos_fine").weight = 0.0 if on else 1.0
            getattr(arms, f"{side}_ee_quat").weight = 0.0 if on else 1.0
            for k in (*ARMS_PRECISION_POS_STDS, *ARMS_PRECISION_ROT_STDS):
                kind = "pos" if k in ARMS_PRECISION_POS_STDS else "quat"
                getattr(arms, f"{side}_ee_{kind}_{k}").weight = 1.0 if on else 0.0
            getattr(arms, f"{side}_ee_pos_linear").weight = ARMS_PRECISION_LINEAR if on else 0.0
        arms.hold_still.params["pos_tol"], arms.hold_still.params["rot_tol"] = (
            ARMS_PRECISION_HOLD_TOLS if on else ARMS_HOLD_STILL_GATE)
        arms.action_rate.weight = ARMS_PRECISION_ACTION_RATE if on else ArmsRewardsCfg().action_rate.weight
