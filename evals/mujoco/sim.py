"""The two-agent policy loop of LocoManip-Marl-Direct-v0 ported to MuJoCo, for sim-to-sim checks without ROS.

Mirrors, step for step, what the Isaac env computes (locomanip_marl_env.py, mdp/actions.py, the observation terms),
for an exported policy and its deploy settings (`settings`: the export's <name>.yaml with export.yaml over it):

    policy step       control_dt; actions -> joint targets; MuJoCo steps of 2 ms with PD torques recomputed each
                      step; then the actors' observations (with the training noise) for the next step
    legs action       target = default_joint_pos + leg_action_scales a, inside [hard lower + 1 mm, hard upper - 1 mm]
                      and the run's target_limits, then inside the targets whose PD torque at the measured state stays
                      within 85% of the effort limit
    arms action       one damped-least-squares IK step per arm toward the believed command (6-D pose error
                      [p_t - p, axis-angle(q_t q^-1)] in the pelvis frame; the wrist Jacobian in the pelvis frame at the
                      link's centre of mass, as PhysX computes it; lambda 0.05; capped at max_ik_step), plus the
                      residual low-passed at arms.residual_cutoff_hz times arm_residual_scales, with the same bounds;
                      with arm_ik (mdp.IKPostureArmAction, LocoManip-ArmsOnly-v0) the step is that term's instead:
                      position weighted over rotation, joints locked short of their bounds, the null-space pull toward
                      the default angles plus the action's second half (the posture offset), see _posture_step
    PD                tau = kp (target - q) - kd qd, clipped to the effort limit (kp, kd from the export)
    command           vel_cmd through the run's command_filter_tau low-pass (zero during an arm goal); the gait clock
                      stands under zero_cmd_eps
    observations      legs 97 / arms 125 values, term by term in the cfg's order: the run's IMU (the MID-360 chip or
                      the torso) with its gyro low-pass sampled at every MuJoCo step, the gait phase's sin and cos
    arm command       moved by the pelvis's true motion during an arm goal (FAST-LIO's on the robot)

Joint limits for the target bounds are the URDF's (as Isaac imported them, and as GOLEM's safety layer clips); the
MJCF's own ranges act as the physical stops.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import common as C

LEG_NAMES = ["left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
             "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
             "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
ARM_NAMES = [f"{s}_{j}_joint" for s in ("left", "right") for j in
             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
ALL_NAMES = LEG_NAMES + ["torso_joint"] + ARM_NAMES                    # the observations' joint order
# effort limits (Nm), assets/h1_2.py's, by the joint name's part
EFFORT = {"hip": 200.0, "knee": 300.0, "ankle_pitch": 60.0, "ankle_roll": 40.0, "torso": 200.0, "shoulder_pitch": 40.0,
          "shoulder_roll": 40.0, "shoulder_yaw": 18.0, "elbow": 18.0, "wrist": 19.0}


def _effort(name: str) -> float:
    for key, value in EFFORT.items():
        if f"_{key}_" in name or name.startswith(f"{key}_"):
            return value
    raise KeyError(name)


def _arm_kinematics():
    """The task's mdp/arm_kinematics.py (plain torch), loaded by path: the task package's __init__ needs Isaac Lab."""
    import importlib.util
    path = (C.REPO / "source" / "locomanipulation_game" / "locomanipulation_game" / "tasks" / "locomanip_marl" / "mdp"
            / "arm_kinematics.py")
    spec = importlib.util.spec_from_file_location("arm_kinematics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joint_noise():
    """The task's joint_noise.py (the encoders' noise, plain torch here), loaded by path as _arm_kinematics."""
    import importlib.util
    path = (C.REPO / "source" / "locomanipulation_game" / "locomanipulation_game" / "tasks" / "locomanip_marl"
            / "joint_noise.py")
    spec = importlib.util.spec_from_file_location("joint_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JN = _joint_noise()


def mjcf_path(assets: Path = C.DEFAULT_ASSETS) -> Path:
    return assets / "mujoco_assets" / "h1_2_magpie.xml"


PIN_WELD = "pelvis_pin"


def with_pelvis_weld(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """spec with the pelvis welded to the world (PIN_WELD), which MarlMujoco(pinned=True) places at each reset. Setting
    the pelvis's qpos back after each step alone leaves the robot falling freely within the step: the arms near
    weightless (shoulder pitch held by 0.9 of its 8 Nm of gravity)."""
    eq = spec.add_equality()
    eq.name, eq.type, eq.objtype, eq.name1 = PIN_WELD, mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    eq.solref, eq.solimp = [0.004, 1.0], [0.99, 0.999, 0.001, 0.5, 2.0]
    return spec


def pinned_model(assets: Path = C.DEFAULT_ASSETS) -> mujoco.MjModel:
    """The CL_Assets MJCF compiled with the pelvis weld, for MarlMujoco(pinned=True)."""
    return with_pelvis_weld(mujoco.MjSpec.from_file(str(mjcf_path(assets)))).compile()


def urdf_limits(urdf: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is not None and j.get("type") in ("revolute", "prismatic"):
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z), numpy

def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.r_[0.0, v]), qconj(q))[1:]


def qunique(q):
    return -q if q[0] < 0 else q


def axis_angle(q):
    """Isaac Lab's axis_angle_from_quat: rotation vector of q (w first), shortest way round."""
    q = qunique(q / np.linalg.norm(q))
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return 2.0 * q[1:]
    angle = 2.0 * math.atan2(s, q[0])
    return q[1:] / s * angle


def relative(pos_a, quat_a, pose):
    """T_a^-1 o pose for pose (7,) = (x y z qw qx qy qz); qw >= 0."""
    qi = qconj(quat_a)
    return np.r_[qrot(qi, pose[:3] - pos_a), qunique(qmul(qi, pose[3:]))]


def xyzw(pose):
    return np.r_[pose[:3], pose[4:7], pose[3]]


# ---------------------------------------------------------------- the sim


def imu_bias_ranges(imu: dict) -> dict:
    """The torso IMU's trained orientation bias ranges in an export's imu block (deg, roll / pitch / yaw: [lo, hi]):
    the calibration's error for imu.orientation calibrated (residual_deg), the raw offset otherwise (quat_bias_deg)."""
    calibrated = imu.get("orientation", "raw") == "calibrated"
    return {k: tuple(v) for k, v in imu["residual_deg" if calibrated else "quat_bias_deg"].items()}

class GyroLowpass:
    """lmg_deploy.livox_imu.GyroLowpass: per axis, a 2nd-order Butterworth at cutoff_hz, exact ZOH discretization over
    each sample's dt, started (and restarted by reset) at x = (u, 0)."""

    ZETA = 1.0 / math.sqrt(2.0)

    def __init__(self, cutoff_hz: float):
        w = 2.0 * math.pi * float(cutoff_hz)
        self._a = np.array([[0.0, 1.0], [-w * w, -2.0 * self.ZETA * w]])
        self._b = np.array([0.0, w * w])
        self._sigma, self._wd, self._w2 = self.ZETA * w, w * math.sqrt(1.0 - self.ZETA ** 2), w * w
        self._x = None

    def reset(self):
        self._x = None

    def __call__(self, u: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        if self._x is None:
            self._x = np.stack([u, np.zeros_like(u)])
            return u.copy()
        e, c, sn = math.exp(-self._sigma * dt), math.cos(self._wd * dt), math.sin(self._wd * dt)
        k = self._sigma / self._wd
        ad = e * np.array([[c + k * sn, sn / self._wd], [-self._w2 * sn / self._wd, c - k * sn]])
        bd = np.linalg.solve(self._a, (ad - np.eye(2)) @ self._b)
        self._x = ad @ self._x + np.outer(bd, u)
        return self._x[0].copy()


# The robot's motor drivers report the velocity of these joints through a 2nd-order Butterworth low-pass at this cutoff
# (~12 ms late; the ankles and arms unfiltered): the training env's JOINT_VEL_LOWPASS_HZ / _JOINTS
# (locomanip_marl_env_cfg.py), which an export records as joint_vel_lowpass. stress_mujoco's hw condition applies it.
ROBOT_JOINT_VEL_LOWPASS = {"cutoff_hz": 21.0, "joints": [".*_hip_(yaw|pitch|roll)_joint", ".*_knee_joint", "torso_joint"]}


class MarlMujoco:
    """One H1-2 in MuJoCo driven by an export's actors (callables obs[None] -> action[None]) and its settings."""

    # uniform +-: the training's observation noise; the joint speeds' is the encoders' (joint_noise.py: Gaussian, its std
    # growing with the speed; jvel scales it, 0 off). Until 2026-11-23 jpos 0.01 and jvel uniform 1.5, the training's
    # then: MuJoCo's targets jittered ~0.017 rad per step standing on it, the robot's 0.001-0.002
    NOISE = {"lin_acc": 0.5, "ang_vel": 0.2, "grav": 0.05, "jpos": JN.JOINT_POS_NOISE, "jvel": 1.0, "wrist": 0.005}
    JVEL_FLOOR = np.array(JN.per_joint(ALL_NAMES, JN.JOINT_VEL_NOISE_FLOOR, JN.JOINT_VEL_NOISE_FLOOR_DEFAULT))

    def __init__(self, actors: dict, settings: dict, physics: str = "robocasa", assets: Path = C.DEFAULT_ASSETS,
                 model: mujoco.MjModel | None = None, pinned: bool = False, arms_hold_zero: bool = False):
        # model: the CL_Assets MJCF compiled with changes that leave the robot's joints untouched; default as it is.
        # pinned: the pelvis held still where each reset leaves it, upright (LocoManip-ArmsOnly-v0's fix_root_link), by
        # a weld (with_pelvis_weld; a model passed in needs it, pinned_model).
        # actors without "legs": no legs policy, the legs held at their default pose; without "arms": the arms' residual
        # 0, the IK alone; arms_hold_zero (evals: a legs eval with no arms policy): every arm joint's target 0 rad, no IK
        self.pinned, self._pin = pinned, None
        self.arms_hold_zero = arms_hold_zero
        if model is None:
            model = pinned_model(assets) if pinned else mujoco.MjModel.from_xml_path(str(mjcf_path(assets)))
        self.m = model
        self._pin_eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, PIN_WELD)
        if pinned and self._pin_eq < 0:
            raise ValueError("pinned needs the pelvis weld: build the model with pinned_model() or with_pelvis_weld()")
        if self._pin_eq >= 0:
            model.eq_active0[self._pin_eq] = pinned
        self.d = mujoco.MjData(self.m)
        m = self.m
        arms_cfg = settings["arms"]
        self.policy_dt = float(settings["control_dt"])
        self.substeps = int(round(self.policy_dt / m.opt.timestep))
        # lmg_deploy's engage blend (export.yaml engage_blend_s; absent: none): over the first steps after a reset the
        # legs' target goes from the last command sent (blend_from) to the policy's, linearly, then is bounded
        self.blend_steps = int(round(float(settings.get("engage_blend_s") or 0.0) / self.policy_dt))
        cutoff = float(settings.get("residual_cutoff_hz", arms_cfg["residual_cutoff_hz"]))  # the run's, else the template's
        self.residual_alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff * self.policy_dt)
        self.ik_damping = float(settings.get("ik_damping", arms_cfg.get("ik_damping", 0.05)))
        self.clip_actions = float(settings.get("clip_actions", 10.0))
        self.leg_scale = np.asarray(settings.get("leg_action_scales", settings["legs"]["action_scale"]), float) * np.ones(12)
        if "legs" not in actors:
            self.leg_scale = np.zeros(12)
        self.arm_scale = np.asarray(settings.get("arm_residual_scales", arms_cfg["residual_scale"]), float) * np.ones(14)
        self.max_ik_step = float(settings.get("max_ik_step", arms_cfg["max_ik_step"]))
        # mdp.IKPostureArmAction's IK (export.yaml arm_ik), its action twice as wide: the residual, then the posture
        # (posture_actions False: the residual alone)
        self.arm_ik = settings.get("arm_ik")
        self.n_arm_actions = 28 if self.arm_ik and self.arm_ik.get("posture_actions", True) else 14
        # IKPostureArmAction's freeze_residual_on_goal: the residual latched once the wrists settle on the command
        self.freeze = (self.arm_ik or {}).get("freeze")
        # its residual_integrate ({gain, leak_s, clamp}: the residual integrated, not low-passed) and integral ({gain,
        # leak_s, clamp}: the measured wrists' pose error integrated onto the target once it has stopped)
        self.residual_integrate = (self.arm_ik or {}).get("residual_integrate")
        self.integral_cfg = (self.arm_ik or {}).get("integral")
        # its posture from the goal table's row nearest the command (export's arm_seed_table, an absolute path here)
        self.seed_table = None
        if self.arm_ik and self.arm_ik.get("posture_from_command"):
            table = np.load(settings["arm_seed_table"])
            self.seed_table = [(table[f"poses_{i}"].astype(float), table[f"joints_{i}"].astype(float)) for i in range(2)]
        # ArmTargetsCommandCfg.seed_solve: the nearest row's angles solved for the command itself (mdp.arm_kinematics)
        self.seed_solve, self._kin, self._seeds_for = (self.arm_ik or {}).get("seed_solve"), None, None
        # IKPostureArmActionCfg.gravity_offset: the arm joints' gravity torque at the target over kp (the same kinematics)
        self.gravity_offset = bool((self.arm_ik or {}).get("gravity_offset"))
        if self.seed_solve or self.gravity_offset:
            self._urdf = assets / "ros_assets" / "h1_2_magpie.urdf"
            self._kin = _arm_kinematics().ArmKinematics(self._urdf,
                                                        assets / "curobo_assets" / "h1_2_magpie_curobo.yml",
                                                        dtype=torch.float64)
        tau = float(settings.get("command_filter_tau") or 0.0)
        self.cmd_alpha = 1.0 - math.exp(-self.policy_dt / tau) if tau > 0 else None
        self.gait_period, self.zero_cmd_eps = float(settings["gait_period"]), float(settings["zero_cmd_eps"])
        self.gait_cos = bool(settings.get("gait_phase_cos", False))
        # each actor's observation terms by name (read_deploy_settings); None: the walking task's layout
        self.legs_terms, self.arms_terms = settings.get("legs_obs_terms"), settings.get("arms_obs_terms")
        jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ALL_NAMES}
        self.qadr = np.array([m.jnt_qposadr[jid[n]] for n in ALL_NAMES])
        self.vadr = np.array([m.jnt_dofadr[jid[n]] for n in ALL_NAMES])
        self.leg_idx = np.arange(12)
        self.arm_idx = np.arange(13, 27)
        self.default = np.array(settings["default_joint_pos"], float)
        self.kp, self.kd = np.array(settings["kp"], float), np.array(settings["kd"], float)
        self.effort = np.array([_effort(n) for n in ALL_NAMES])
        lim = urdf_limits(assets / "ros_assets" / "h1_2_magpie.urdf")
        self.lower = np.array([lim[n][0] for n in ALL_NAMES]) + 0.001
        self.upper = np.array([lim[n][1] for n in ALL_NAMES]) - 0.001
        # the run's own target clip, GOLEM's position e-stop range (export.yaml target_limits)
        target_limits = settings.get("target_limits") or {}
        for i, n in enumerate(ALL_NAMES):
            if n in target_limits:
                low, high = target_limits[n]
                self.lower[i], self.upper[i] = max(self.lower[i], low), min(self.upper[i], high)
        self.act_id = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ALL_NAMES])
        if (self.act_id < 0).any():
            raise RuntimeError("MJCF lacks a motor for some joint")
        if physics == "isaac":
            # the training plant's joint dynamics: no passive damping or friction, armature 0.01 (h1_2.py)
            for n in ALL_NAMES:
                a = m.jnt_dofadr[jid[n]]
                m.dof_damping[a], m.dof_armature[a], m.dof_frictionloss[a] = 0.0, 0.01, 0.0
            # and the grippers' weight: the MJCF gives them gravcomp 1, Isaac's USD 9 x 0.05 kg per gripper
            m.body_gravcomp[:] = 0.0
        self.physics = physics
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.pelvis = body("pelvis")
        self.wrists = [body("left_wrist_yaw_link"), body("right_wrist_yaw_link")]
        self.feet = [body("left_ankle_roll_link"), body("right_ankle_roll_link")]
        # the run's IMU: the head's MID-360 chip, or the torso's
        imu = settings.get("imu") or {}
        head = imu.get("frame", "imu_link") != "imu_link"
        sens = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]  # noqa: E731
        self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "livox_imu_site" if head else "imu")
        self.s_acc, self.s_gyro = (sens("livox_imu_acc"), sens("livox_imu_gyro")) if head else (sens("imu_acc"), sens("imu_gyro"))
        self.foot_geoms = [set(np.nonzero(m.geom_bodyid == f)[0].tolist()) for f in self.feet]
        self.arm_cols = [self.vadr[13:20], self.vadr[20:27]]
        self.actors = actors
        self.rest_b = np.array(settings["rest_pose"], float)
        self.rng = np.random.default_rng(0)
        self.gyro_lowpass = GyroLowpass(imu["gyro_lowpass_hz"]) if imu.get("gyro_lowpass_hz") else None
        self._gyro = None
        # the reported joint velocities' low-pass the run trained with (export joint_vel_lowpass; absent: none)
        self.set_dq_lowpass(settings.get("joint_vel_lowpass"))
        # LocoManip-Marl-Direct-v0 (ArmTargetsCommandCfg.pelvis_fixed_goals, independent_of_legs): an arm goal rides
        # on the pelvis, and the velocity command runs through it. The caller sets it.
        self.pelvis_goals = False
        # /lowstate's torso IMU (export imu.source lowstate): its quaternion, as the policy reads it, reads this
        # rotation when level (the trained range's centre; set imu_bias_deg to move it): for imu.orientation
        # calibrated what lmg_deploy's calibration leaves (residual_deg, 0 at its centre), for raw the offset itself
        # (quat_bias_deg); its noise is the model's, not the head's
        self.lowstate = imu.get("source") == "lowstate"
        self.imu_bias_deg = ({k: 0.5 * (lo + hi) for k, (lo, hi) in imu_bias_ranges(imu).items()} if self.lowstate
                             else {"roll": 0.0, "pitch": 0.0, "yaw": 0.0})
        if self.lowstate:
            # the gyro's per-sample noise and the quaternion's per-step noise rotation (0.05 deg) of the IMU model
            self.NOISE = {**self.NOISE, "ang_vel": 0.03, "grav": 0.00087}

    # -- state
    def reset(self, rng: np.random.Generator):
        """On the ground near the default pose, nearly still, from a fresh episode (gait clock, filters, residual)."""
        m, d = self.m, self.d
        self.rng = rng
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = self.default + rng.uniform(-0.05, 0.05, len(ALL_NAMES))
        d.qvel[self.vadr] = rng.uniform(-0.1, 0.1, len(ALL_NAMES))
        roll, pitch = (0.0, 0.0) if self.pinned else rng.uniform(-0.05, 0.05, 2)
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        d.qpos[3:7] = qmul(np.array([cr, sr, 0, 0]), np.array([cp, 0, sp, 0]))
        d.qpos[0:3] = (0.0, 0.0, 1.0)
        mujoco.mj_forward(m, d)
        sole = min(d.xpos[f][2] for f in self.feet) + C.SOLE_Z
        d.qpos[2] += -sole + rng.uniform(0.0, 0.02)
        d.qvel[0:6] = np.r_[rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.05, 0.05), rng.uniform(-0.1, 0.1, 3)]
        if self.pinned:
            d.qvel[0:6] = 0.0
            self._pin = d.qpos[0:7].copy()
            # the weld at the pin: its anchor (a world point) in the pelvis frame, and the world's rotation in it
            anchor = m.eq_data[self._pin_eq, 0:3]
            m.eq_data[self._pin_eq, 3:6] = qrot(qconj(self._pin[3:7]), anchor - self._pin[0:3])
            m.eq_data[self._pin_eq, 6:10] = qconj(self._pin[3:7])
        mujoco.mj_forward(m, d)
        self.applied = d.qpos[self.qadr].copy()
        self.residual = np.zeros(self.n_arm_actions)
        self.blend_from = self.applied[self.leg_idx].copy()  # the target held before the first step: here the pose
        self.ik_q = d.qpos[self.qadr][self.arm_idx].copy()  # arm_ik.trajectory: the target's own last value
        self.traj_v = np.zeros(14)  # arm_ik.trajectory_max_vel: the target's velocity (rad/s)
        self.frozen, self.settled_steps = False, 0  # arm_ik.freeze
        self.integral = np.zeros(14)  # arm_ik.integral: the pose error's integral (rad)
        self._seeds_for = None
        self.arm_mode = False
        self.believed = self.rest_b.copy()
        self.vel_cmd = np.zeros(3)
        self.cmd_filtered = np.zeros(3)
        self.steps = 0
        self._prev_root = (d.qpos[0:3].copy(), d.qpos[3:7].copy())
        if self.gyro_lowpass is not None:
            self.gyro_lowpass.reset()
            self._gyro = self.gyro_lowpass(self._raw_gyro(), m.opt.timestep)
        if self.dq_lowpass is not None:
            self.dq_lowpass.reset()
            self._dq_filtered = None
            self._sample_dq()

    # -- kinematics in the pelvis frame, sensors
    def root(self):
        return self.d.qpos[0:3].copy(), self.d.qpos[3:7].copy()

    def wrist_poses_b(self) -> np.ndarray:
        p, q = self.root()
        return np.stack([relative(p, q, np.r_[self.d.xpos[w], self.d.xquat[w]]) for w in self.wrists])

    def _raw_gyro(self) -> np.ndarray:
        return self.d.sensordata[self.s_gyro:self.s_gyro + 3].copy()

    def imu(self):
        d = self.d
        acc = d.sensordata[self.s_acc:self.s_acc + 3].copy()
        gyro = self._raw_gyro() if self.gyro_lowpass is None else self._gyro.copy()
        R = d.site_xmat[self.imu_site].reshape(3, 3)
        b = {k: math.radians(v) for k, v in self.imu_bias_deg.items()}
        if any(b.values()):  # the reported orientation: the true one times the bias rotation (body frame, x-y-z)
            cr, sr, cp, sp, cy, sy = (f(b[k]) for k in ("roll", "pitch", "yaw") for f in (math.cos, math.sin))
            rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R = R @ (rz @ ry @ rx)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return acc, gyro, grav

    def measured_q(self) -> np.ndarray:
        """The joint angles the policy and the torque bound see (a subclass adds encoder offsets)."""
        return self.d.qpos[self.qadr].copy()

    def set_dq_lowpass(self, spec: dict | None):
        """Low-pass the reported velocity of spec["joints"] (name patterns) at spec["cutoff_hz"] (GyroLowpass, sampled
        every physics substep, as /lowstate's motor drivers do); None: report them unfiltered."""
        self.dq_lowpass, self.dq_lowpass_idx, self._dq_filtered = None, [], None
        if spec and float(spec.get("cutoff_hz") or 0.0) > 0.0:
            pats = spec.get("joints") or ROBOT_JOINT_VEL_LOWPASS["joints"]
            self.dq_lowpass_idx = [i for i, n in enumerate(ALL_NAMES) if any(re.fullmatch(p, n) for p in pats)]
            self.dq_lowpass = GyroLowpass(float(spec["cutoff_hz"]))

    def _sample_dq(self):
        """One physics substep's sample into the velocity low-pass."""
        if self.dq_lowpass is not None:
            self._dq_filtered = self.dq_lowpass(self.d.qvel[self.vadr][self.dq_lowpass_idx], self.m.opt.timestep)

    def measured_dq(self) -> np.ndarray:
        """The joint velocities the policy sees: the low-passed ones at the filter's output (a subclass adds
        LowState's lag)."""
        qd = self.d.qvel[self.vadr].copy()
        if self.dq_lowpass is not None:
            if self._dq_filtered is None:
                self._sample_dq()
            qd[self.dq_lowpass_idx] = self._dq_filtered
        return qd

    def pd_torque(self, target):
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        return np.clip(self.kp * (target - q) - self.kd * qd, -self.effort, self.effort)

    def _torque_bounds(self, idx):
        """The deploy's bound: the nominal gains on the measured state."""
        q, qd = self.measured_q()[idx], self.d.qvel[self.vadr][idx]
        h = 0.85 * self.effort[idx]
        return q + (self.kd[idx] * qd - h) / self.kp[idx], q + (self.kd[idx] * qd + h) / self.kp[idx]

    # -- actions
    def process_actions(self, a_legs: np.ndarray, a_arms: np.ndarray):
        clip = self.clip_actions
        a_legs, a_arms = np.clip(a_legs, -clip, clip), np.clip(a_arms, -clip, clip)
        idx = self.leg_idx
        t = self.default[idx] + self.leg_scale * a_legs
        self.policy_target = t.copy()  # the policy's own leg targets, before the blend and the bounds
        if self.steps < self.blend_steps:
            t = self.blend_from + (self.steps + 1) / self.blend_steps * (t - self.blend_from)
        t = np.clip(t, self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        # arms
        if self.arms_hold_zero:
            self.applied[self.arm_idx] = np.clip(0.0, self.lower[self.arm_idx], self.upper[self.arm_idx])
            self.applied[12] = 0.0
            return
        if not self.frozen:
            if self.residual_integrate:
                c = self.residual_integrate
                leak = math.exp(-self.policy_dt / c["leak_s"])
                self.residual = np.clip(leak * self.residual + c["gain"] * a_arms, -c["clamp"], c["clamp"])
            else:
                self.residual = self.residual + self.residual_alpha * (a_arms - self.residual)
        m, d = self.m, self.d
        p, qr = self.root()
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, qr)
        Rr = Rr.reshape(3, 3)
        ik = d.qpos[self.qadr][self.arm_idx].copy()
        if self.arm_ik and self.arm_ik.get("trajectory"):
            # IKPostureArmAction's trajectory: the target from its own last value straight to the posture, no IK
            c, idx = self.arm_ik, self.arm_idx
            low, high = self.lower[idx] + c["limit_margin"], self.upper[idx] - c["limit_margin"]
            base = np.concatenate([self.seed_q(0), self.seed_q(1)]) if self.seed_table is not None else self.default[idx]
            offset = self.residual[14:] if self.n_arm_actions == 28 else 0.0
            posture = np.clip(base + c["posture_scale"] * c["posture_gain"] * offset, low, high)
            if c.get("trajectory_max_vel") is None:
                ik = self.ik_q + np.clip(posture - self.ik_q, -self.max_ik_step, self.max_ik_step)
            else:
                ik = self._synchronized_step(posture)
            self.ik_q = np.clip(ik, self.lower[idx], self.upper[idx])
            if self.integral_cfg:
                self._update_integral(posture, Rr)
            t = np.clip(self.ik_q + self.gravity_target_offset(Rr) + self.integral + self.arm_scale * self.residual[:14],
                        self.lower[idx], self.upper[idx])
            lo, hi = self._torque_bounds(idx)
            self.applied[idx] = np.clip(t, lo, hi)
            self.applied[12] = 0.0
            if self.freeze:
                self._update_freeze(posture)
            return
        cur = self.wrist_poses_b()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for arm in range(2):
            # PhysX's articulation Jacobian is taken at the link's centre of mass (the task's pose error is at the
            # link origin); mj_jacBodyCom reproduces Isaac's IK step
            mujoco.mj_jacBodyCom(m, d, jacp, jacr, self.wrists[arm])
            J = np.vstack([Rr.T @ jacp[:, self.arm_cols[arm]], Rr.T @ jacr[:, self.arm_cols[arm]]])
            tgt = self.believed[arm]
            err = np.r_[tgt[:3] - cur[arm, :3], axis_angle(qmul(tgt[3:], qconj(cur[arm, 3:])))]
            sl = slice(7 * arm, 7 * arm + 7)
            if self.arm_ik:
                ik[sl] = ik[sl] + self._posture_step(J, err, ik[sl], self.residual[14:][sl], self.arm_idx[sl])
                continue
            step = J.T @ np.linalg.solve(J @ J.T + self.ik_damping ** 2 * np.eye(6), err)
            ik[sl] = ik[sl] + np.clip(step, -self.max_ik_step, self.max_ik_step)
        idx = self.arm_idx
        self.ik_q = np.clip(ik, self.lower[idx], self.upper[idx])  # where trajectory would start from
        t = np.clip(ik + self.arm_scale * self.residual[:14], self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        self.applied[12] = 0.0                                    # torso held at default by its own actuator

    def _update_integral(self, posture: np.ndarray, R_pelvis: np.ndarray):
        """mdp.IKPostureArmAction._update_integral: leak the pose-error integral; where the target has stopped at the
        posture, add gain x each arm's weighted DLS step on its measured wrist's error; clamp."""
        c, ik = self.integral_cfg, self.arm_ik
        arrived = np.abs(self.ik_q - posture).max() < 1e-4 and np.abs(self.traj_v).max() < 1e-3
        self.integral *= math.exp(-self.policy_dt / c["leak_s"])
        if arrived:
            m, d = self.m, self.d
            cur = self.wrist_poses_b()
            weight = np.r_[np.ones(3), np.full(3, ik["rot_weight"])]
            jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            for arm in range(2):
                mujoco.mj_jacBodyCom(m, d, jacp, jacr, self.wrists[arm])  # as PhysX's (process_actions' comment)
                J = np.vstack([R_pelvis.T @ jacp[:, self.arm_cols[arm]], R_pelvis.T @ jacr[:, self.arm_cols[arm]]])
                tgt = self.believed[arm]
                err = np.r_[tgt[:3] - cur[arm, :3], axis_angle(qmul(tgt[3:], qconj(cur[arm, 3:])))]
                Jw = J * weight[:, None]
                step = Jw.T @ np.linalg.solve(Jw @ Jw.T + ik["ik_damping"] ** 2 * np.eye(6), err * weight)
                self.integral[7 * arm:7 * arm + 7] += c["gain"] * step
        self.integral = np.clip(self.integral, -c["clamp"], c["clamp"])

    def _update_freeze(self, posture: np.ndarray):
        """mdp.IKPostureArmAction._update_freeze: latch the residual once the target has stopped at the posture and both
        wrists have been inside the tolerances of the believed command for settle_s; release when the posture moves."""
        f = self.freeze
        arrived = np.abs(self.ik_q - posture).max() < 1e-4 and np.abs(self.traj_v).max() < 1e-3
        self.frozen = self.frozen and arrived and self.arm_mode
        cur = self.wrist_poses_b()
        on_goal = self.arm_mode and all(
            np.linalg.norm(cur[a, :3] - self.believed[a, :3]) < f["pos_tol"]
            and np.linalg.norm(axis_angle(qmul(self.believed[a, 3:], qconj(cur[a, 3:])))) < f["rot_tol"]
            for a in range(2))
        self.settled_steps = self.settled_steps + 1 if (arrived and on_goal and not self.frozen) else 0
        self.frozen = self.frozen or self.settled_steps >= max(1, int(round(f["settle_s"] / self.policy_dt)))

    def _synchronized_step(self, posture: np.ndarray) -> np.ndarray:
        """mdp.IKPostureArmAction._synchronized_step: per arm, all joints along the straight line to the posture, the
        fastest at trajectory_max_vel, slowed to stop there, the velocity changed by at most trajectory_max_acc."""
        c, dt = self.arm_ik, self.policy_dt
        vmax, amax = c["trajectory_max_vel"], c.get("trajectory_max_acc")
        ik = self.ik_q.copy()
        for arm in range(2):
            sl = slice(7 * arm, 7 * arm + 7)
            delta = posture[sl] - self.ik_q[sl]
            dist = max(np.abs(delta).max(), 1e-9)
            cap = min(vmax, math.sqrt(2.0 * amax * dist)) if amax is not None else vmax
            v_want = delta / dt * min(1.0, cap * dt / dist)
            if amax is not None:
                dv = v_want - self.traj_v[sl]
                v = self.traj_v[sl] + dv * min(1.0, amax * dt / max(np.abs(dv).max(), 1e-12))
            else:
                v = v_want
            self.traj_v[sl] = v
            ik[sl] = self.ik_q[sl] + v * dt
        return ik

    def gravity_target_offset(self, R_pelvis: np.ndarray) -> np.ndarray | float:
        """mdp.IKPostureArmAction's gravity_offset: (14,) minus each arm joint's gravity torque at the target ik_q
        (ArmKinematics.gravity_torques, gravity in the pelvis frame) over its kp; 0 without it."""
        if not self.gravity_offset:
            return 0.0
        gravity_b = torch.tensor(R_pelvis.T @ self.m.opt.gravity, dtype=torch.float64)[None]
        tau = self._kin.gravity_torques(torch.tensor(self.ik_q, dtype=torch.float64).view(1, 2, 7), gravity_b)
        return -tau.numpy().reshape(14) / self.kp[self.arm_idx]

    def seed_q(self, arm: int) -> np.ndarray:
        """mdp.ArmTargetsCommand.seed_q for one arm: during an arm goal the joint angles of the seed table's row
        nearest the believed command (position, rotation at seed_rot_weight m per rad), with seed_solve solved from
        there for the command itself (once per command; the rest pose's seed the default angles), else the default
        angles."""
        idx = self.arm_idx[7 * arm:7 * arm + 7]
        if self.seed_table is None or not self.arm_mode:
            return self.default[idx]
        if self.seed_solve:
            if self._seeds_for is None or not np.array_equal(self._seeds_for[0], self.believed):
                self._seeds_for = (self.believed.copy(), self._solved_seeds())
            return self._seeds_for[1][arm]
        return self._nearest_row(arm)

    def _nearest_row(self, arm: int) -> np.ndarray:
        poses, joints = self.seed_table[arm]
        b = self.believed[arm]
        d = ((poses[:, :3] - b[:3]) ** 2).sum(1) + self.arm_ik["seed_rot_weight"] ** 2 * 8.0 * (1.0 - np.abs(poses[:, 3:] @ b[3:]))
        return joints[int(np.argmin(d))]

    def _solved_seeds(self) -> np.ndarray:
        """(2, 7): ArmTargetsCommand's seed solve for the believed command, inside the URDF limits narrowed by the
        tables' joint_range_margin."""
        if np.allclose(self.believed, self.rest_b, atol=1e-9):
            return self.default[self.arm_idx].reshape(2, 7)
        c = self.seed_solve
        lim = urdf_limits(self._urdf)
        names = ALL_NAMES[13:27]
        low = torch.tensor([lim[n][0] for n in names], dtype=torch.float64).view(2, 7) + c["joint_range_margin"]
        high = torch.tensor([lim[n][1] for n in names], dtype=torch.float64).view(2, 7) - c["joint_range_margin"]
        q0 = torch.tensor(np.stack([self._nearest_row(0), self._nearest_row(1)]), dtype=torch.float64)[None]
        goal = torch.tensor(self.believed, dtype=torch.float64)[None]
        q = self._kin.solve(goal, q0, low, high, iters=c["iters"], damping=c["damping"], rot_weight=c["rot_weight"])
        return q[0].numpy()

    def _posture_step(self, J, err, q, offset, idx) -> np.ndarray:
        """mdp.IKPostureArmAction._step for one arm: the weighted DLS task step and the null-space pull toward the
        posture (the default angles, or with posture_from_command seed_q, plus posture_scale x posture_gain x offset,
        inside the IK's bounds), joints that would pass limit_margin short of their bounds locked and solved out,
        ik_passes times; capped at max_ik_step."""
        c = self.arm_ik
        low, high = self.lower[idx] + c["limit_margin"], self.upper[idx] - c["limit_margin"]
        base = self.seed_q(0 if idx[0] == self.arm_idx[0] else 1) if self.seed_table is not None else self.default[idx]
        posture = np.clip(base + c["posture_scale"] * c["posture_gain"] * offset, low, high)
        w = np.r_[1.0, 1.0, 1.0, c["rot_weight"], c["rot_weight"], c["rot_weight"]]
        jac_w, err_w = J * w[:, None], err * w
        pull = c["null_gain"] * (posture - q)
        free = np.ones(len(q), bool)
        for _ in range(int(c["ik_passes"])):
            f = free.astype(float)
            jf = jac_w * f[None, :]
            pinv = jf.T @ np.linalg.inv(jf @ jf.T + c["ik_damping"] ** 2 * np.eye(6))
            step = (pinv @ err_w + (np.eye(len(q)) - pinv @ jf) @ (pull * f)) * f
            step = np.clip(step, -self.max_ik_step, self.max_ik_step)
            new = q + step
            hit = ((new < low) & (step < 0)) | ((new > high) & (step > 0))
            if not (hit & free).any():
                break
            free &= ~hit
        return step

    def physics_step(self):
        for _ in range(self.substeps):
            self.d.ctrl[self.act_id] = self.pd_torque(self.applied)
            mujoco.mj_step(self.m, self.d)  # pinned: the weld holds the pelvis (setting its qpos back too twisted the torso)
            if self.gyro_lowpass is not None:
                self._gyro = self.gyro_lowpass(self._raw_gyro(), self.m.opt.timestep)
            self._sample_dq()
        self.last_tau = self.pd_torque(self.applied)

    def _update_odometry(self):
        """During an arm goal, move the believed command by the pelvis's true motion since the last step."""
        p, q = self.root()
        if self.arm_mode and not self.pelvis_goals:
            pp, pq = self._prev_root
            dp, dq = qrot(qconj(pq), p - pp), qmul(qconj(pq), q)
            self.believed = np.stack([relative(dp, dq, b) for b in self.believed])
        self._prev_root = (p, q)

    # -- the command and the gait clock
    def observed_cmd(self) -> np.ndarray:
        return self.cmd_filtered.copy() if self.cmd_alpha is not None else np.asarray(self.vel_cmd, float)

    def _filter_command(self):
        if self.cmd_alpha is None:
            return
        if self.arm_mode and not self.pelvis_goals:
            self.cmd_filtered[:] = 0.0
        else:
            self.cmd_filtered += self.cmd_alpha * (np.asarray(self.vel_cmd, float) - self.cmd_filtered)

    def gait(self) -> list[np.ndarray]:
        """Each leg's phase sin (and cos), both legs at phase 0 under a command below zero_cmd_eps."""
        if np.linalg.norm(self.observed_cmd()) < self.zero_cmd_eps:
            phase = np.zeros(2)
        else:
            p = (self.steps * self.policy_dt % self.gait_period) / self.gait_period
            phase = np.array([p, (p + 0.5) % 1.0])
        out = [np.sin(2 * math.pi * phase)]
        return out + [np.cos(2 * math.pi * phase)] if self.gait_cos else out

    # -- observations
    def noisy_dq(self, scale: float) -> np.ndarray:
        """The measured joint speeds with the encoders' noise (joint_noise.py) times scale."""
        dq = self.measured_dq()
        std = JN.joint_vel_noise_std(dq, self.JVEL_FLOOR) * scale
        return dq + std * self.rng.standard_normal(np.shape(dq))

    def observations(self) -> dict[str, np.ndarray]:
        acc, gyro, grav = self.imu()
        n, r = self.NOISE, self.rng
        u = lambda x, s: x + r.uniform(-s, s, np.shape(x))  # noqa: E731
        believed = np.concatenate([xyzw(b) for b in self.believed])
        gait = self.gait()
        cur = self.wrist_poses_b()
        wrist_pose = np.concatenate([xyzw(np.r_[c[:3], qunique(c[3:])]) for c in cur])
        errs = np.concatenate([np.r_[b[:3] - c[:3], axis_angle(qmul(b[3:], qconj(c[3:])))]
                               for b, c in zip(self.believed, cur)])
        # each observation term by its env cfg name
        terms = {"base_lin_acc": u(acc, n["lin_acc"]), "base_ang_vel": u(gyro, n["ang_vel"]),
                 "projected_gravity": u(grav, n["grav"]), "velocity_commands": self.observed_cmd(),
                 "arm_goal": np.array([1.0 if self.arm_mode else 0.0]), "ee_targets": believed,
                 "joint_pos": u(self.measured_q() - self.default, n["jpos"]), "joint_vel": self.noisy_dq(n["jvel"]),
                 "gait_phase": gait[0]}
        if len(gait) > 1:
            terms["gait_phase_cos"] = gait[1]
        # a leg scale of 0 holds the default pose (LocoManip-ArmsOnly-v0's legs): its action reads 0, as in the env
        held = self.leg_scale == 0.0
        legs_action = (self.applied[self.leg_idx] - self.default[self.leg_idx]) / np.where(held, 1.0, self.leg_scale)
        legs = {**terms, "actions": np.where(held, 0.0, legs_action)}
        arms = {**terms, "actions": self.residual, "wrist_poses": u(wrist_pose, n["wrist"]),
                "wrist_errors": u(errs, n["wrist"])}
        # the walking task's layout: the shared terms, then each agent's own
        default = list(terms)
        legs_terms = self.legs_terms or default + ["actions"]
        arms_terms = self.arms_terms or default + ["actions", "wrist_poses", "wrist_errors"]
        return {"legs": np.concatenate([legs[k] for k in legs_terms]),
                "arms": np.concatenate([arms[k] for k in arms_terms])}

    def step(self, obs: dict):
        """One policy step from obs: act, simulate, then the next observations."""
        with torch.no_grad():
            a_l = (self.actors["legs"](torch.as_tensor(obs["legs"], dtype=torch.float32)[None])[0].numpy().astype(float)
                   if "legs" in self.actors else np.zeros(12))
            a_a = (self.actors["arms"](torch.as_tensor(obs["arms"], dtype=torch.float32)[None])[0].numpy().astype(float)
                   if "arms" in self.actors else np.zeros(self.n_arm_actions))
        self.process_actions(a_l, a_a)
        self.physics_step()
        self.steps += 1                    # the env's episode_length_buf counts the step before its observations
        self._update_odometry()
        self._filter_command()
        return self.observations()

    def foot_forces(self) -> np.ndarray:
        m, d = self.m, self.d
        f = np.zeros(2)
        c6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            for k in range(2):
                if con.geom1 in self.foot_geoms[k] or con.geom2 in self.foot_geoms[k]:
                    mujoco.mj_contactForce(m, d, i, c6)
                    f[k] += abs(c6[0])
        return f

    def fell(self) -> bool:
        p, q = self.root()
        zb = qrot(q, np.array([0.0, 0.0, 1.0]))
        return math.acos(max(-1.0, min(1.0, zb[2]))) > 1.0
