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
                      residual low-passed at arms.residual_cutoff_hz times arm_residual_scales, with the same bounds
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


def mjcf_path(assets: Path = C.DEFAULT_ASSETS) -> Path:
    return assets / "mujoco_assets" / "h1_2_magpie.xml"


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


class MarlMujoco:
    """One H1-2 in MuJoCo driven by an export's actors (callables obs[None] -> action[None]) and its settings."""

    NOISE = {"lin_acc": 0.5, "ang_vel": 0.2, "grav": 0.05, "jpos": 0.01, "jvel": 1.5, "wrist": 0.005}

    def __init__(self, actors: dict, settings: dict, physics: str = "robocasa", assets: Path = C.DEFAULT_ASSETS,
                 model: mujoco.MjModel | None = None):
        # model: the CL_Assets MJCF compiled with changes that leave the robot's joints untouched; default as it is
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(str(mjcf_path(assets)))
        self.d = mujoco.MjData(self.m)
        m = self.m
        arms_cfg = settings["arms"]
        self.policy_dt = float(settings["control_dt"])
        self.substeps = int(round(self.policy_dt / m.opt.timestep))
        self.residual_alpha = 1.0 - math.exp(-2.0 * math.pi * float(arms_cfg["residual_cutoff_hz"]) * self.policy_dt)
        self.leg_scale = np.asarray(settings.get("leg_action_scales", settings["legs"]["action_scale"]), float) * np.ones(12)
        self.arm_scale = np.asarray(settings.get("arm_residual_scales", arms_cfg["residual_scale"]), float) * np.ones(14)
        self.max_ik_step = float(settings.get("max_ik_step", arms_cfg["max_ik_step"]))
        tau = float(settings.get("command_filter_tau") or 0.0)
        self.cmd_alpha = 1.0 - math.exp(-self.policy_dt / tau) if tau > 0 else None
        self.gait_period, self.zero_cmd_eps = float(settings["gait_period"]), float(settings["zero_cmd_eps"])
        self.gait_cos = bool(settings.get("gait_phase_cos", False))
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

    # -- state
    def reset(self, rng: np.random.Generator):
        """On the ground near the default pose, nearly still, from a fresh episode (gait clock, filters, residual)."""
        m, d = self.m, self.d
        self.rng = rng
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = self.default + rng.uniform(-0.05, 0.05, len(ALL_NAMES))
        d.qvel[self.vadr] = rng.uniform(-0.1, 0.1, len(ALL_NAMES))
        roll, pitch = rng.uniform(-0.05, 0.05, 2)
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        d.qpos[3:7] = qmul(np.array([cr, sr, 0, 0]), np.array([cp, 0, sp, 0]))
        d.qpos[0:3] = (0.0, 0.0, 1.0)
        mujoco.mj_forward(m, d)
        sole = min(d.xpos[f][2] for f in self.feet) + C.SOLE_Z
        d.qpos[2] += -sole + rng.uniform(0.0, 0.02)
        d.qvel[0:6] = np.r_[rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.05, 0.05), rng.uniform(-0.1, 0.1, 3)]
        mujoco.mj_forward(m, d)
        self.applied = d.qpos[self.qadr].copy()
        self.residual = np.zeros(14)
        self.arm_mode = False
        self.believed = self.rest_b.copy()
        self.vel_cmd = np.zeros(3)
        self.cmd_filtered = np.zeros(3)
        self.steps = 0
        self._prev_root = (d.qpos[0:3].copy(), d.qpos[3:7].copy())
        if self.gyro_lowpass is not None:
            self.gyro_lowpass.reset()
            self._gyro = self.gyro_lowpass(self._raw_gyro(), m.opt.timestep)

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
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return acc, gyro, grav

    def measured_q(self) -> np.ndarray:
        """The joint angles the policy and the torque bound see (a subclass adds encoder offsets)."""
        return self.d.qpos[self.qadr].copy()

    def measured_dq(self) -> np.ndarray:
        """The joint velocities the policy sees (a subclass adds LowState's lag)."""
        return self.d.qvel[self.vadr].copy()

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
        a_legs, a_arms = np.clip(a_legs, -10, 10), np.clip(a_arms, -10, 10)
        idx = self.leg_idx
        t = np.clip(self.default[idx] + self.leg_scale * a_legs, self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        # arms
        self.residual = self.residual + self.residual_alpha * (a_arms - self.residual)
        m, d = self.m, self.d
        p, qr = self.root()
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, qr)
        Rr = Rr.reshape(3, 3)
        ik = d.qpos[self.qadr][self.arm_idx].copy()
        cur = self.wrist_poses_b()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for arm in range(2):
            # PhysX's articulation Jacobian is taken at the link's centre of mass (the task's pose error is at the
            # link origin); mj_jacBodyCom reproduces Isaac's IK step
            mujoco.mj_jacBodyCom(m, d, jacp, jacr, self.wrists[arm])
            J = np.vstack([Rr.T @ jacp[:, self.arm_cols[arm]], Rr.T @ jacr[:, self.arm_cols[arm]]])
            tgt = self.believed[arm]
            err = np.r_[tgt[:3] - cur[arm, :3], axis_angle(qmul(tgt[3:], qconj(cur[arm, 3:])))]
            step = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), err)
            sl = slice(7 * arm, 7 * arm + 7)
            ik[sl] = ik[sl] + np.clip(step, -self.max_ik_step, self.max_ik_step)
        idx = self.arm_idx
        t = np.clip(ik + self.arm_scale * self.residual, self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        self.applied[12] = 0.0                                    # torso held at default by its own actuator

    def physics_step(self):
        for _ in range(self.substeps):
            self.d.ctrl[self.act_id] = self.pd_torque(self.applied)
            mujoco.mj_step(self.m, self.d)
            if self.gyro_lowpass is not None:
                self._gyro = self.gyro_lowpass(self._raw_gyro(), self.m.opt.timestep)
        self.last_tau = self.pd_torque(self.applied)

    def _update_odometry(self):
        """During an arm goal, move the believed command by the pelvis's true motion since the last step."""
        p, q = self.root()
        if self.arm_mode:
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
        if self.arm_mode:
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
    def observations(self) -> dict[str, np.ndarray]:
        acc, gyro, grav = self.imu()
        n, r = self.NOISE, self.rng
        u = lambda x, s: x + r.uniform(-s, s, np.shape(x))  # noqa: E731
        believed = np.concatenate([xyzw(b) for b in self.believed])
        common = [u(acc, n["lin_acc"]), u(gyro, n["ang_vel"]), u(grav, n["grav"]), self.observed_cmd(),
                  np.array([1.0 if self.arm_mode else 0.0]), believed, u(self.measured_q() - self.default, n["jpos"]),
                  u(self.measured_dq(), n["jvel"])] + self.gait()
        legs_applied = (self.applied[self.leg_idx] - self.default[self.leg_idx]) / self.leg_scale
        cur = self.wrist_poses_b()
        wrist_pose = np.concatenate([xyzw(np.r_[c[:3], qunique(c[3:])]) for c in cur])
        errs = np.concatenate([np.r_[b[:3] - c[:3], axis_angle(qmul(b[3:], qconj(c[3:])))]
                               for b, c in zip(self.believed, cur)])
        return {"legs": np.concatenate(common + [legs_applied]),
                "arms": np.concatenate(common + [self.residual, u(wrist_pose, n["wrist"]), u(errs, n["wrist"])])}

    def step(self, obs: dict):
        """One policy step from obs: act, simulate, then the next observations."""
        with torch.no_grad():
            a_l = self.actors["legs"](torch.as_tensor(obs["legs"], dtype=torch.float32)[None])[0].numpy().astype(float)
            a_a = self.actors["arms"](torch.as_tensor(obs["arms"], dtype=torch.float32)[None])[0].numpy().astype(float)
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
