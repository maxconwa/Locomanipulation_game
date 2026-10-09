"""The two-agent policy loop of LocoManip-Marl-Direct-v0 ported to MuJoCo, for sim-to-sim evaluation without ROS.

Mirrors, step for step, what the Isaac env computes (locomanip_marl_env.py, mdp/actions.py, mdp/observations.py,
odometry.py at marl-direct 72171c7):

    policy step (50 Hz)   actions -> joint targets; 10 MuJoCo steps of 2 ms with PD torques recomputed each step;
                          then the odometry history and the learned pelvis estimator move the arm command; then the
                          actors' observations (with the training noise) for the next step
    legs action           target = default + 0.25 a, inside [hard lower + 1 mm, hard upper - 1 mm] and then inside the
                          targets whose PD torque at the measured state stays within 85% of the effort limit
    arms action           one damped-least-squares IK step per arm toward the believed command (6-D pose error
                          [p_t - p, axis-angle(q_t q^-1)] in the pelvis frame; the wrist Jacobian in the pelvis frame
                          at the link's centre of mass, as PhysX computes it; lambda 0.05; capped at 0.1 rad),
                          plus the residual low-passed at 3 Hz
                          times 0.2, with the same two bounds
    PD                    tau = kp (target - q) - kd qd, clipped to the effort limit: the Isaac actuator gains
    observations          legs 95 / arms 123 values, term by term in the cfg's order (see LEGS_TERMS / ARMS_TERMS)

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
DEFAULT = {n: 0.0 for n in ALL_NAMES} | {f"{s}_hip_pitch_joint": -0.3 for s in ("left", "right")} | \
    {f"{s}_knee_joint": 0.5 for s in ("left", "right")} | {f"{s}_ankle_pitch_joint": -0.2 for s in ("left", "right")}


def _gains(name: str) -> tuple[float, float, float]:
    """(kp, kd, effort limit) of the Isaac actuator for a joint (assets/h1_2.py _ACTUATORS)."""
    if "_hip_" in name:
        return 200.0, 2.5, 200.0
    if "_knee_" in name:
        return 300.0, 4.0, 300.0
    if "ankle_pitch" in name:
        return 40.0, 2.0, 60.0
    if "ankle_roll" in name:
        return 40.0, 2.0, 40.0
    if name == "torso_joint":
        return 300.0, 3.0, 200.0
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return 120.0, 2.0, 40.0
    if "shoulder_yaw" in name:
        return 120.0, 2.0, 18.0
    if "elbow" in name:
        return 80.0, 1.0, 18.0
    if "wrist" in name:
        return 40.0, 1.0, 19.0
    raise KeyError(name)


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


def quat_from_rotvec(r):
    a = np.linalg.norm(r)
    if a < 1e-12:
        return np.array([1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]]) / np.linalg.norm([1.0, *(0.5 * r)])
    return np.r_[math.cos(a / 2), math.sin(a / 2) * r / a]


def relative(pos_a, quat_a, pose):
    """T_a^-1 o pose for pose (7,) = (x y z qw qx qy qz); qw >= 0."""
    qi = qconj(quat_a)
    return np.r_[qrot(qi, pose[:3] - pos_a), qunique(qmul(qi, pose[3:]))]


def apply(pos_a, quat_a, pose):
    return np.r_[qrot(quat_a, pose[:3]) + pos_a, qmul(quat_a, pose[3:])]


def xyzw(pose):
    return np.r_[pose[:3], pose[4:7], pose[3]]


# ---------------------------------------------------------------- the sim

class MarlMujoco:
    """One H1-2 in MuJoCo driven by a checkpoint's actors (legs and arms, or one whole-body actor) and its pelvis
    estimator."""

    POLICY_DT = C.POLICY_DT
    RESIDUAL_ALPHA = 1.0 - math.exp(-2.0 * math.pi * 3.0 * C.POLICY_DT)
    NOISE_POLICY = {"lin_acc": 0.5, "ang_vel": 0.2, "grav": 0.05, "jpos": 0.01, "jvel": 1.5, "wrist": 0.005}
    NOISE_ODOM = {"lin_acc": 0.05, "ang_vel": 0.02, "grav": 0.005, "jpos": 0.001, "jvel": 0.05, "tau": 2.0}

    def __init__(self, actors: dict, estimator: torch.nn.Module | None, physics: str = "robocasa",
                 assets: Path = C.DEFAULT_ASSETS, rest_pose_b: np.ndarray | None = None):
        self.m = mujoco.MjModel.from_xml_path(str(assets / "mujoco_assets" / "h1_2_magpie.xml"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.substeps = int(round(self.POLICY_DT / m.opt.timestep))
        jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ALL_NAMES}
        self.qadr = np.array([m.jnt_qposadr[jid[n]] for n in ALL_NAMES])
        self.vadr = np.array([m.jnt_dofadr[jid[n]] for n in ALL_NAMES])
        self.leg_idx = np.arange(12)
        self.arm_idx = np.arange(13, 27)
        self.default = np.array([DEFAULT[n] for n in ALL_NAMES])
        g = np.array([_gains(n) for n in ALL_NAMES])
        self.kp, self.kd, self.effort = g[:, 0], g[:, 1], g[:, 2]
        lim = urdf_limits(assets / "ros_assets" / "h1_2_magpie.urdf")
        self.lower = np.array([lim[n][0] for n in ALL_NAMES]) + 0.001
        self.upper = np.array([lim[n][1] for n in ALL_NAMES]) - 0.001
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
        self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        sens = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]  # noqa: E731
        self.s_acc, self.s_gyro = sens("imu_acc"), sens("imu_gyro")
        self.foot_geoms = [set(np.nonzero(m.geom_bodyid == f)[0].tolist()) for f in self.feet]
        self.arm_cols = [self.vadr[13:20], self.vadr[20:27]]
        self.actors, self.estimator = actors, estimator
        self.rest_b = np.asarray(rest_pose_b, float) if rest_pose_b is not None else None
        self.rng = np.random.default_rng(0)
        self.total_mass = float(m.body_subtreemass[self.pelvis])

    # -- state
    def reset(self, rng: np.random.Generator):
        m, d = self.m, self.d
        self.rng = rng
        mujoco.mj_resetData(m, d)
        q = self.default + rng.uniform(-0.05, 0.05, len(ALL_NAMES))
        d.qpos[self.qadr] = q
        d.qvel[self.vadr] = rng.uniform(-0.1, 0.1, len(ALL_NAMES))
        roll, pitch = rng.uniform(-0.05, 0.05, 2)
        cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        d.qpos[3:7] = qmul(np.array([cr, sr, 0, 0]), np.array([cp, 0, sp, 0]))
        d.qpos[0:3] = (0.0, 0.0, C.STANDING_PELVIS_HEIGHT)
        mujoco.mj_forward(m, d)
        sole = min(d.xpos[f][2] for f in self.feet) + C.SOLE_Z
        d.qpos[2] += -sole + rng.uniform(0.0, 0.02)
        d.qvel[0:6] = np.r_[rng.uniform(-0.1, 0.1, 2), rng.uniform(-0.05, 0.05), rng.uniform(-0.1, 0.1, 3)]
        mujoco.mj_forward(m, d)
        self.applied = d.qpos[self.qadr].copy()
        self.residual = np.zeros(14)
        self.raw_actions = np.zeros(26)
        self.arm_mode = False
        self.believed = self.rest_b.copy()
        self.anchor = None
        self.use_estimate = True
        self.vel_cmd = np.zeros(3)
        self._prev_root = (d.qpos[0:3].copy(), d.qpos[3:7].copy())
        self._hist = None
        self._update_odometry(first=True)

    # -- kinematics in the pelvis frame
    def root(self):
        return self.d.qpos[0:3].copy(), self.d.qpos[3:7].copy()

    def wrist_poses_b(self) -> np.ndarray:
        p, q = self.root()
        return np.stack([relative(p, q, np.r_[self.d.xpos[w], self.d.xquat[w]]) for w in self.wrists])

    def imu(self):
        d = self.d
        acc = d.sensordata[self.s_acc:self.s_acc + 3].copy()
        gyro = d.sensordata[self.s_gyro:self.s_gyro + 3].copy()
        R = d.site_xmat[self.imu_site].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return acc, gyro, grav

    def pd_torque(self, target):
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        return np.clip(self.kp * (target - q) - self.kd * qd, -self.effort, self.effort)

    def _torque_bounds(self, idx):
        q, qd = self.d.qpos[self.qadr][idx], self.d.qvel[self.vadr][idx]
        h = 0.85 * self.effort[idx]
        return q + (self.kd[idx] * qd - h) / self.kp[idx], q + (self.kd[idx] * qd + h) / self.kp[idx]

    # -- actions
    def process_actions(self, a_legs: np.ndarray, a_arms: np.ndarray):
        self.raw_actions = np.r_[np.clip(a_legs, -10, 10), np.clip(a_arms, -10, 10)]
        a_legs, a_arms = self.raw_actions[:12], self.raw_actions[12:]
        idx = self.leg_idx
        t = np.clip(self.default[idx] + 0.25 * a_legs, self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        # arms
        self.residual = self.residual + self.RESIDUAL_ALPHA * (a_arms - self.residual)
        m, d = self.m, self.d
        p, qr = self.root()
        Rr = np.zeros(9)
        mujoco.mju_quat2Mat(Rr, qr)
        Rr = Rr.reshape(3, 3)
        q_all = d.qpos[self.qadr]
        ik = q_all[self.arm_idx].copy()
        cur = self.wrist_poses_b()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for arm in range(2):
            # PhysX's articulation Jacobian is taken at the link's centre of mass (the task's pose error is at the
            # link origin); mj_jacBodyCom reproduces Isaac's IK step to 2e-5 rad (parity_check.py)
            mujoco.mj_jacBodyCom(m, d, jacp, jacr, self.wrists[arm])
            J = np.vstack([Rr.T @ jacp[:, self.arm_cols[arm]], Rr.T @ jacr[:, self.arm_cols[arm]]])
            tgt = self.believed[arm]
            err = np.r_[tgt[:3] - cur[arm, :3], axis_angle(qmul(tgt[3:], qconj(cur[arm, 3:])))]
            step = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), err)
            sl = slice(7 * arm, 7 * arm + 7)
            ik[sl] = ik[sl] + np.clip(step, -0.1, 0.1)
        idx = self.arm_idx
        t = np.clip(ik + 0.2 * self.residual, self.lower[idx], self.upper[idx])
        lo, hi = self._torque_bounds(idx)
        self.applied[idx] = np.clip(t, lo, hi)
        self.applied[12] = 0.0                                    # torso held at default by its own actuator

    def physics_step(self):
        for _ in range(self.substeps):
            tau = self.pd_torque(self.applied)
            self.d.ctrl[self.act_id] = tau
            mujoco.mj_step(self.m, self.d)
        self.last_tau = self.pd_torque(self.applied)

    # -- odometry: the learned pelvis estimator moves the believed command
    def _odom_frame(self):
        acc, gyro, grav = self.imu()
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        tau = self.pd_torque(self.applied)[self.leg_idx]
        n, r = self.NOISE_ODOM, self.rng
        u = lambda x, s: x + r.uniform(-s, s, np.shape(x))  # noqa: E731
        return [u(acc, n["lin_acc"]), u(gyro, n["ang_vel"]), u(grav, n["grav"]), u(q - self.default, n["jpos"]),
                u(qd, n["jvel"]), u(tau, n["tau"])]

    def _update_odometry(self, first=False):
        frame = self._odom_frame()
        if first or self._hist is None:
            self._hist = [[f.copy() for _ in range(4)] for f in frame]
        else:
            for h, f in zip(self._hist, frame):
                h.pop(0)
                h.append(f)
        p, q = self.root()
        if not first and self.arm_mode:
            pp, pq = self._prev_root
            true_dp = qrot(qconj(pq), p - pp)
            true_dq = qmul(qconj(pq), q)
            if self.use_estimate and self.estimator is not None:
                x = np.concatenate([np.concatenate(h) for h in self._hist] + [self.raw_actions])
                with torch.no_grad():
                    v = self.estimator(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy().astype(float)
                dp, dq = v[:3] * self.POLICY_DT, quat_from_rotvec(v[3:] * self.POLICY_DT)
            else:
                dp, dq = true_dp, true_dq
            self.believed = np.stack([relative(dp, dq, b) for b in self.believed])
        self._prev_root = (p, q)

    # -- observations
    def observations(self) -> dict[str, np.ndarray]:
        acc, gyro, grav = self.imu()
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        n, r = self.NOISE_POLICY, self.rng
        u = lambda x, s: x + r.uniform(-s, s, np.shape(x))  # noqa: E731
        believed = np.concatenate([xyzw(b) for b in self.believed])
        common = [u(acc, n["lin_acc"]), u(gyro, n["ang_vel"]), u(grav, n["grav"]), self.vel_cmd,
                  np.array([1.0 if self.arm_mode else 0.0]), believed, u(q - self.default, n["jpos"]),
                  u(qd, n["jvel"]), np.zeros(2)]               # gait phase: sin(0) under a zero velocity command
        legs_applied = (self.applied[self.leg_idx] - self.default[self.leg_idx]) / 0.25
        cur = self.wrist_poses_b()
        wrist_pose = np.concatenate([xyzw(np.r_[c[:3], qunique(c[3:])]) for c in cur])
        errs = np.concatenate([np.r_[b[:3] - c[:3], axis_angle(qmul(b[3:], qconj(c[3:])))]
                               for b, c in zip(self.believed, cur)])
        if "whole" in self.actors:
            # LocoManip-WholeBody-Direct-v0: the shared terms once, the legs' and the arms' applied actions, then the
            # arms' wrist terms (135)
            return {"whole": np.concatenate(common + [legs_applied, self.residual, u(wrist_pose, n["wrist"]),
                                                      u(errs, n["wrist"])])}
        legs = np.concatenate(common + [legs_applied])
        arms = np.concatenate(common + [self.residual, u(wrist_pose, n["wrist"]), u(errs, n["wrist"])])
        return {"legs": legs, "arms": arms}

    def step(self, obs: dict, legs_blind: bool = False, arms_ik: bool = False):
        if "whole" in obs:
            if legs_blind:
                raise ValueError("legs_blind needs separate leg and arm actors")
            with torch.no_grad():
                a = self.actors["whole"](torch.as_tensor(obs["whole"], dtype=torch.float32)[None])[0].numpy()
            a_l, a_a = a[:12].astype(float), a[12:].astype(float)
        else:
            lo = obs["legs"].copy()
            if legs_blind:
                lo[12] = 0.0
                lo[13:27] = np.concatenate([xyzw(b) for b in self.rest_b])
            with torch.no_grad():
                a_l = self.actors["legs"](torch.as_tensor(lo, dtype=torch.float32)[None])[0].numpy().astype(float)
                a_a = self.actors["arms"](torch.as_tensor(obs["arms"], dtype=torch.float32)[None])[0].numpy().astype(float)
        if arms_ik:
            a_a[:] = 0.0
        self.process_actions(a_l, a_a)
        self.physics_step()
        self._update_odometry()
        return self.observations()

    # -- goals
    def set_goal(self, poses_s: np.ndarray, use_estimate: bool = True):
        """Arm goal from standing-frame poses (2, 7): placed at the pelvis's x, y, yaw and standing height now."""
        p, q = self.root()
        w, x, y, z = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        origin = np.array([p[0], p[1], C.STANDING_PELVIS_HEIGHT])           # flat ground at z = 0
        self.anchor = np.stack([apply(origin, q_yaw, ps) for ps in poses_s])
        self.believed = np.stack([relative(p, q, a) for a in self.anchor])
        self.arm_mode = True
        self.use_estimate = use_estimate

    def true_errors(self):
        out_p, out_r = [], []
        for arm, w in enumerate(self.wrists):
            out_p.append(np.linalg.norm(self.d.xpos[w] - self.anchor[arm, :3]))
            dq = qmul(self.anchor[arm, 3:], qconj(self.d.xquat[w]))
            out_r.append(np.linalg.norm(axis_angle(dq)))
        return np.array(out_p), np.array(out_r)

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

    def log_step(self) -> dict:
        m, d = self.m, self.d
        pe, re_ = self.true_errors()
        p, q = self.root()
        mujoco.mj_subtreeVel(m, d)
        vel6 = np.zeros(6)
        feet_vel = []
        for f in self.feet:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, f, vel6, 0)
            feet_vel.append(vel6[3:].copy())
        ang_b = d.qvel[3:6].copy()                       # free joint angular velocity is in the body frame
        tau = np.abs(self.pd_torque(self.applied)) / self.effort
        groups = [np.r_[0:3, 6:9], np.r_[3, 9], np.r_[4:6, 10:12], self.arm_idx]
        believed_w = np.stack([apply(p, q, b) for b in self.believed])
        return {
            "pos_err": pe, "rot_err": re_, "root_pos": p, "root_quat": q, "root_ang_vel_b": ang_b, "ground": 0.0,
            "com": d.subtree_com[self.pelvis].copy(), "com_vel": d.subtree_linvel[self.pelvis].copy(),
            "feet_pos": np.stack([d.xpos[f] for f in self.feet]), "feet_quat": np.stack([d.xquat[f] for f in self.feet]),
            "feet_vel": np.stack(feet_vel), "feet_force": self.foot_forces(),
            "wrist_pos": np.stack([d.xpos[w] for w in self.wrists]),
            "torque_ratio": np.array([tau[g].max() for g in groups]),
            "cmd_drift": float(np.mean(np.linalg.norm(believed_w[:, :3] - self.anchor[:, :3], axis=1))),
        }

    def fell(self) -> bool:
        p, q = self.root()
        zb = qrot(q, np.array([0.0, 0.0, 1.0]))
        return math.acos(max(-1.0, min(1.0, zb[2]))) > 1.0


def load_estimator(path: str | None) -> torch.nn.Module | None:
    """The pelvis estimator saved beside a checkpoint: input norm + MLP (odometry.PelvisMotionEstimator)."""
    if not path:
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    from .policy import _mlp

    class Estimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", state["norm.mean"].float())
            self.register_buffer("var", state["norm.var"].float())
            self.net = _mlp(state, "net")

        def forward(self, x):
            return self.net((x - self.mean) / torch.sqrt(self.var + 1e-8))

    return Estimator().eval()
