"""The hardware-perturbation stress test in MuJoCo: an exported policy (GOLEM lmg_deploy's legs.pt / arms.pt,
TorchScript, LSTM state carried) on CL_Assets' H1-2 MJCF, the model GOLEM's RoboCasa sim uses, at 500 Hz.

Builds on evals.mujoco.sim.MarlMujoco (the env's action bounds, PD, IK, command filter, gait clock and
observations, all from the export's settings) and adds the perturbations (CONDITIONS):
  plant          robocasa: the MJCF's joint dynamics (damping 10, frictionloss 0.2, armature 0.1, as RoboCasa);
                 isaac: the training plant's (none, armature 0.01)
  strength       the motors' torque x s (kp, kd and the effort limit together); gain: kp and kd x g
  dq_lag         the observed hip pitch and knee velocity (LowState dq) this many ms late
  gyro_k         the head's 20-21 Hz pitch resonance added to the gyro before its low-pass, at gyro_k x the measured
                 amplitude
  delay_ms       the joint targets this late; torso_kg: added to the torso; enc: encoder offsets on the legs (rad)
  hang_s         held on the gantry (feet 8 cm up) this long, then lowered onto the feet over LOWER_S
  prepose        the engage from lmg_deploy's pre-pose (pp_* conditions; GOLEM's marl_deploy_node._prepose and
                 controller.reset, the sequence of sagged_engage.py of 2026-11-20): on the gantry, lowered onto the
                 feet with a rope carrying `rope` of the weight, the pre-pose easing to the start pose, then the
                 policy engaged with the deploy's reset; the rope kept or released (`release`) at the engage. Starts:
                   hung     engaged on the gantry, feet 8 cm up; lowered PP_HUNG_HOLD_S later
                   settled  the default pose on the feet
                   bag      the robot's measured pre-pose at its 2026-11-20 11:57 engage (data/bag_engage_q.npy)
                   bag1449  the one at its 14:49 engage (data/bag_engage_q_1449.npy: unsettled, 0.16 rad off)
                   random   a draw from the training's pre-pose range (PREPOSE_JOINT_RANGES; per seed)
                 Run for PP_AFTER_S with a zero command; scored on the first 0.5 s, 0.5-1 s and the stance after (pp
                 table). The sag comes from the pre-pose commanding the start pose (this MJCF's ankles are stiffer
                 than the robot's, which sags under the default pose's command): the jumps count from the measured
                 pose, the robot's from the default. An export's engage blend (engage_blend_s) starts from that
                 command, as lmg_deploy's starts from the pre-pose's last
  imu_pitch, vib_k   for an export on the torso IMU (export imu.source lowstate): the quaternion's pitch bias as the
                 policy reads it ("lo" / "hi": the trained range's ends; default its centre): for imu.orientation
                 calibrated the error lmg_deploy's calibration leaves, for raw the offset itself; and its gyro's
                 vibration tied to leg motion, x TORSO_GYRO_PER_DQ
Per job (one condition, profile, plant and seed) one robot for --seconds; jobs run in parallel processes. --video COND
renders one job of COND (seed 0) to an mp4 instead.

    python -m evals.deploy.stress_mujoco --export <policies/name> --label <name> [--seeds 8] [--plants robocasa,isaac]
        [--conditions ...] [--profiles zero,small,small60] [--workers 24]
    python -m evals.deploy.stress_mujoco --export <...> --label <name> --conditions pp   (the pre-pose grid)
    python -m evals.deploy.stress_mujoco --export <...> --label <name> --video pp_bag_release

Writes logs/hw_stress_mujoco/<label>/summary.json and summary.md. Any python with mujoco, torch, numpy and pyyaml.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evals.mujoco.common import load_deploy_cfg  # noqa: E402
from evals.mujoco.policy import Recurrent  # noqa: E402, F401  (bag_replay uses H.Recurrent)
from evals.mujoco.sim import (  # noqa: E402
    ALL_NAMES, MarlMujoco, imu_bias_ranges, mjcf_path, qrot)

# the conditions; hw: the measured hardware effects together
_HW = {"strength": 0.85, "dq_lag": 12.0, "gyro_k": 1.0, "delay_ms": 20.0, "enc": 0.02}
CONDITIONS = {
    "nominal": {},
    "strength085": {"strength": 0.85},
    "strength075": {"strength": 0.75},
    "gain080": {"gain": 0.8},
    "gain120": {"gain": 1.2},
    "dqlag12": {"dq_lag": 12.0},
    "dqlag24": {"dq_lag": 24.0},
    "gyro1": {"gyro_k": 1.0},
    "gyro2": {"gyro_k": 2.0},
    "delay20": {"delay_ms": 20.0},
    "delay30": {"delay_ms": 30.0},
    "delay40": {"delay_ms": 40.0},
    "torso5kg": {"torso_kg": 5.0},
    "enc002": {"enc": 0.02},
    "hw": dict(_HW),
    "hw_no_gyro": {k: v for k, v in _HW.items() if k != "gyro_k"},
    "hw_no_dqlag": {k: v for k, v in _HW.items() if k != "dq_lag"},
    "hw_no_strength": {k: v for k, v in _HW.items() if k != "strength"},
    "hw_no_delay": {k: v for k, v in _HW.items() if k != "delay_ms"},
    "hw_no_enc": {k: v for k, v in _HW.items() if k != "enc"},
    # engaged hanging from the gantry (feet HANG_LIFT up) for hang_s, then lowered onto the feet over LOWER_S and let go
    "hang": {"hang_s": 4.0},
    "hang_hw": dict(_HW, hang_s=4.0),
    # the actors' LSTM state zeroed every 10 s (both, the legs' only, the arms' only): does the policy go wrong when
    # its state has run long?
    "lstm_reset": {"lstm_reset_s": 10.0},
    "lstm_reset_legs": {"lstm_reset_s": 10.0, "lstm_reset_which": "legs"},
    "lstm_reset_arms": {"lstm_reset_s": 10.0, "lstm_reset_which": "arms"},
}
# the pre-pose grid: each start with the rope kept and released, with the hardware effects and the torso IMU's
# vibration at x4 (sagged_engage.py's --head-res 4 matched the bag's input); the bag's pose and the settled one also at
# the ends of the quaternion bias range the export trained on (pitch_lo, pitch_hi)
PP_STARTS = ("hung", "settled", "bag", "bag1449", "random")
for _start in PP_STARTS:
    for _rope in ("keep", "release"):
        CONDITIONS[f"pp_{_start}_{_rope}"] = dict(_HW, prepose=_start, release=_rope == "release", vib_k=4.0)
for _start in ("settled", "bag"):
    for _end in ("lo", "hi"):
        CONDITIONS[f"pp_{_start}_release_pitch_{_end}"] = dict(_HW, prepose=_start, release=True, vib_k=4.0,
                                                               imu_pitch=_end)
# delay_ms: the robot's /lowstate to command 1.1 ms (p99 2.1) plus the safety relay's up to 2 ms (2026-11-23 bags; the
# motors' own latency unmeasured): 4 ms nominal, 20 in the hardware bundle (_HW). 10 until 2026-11-23
NOMINAL = {"strength": 1.0, "gain": 1.0, "dq_lag": 0.0, "gyro_k": 0.0, "delay_ms": 4.0, "torso_kg": 0.0, "enc": 0.0,
           "hang_s": 0.0, "lstm_reset_s": 0.0, "lstm_reset_which": "both", "prepose": "", "rope": 0.4,
           "release": False, "imu_pitch": None, "vib_k": 0.0}
HANG_LIFT, LOWER_S = 0.08, 1.0
GYRO_RMS_PER_DQ_RMS = 3.4
LAG_IDX = [ALL_NAMES.index(f"{s}_{j}_joint") for s in ("left", "right") for j in ("hip_pitch", "knee")]
ANKLE_IDX = [ALL_NAMES.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
KNEE_IDX = [ALL_NAMES.index(n) for n in ("left_knee_joint", "right_knee_joint")]
FALL_TILT, FALL_HEIGHT = 0.8, 0.55
FLOOR_MARGIN, FLOOR_HOLD_S = 0.1, 0.5
STAND_WINDOW = (2.0, 10.0)
# the pre-pose engage (sagged_engage.py's sequence): s on the gantry, s lowering, s easing to the start pose, the
# engage's time, s run after it; the hung start stays on the gantry this long after its engage
PP_HANG_S, PP_LOWER_S, PP_EASE_S, PP_ENGAGE_S, PP_AFTER_S, PP_HUNG_HOLD_S = 1.5, 1.0, 2.0, 8.0, 8.0, 2.0
PP_TETHER_K, PP_TETHER_D = 1500.0, 300.0  # N/m, N s/m: the rope's horizontal pull back to where it took over
# 27 joint angles, policy order (rad), at the robot's engages of 2026-11-20 (lmg_legsenergy200k_s43_2026-11-20_1157 and
# _1449, the last 20 ms before each)
BAG_ENGAGE_Q = {"bag": REPO / "evals" / "deploy" / "data" / "bag_engage_q.npy",
                "bag1449": REPO / "evals" / "deploy" / "data" / "bag_engage_q_1449.npy"}
# the training's pre-pose joint offsets (locomanip_marl_env_cfg.PREPOSE_JOINT_RANGES; rad from the default pose)
PREPOSE_JOINT_RANGES = {"ankle_pitch": (-0.3, 0.15), "ankle_roll": (-0.1, 0.1), "knee": (-0.1, 0.2), "hip": (-0.1, 0.1),
                        "torso": (-0.05, 0.05), "shoulder": (-0.15, 0.15), "elbow": (-0.15, 0.15), "wrist": (-0.15, 0.15)}
# the torso IMU's gyro vibration: rad/s rms per rad/s rms of leg joint speed at vib_k 1 (on the robot 0.4-0.8 rad/s
# while the legs thrashed at a few rad/s); and the lowstate model's noise and lag (lowstate_imu.LowstateImuCfg). The
# noise is the robot's of 2026-11-23 (the quaternion's per-step noise 0.001 deg standing, 0.025-0.046 walking; the
# gyro's 0.009-0.011 rad/s standing): until then 0.1 deg and 0.02 rad/s
TORSO_GYRO_PER_DQ = 0.2
TORSO_GYRO_AXES = np.array([0.6, 1.0, 0.4])
LOWSTATE_QUAT_NOISE_DEG, LOWSTATE_LAG_MS, LOWSTATE_GYRO_NOISE = 0.05, 10.0, 0.01
# the pp table's pass marks
PP_MAX_MEAN_JUMP, PP_MAX_TORQUE_SHARE, PP_MAX_KNEE_OFFSET, PP_MAX_AT_LIMIT = 0.03, 0.85, 0.15, 0.05
# the stance's pitch from 2 s on (deg, the pelvis against gravity): the 2026-11-23 bags of LegsFastEnv60k_s42 stood
# at -0.5 deg after one engage and leaned back to -3.4 and -5.8 after two others (the ankles pulling, the weight
# behind them), and those two tipped back to -10 deg when they stopped walking
PP_MAX_STANCE_PITCH_DEG = 2.0


class StressMujoco(MarlMujoco):
    """MarlMujoco with the perturbations of one condition (CONDITIONS, f): the motors' strength and gain, the encoders'
    offsets, LowState's joint-velocity lag, the joint targets' delay, the head's gyro resonance, a heavier torso, and
    the gantry (held, then lowered onto the feet)."""

    def __init__(self, actors, f: dict, deploy: dict, physics: str, rng: np.random.Generator):
        model = mujoco.MjModel.from_xml_path(str(mjcf_path()))
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        model.body_mass[torso] += f["torso_kg"]
        super().__init__(actors, deploy, physics=physics, model=model)
        m = self.m
        self.head = (deploy.get("imu") or {}).get("frame", "imu_link") != "imu_link"
        self.f = f
        self.enc = np.zeros(len(ALL_NAMES))
        self.enc[:12] = rng.uniform(-f["enc"], f["enc"], 12)
        self.delay = int(round(f["delay_ms"] / 1000 / m.opt.timestep))
        self.lag = int(round(f["dq_lag"] / 1000 / m.opt.timestep))
        self.res_freq = 20.0 + rng.uniform()
        self.res_phase = 2 * math.pi * rng.uniform()
        self.torso_body = torso
        # the torso IMU as /lowstate reports it: the quaternion lagged and biased, the gyro biased, noisy and shaken
        imu = deploy.get("imu") or {}
        self.lowstate = imu.get("source") == "lowstate"
        if self.lowstate:
            ranges = imu_bias_ranges(imu)
            bias = {k: 0.5 * (lo + hi) for k, (lo, hi) in ranges.items()}
            if f["imu_pitch"] is not None:
                bias["pitch"] = ranges["pitch"][0 if f["imu_pitch"] == "lo" else 1]
            self.imu_bias_deg = bias
            r, pt, y = (math.radians(bias[k]) for k in ("roll", "pitch", "yaw"))
            rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
            ry = np.array([[math.cos(pt), 0, math.sin(pt)], [0, 1, 0], [-math.sin(pt), 0, math.cos(pt)]])
            rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
            self.R_bias = rz @ ry @ rx
            self.gyro_bias = rng.uniform(-1, 1, 3) * np.array([0.005, 0.005, 0.02])
            self.NOISE = {**self.NOISE, "ang_vel": 0.0, "grav": 0.0}  # the model's own noise instead
            self.R_hist = deque(maxlen=int(round(LOWSTATE_LAG_MS / 1000 / m.opt.timestep)) + 1)
        self.weight = float(m.body_mass.sum() * -m.opt.gravity[2])
        self.rope_share, self.rope_anchor = 0.0, None
        self.pp_held = None  # the pre-pose's gantry: callable t -> pelvis height or None

    def held_z(self):
        """The gantry's pelvis height now, or None once let go."""
        if self.pp_held is not None:
            return self.pp_held(self.sub * self.m.opt.timestep)
        if self.f["hang_s"] <= 0:
            return None
        t = self.sub * self.m.opt.timestep
        if t < self.f["hang_s"]:
            return self.hold_pose[2] + HANG_LIFT
        if t < self.f["hang_s"] + LOWER_S:
            return self.hold_pose[2] + HANG_LIFT * (1 - (t - self.f["hang_s"]) / LOWER_S)
        return None

    def reset(self, rng):
        self.targets = deque(maxlen=self.delay + 1)
        self.qd_hist = deque(maxlen=self.lag + 1)
        self.leg_rms = 0.0
        self.sub = 0
        super().reset(rng)
        self.hold_pose = self.d.qpos[0:7].copy()
        if self.f["hang_s"] > 0:
            self.d.qpos[2] += HANG_LIFT
            mujoco.mj_forward(self.m, self.d)
        self.qd_hist.append(self.d.qvel[self.vadr].copy())
        if self.lowstate:
            self.R_hist.clear()
            self.R_hist.append(self.d.site_xmat[self.imu_site].reshape(3, 3).copy())

    def imu(self):
        if not self.lowstate:
            return super().imu()
        R = self.R_hist[0] @ self.R_bias  # lag substeps old once the deque is full
        n = np.radians(LOWSTATE_QUAT_NOISE_DEG) * self.rng.standard_normal(2)
        cr, sr, cp, sp = math.cos(n[0]), math.sin(n[0]), math.cos(n[1]), math.sin(n[1])
        R = R @ (np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        gyro = self._gyro.copy() if self.gyro_lowpass is not None else self._raw_gyro()
        return np.zeros(3), gyro, R.T @ np.array([0.0, 0.0, -1.0])

    def rope(self):
        """The pre-pose's rope on the pelvis: rope_share of the weight up and a soft horizontal tether."""
        d = self.d
        d.xfrc_applied[self.pelvis] = 0.0
        if self.rope_share <= 0.0:
            self.rope_anchor = None
            return
        if self.rope_anchor is None:
            self.rope_anchor = d.qpos[0:2].copy()
        d.xfrc_applied[self.pelvis, 0:2] = -PP_TETHER_K * (d.qpos[0:2] - self.rope_anchor) - PP_TETHER_D * d.qvel[0:2]
        d.xfrc_applied[self.pelvis, 2] = self.rope_share * self.weight

    # the motor: what it measures (q + encoder offset) and how strong it is
    def measured_q(self) -> np.ndarray:
        return self.d.qpos[self.qadr] + self.enc

    def measured_dq(self) -> np.ndarray:
        qd = self.d.qvel[self.vadr].copy()
        qd[LAG_IDX] = self.qd_hist[0][LAG_IDX]  # lag substeps old once the deque is full
        return qd

    def pd_torque(self, target):
        q, qd = self.measured_q(), self.d.qvel[self.vadr]
        k = self.f["strength"] * self.f["gain"]
        return np.clip(k * self.kp * (target - q) - k * self.kd * qd, -self.f["strength"] * self.effort,
                       self.f["strength"] * self.effort)

    def physics_step(self):
        m, d = self.m, self.d
        for _ in range(self.substeps):
            self.targets.append(self.applied.copy())
            target = self.targets[0]  # delay substeps old once the deque is full
            d.ctrl[self.act_id] = self.pd_torque(target)
            self.rope()
            mujoco.mj_step(m, d)
            held = self.held_z()
            if held is not None:
                # the gantry: the pelvis kinematically held at its start pose, at height held
                d.qpos[0:2], d.qpos[2], d.qpos[3:7] = self.hold_pose[0:2], held, self.hold_pose[3:7]
                d.qvel[0:6] = 0.0
            qd = d.qvel[self.vadr].copy()
            self.qd_hist.append(qd)
            self.sub += 1
            # the legs' motion, rad/s rms over 0.1 s: what drives the structure-borne shake
            self.leg_rms += m.opt.timestep / 0.1 * (math.sqrt(np.mean(qd[:12] ** 2)) - self.leg_rms)
            wave = math.sin(2 * math.pi * self.res_freq * self.sub * m.opt.timestep + self.res_phase)
            if self.lowstate:
                self.R_hist.append(d.site_xmat[self.imu_site].reshape(3, 3).copy())
            if self.gyro_lowpass is not None:
                u = self._raw_gyro()
                if self.f["gyro_k"] > 0 and self.head:
                    amp = self.f["gyro_k"] * GYRO_RMS_PER_DQ_RMS * math.sqrt(2) * self.leg_rms
                    u[1] += amp * wave
                if self.lowstate:
                    u += self.gyro_bias + LOWSTATE_GYRO_NOISE * self.rng.standard_normal(3)
                    u += self.f["vib_k"] * TORSO_GYRO_PER_DQ * math.sqrt(2) * self.leg_rms * wave * TORSO_GYRO_AXES
                self._gyro = self.gyro_lowpass(u, m.opt.timestep)
        self.last_tau = self.pd_torque(self.applied)

    def torso_roll(self):
        w, x, y, z = self.d.xquat[self.torso_body]
        return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def tilt(self):
        p, q = self.root()
        zb = qrot(q, np.array([0.0, 0.0, 1.0]))
        return math.acos(max(-1.0, min(1.0, zb[2])))

    def pitch(self):
        """The pelvis's pitch against gravity (rad), positive nose down (lmg_deploy's imu_calibration.tilt_deg)."""
        p, q = self.root()
        g = qrot(q * np.array([1.0, -1.0, -1.0, -1.0]), np.array([0.0, 0.0, -1.0]))
        return math.atan2(g[0], -g[2])


def command(t: float, profile: str) -> np.ndarray:
    if profile == "small60":
        # bag 1's trigger after a long stand: zero for 60 s, then the small profile's command
        t, profile = t - 50.0, "small"
    if profile == "zero" or t < 10.0:
        return np.zeros(3)
    vx = 0.04 if t < 12.0 else 0.04 + (0.3 - 0.04) * min((t - 12.0) / 2.0, 1.0)
    return np.array([vx, 0.01, 0.04])


def prepose_pose(sim: StressMujoco, start: str, rng: np.random.Generator) -> np.ndarray:
    """The joint angles (policy order) the pre-pose eases to before the engage."""
    if start in BAG_ENGAGE_Q:
        return np.load(BAG_ENGAGE_Q[start]).astype(float)
    if start != "random":
        return sim.default.copy()
    q = sim.default.copy()
    for i, name in enumerate(ALL_NAMES):
        for key, (lo, hi) in PREPOSE_JOINT_RANGES.items():
            if key in name:
                q[i] += rng.uniform(lo, hi)
                break
    return np.clip(q, sim.lower, sim.upper)


def run_prepose_job(job, video: str | None = None):
    """One pre-pose engage (module docstring, prepose): the sequence, the engage, PP_AFTER_S of policy."""
    torch.set_num_threads(1)
    export, deploy, cond, profile, plant, seed, seconds = job
    f = dict(NOMINAL, **CONDITIONS[cond])
    actors = {a: Recurrent(torch.jit.load(os.path.join(export, f"{a}.pt"), map_location="cpu").eval())
              for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = StressMujoco(actors, f, deploy, plant, rng)
    sim.reset(rng)
    d, start, DT = sim.d, f["prepose"], sim.policy_dt
    d.qpos[sim.qadr], d.qvel[:] = sim.default, 0.0
    mujoco.mj_forward(sim.m, d)
    sim.hold_pose = d.qpos[0:7].copy()
    d.qpos[2] += HANG_LIFT
    mujoco.mj_forward(sim.m, d)
    lowered = PP_ENGAGE_S + PP_HUNG_HOLD_S if start == "hung" else PP_HANG_S
    sim.pp_held = lambda t: (sim.hold_pose[2] + HANG_LIFT if t < lowered else
                             sim.hold_pose[2] + HANG_LIFT * (1 - (t - lowered) / PP_LOWER_S) if t < lowered + PP_LOWER_S
                             else None)
    pose = prepose_pose(sim, start, rng)
    if sim.lowstate:
        sim.R_hist.clear()
        sim.R_hist.append(d.site_xmat[sim.imu_site].reshape(3, 3).copy())
    writer = renderer = None
    if video:
        import imageio
        W, H = 640, 544
        sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight = max(W, sim.m.vis.global_.offwidth), max(H, sim.m.vis.global_.offheight)
        renderer = mujoco.Renderer(sim.m, H, W)
        cams = []
        for az in (90.0, 180.0):  # side, front
            c = mujoco.MjvCamera()
            c.type, c.trackbodyid, c.distance, c.elevation, c.azimuth = mujoco.mjtCamera.mjCAMERA_TRACKING, sim.pelvis, 3.2, -12.0, az
            cams.append(c)
        writer = imageio.get_writer(video, fps=50, codec="libx264", quality=8, macro_block_size=8)
    engaged, obs, rec, k = False, None, [], 0
    fell_t, w0 = None, None
    total = PP_ENGAGE_S + PP_AFTER_S
    with torch.no_grad():
        while k * DT < total:
            t = k * DT
            on_feet = t >= lowered + PP_LOWER_S
            sim.rope_share = (f["rope"] if on_feet and not (engaged and f["release"]) else 0.0)
            if start == "hung" and on_feet and f["release"]:
                sim.rope_share = 0.0
            if not engaged and t >= PP_ENGAGE_S:
                # lmg_deploy's controller.reset: LSTM zeroed, the previous actions the measured offsets, the arms'
                # residual zero and their IK from the measured angles, a zero command at the gait clock's start
                for a in actors.values():
                    a.reset_state()
                q = sim.measured_q()
                # the engage blend (export.yaml engage_blend_s) starts from the last command sent, the pre-pose's
                sim.blend_from = sim.applied[sim.leg_idx].copy()
                sim.policy_target = sim.blend_from.copy()
                sim.applied[:] = q
                sim.residual[:] = 0.0
                sim.ik_q = q[sim.arm_idx].copy()
                sim.traj_v[:] = 0.0
                sim.integral[:] = 0.0
                sim.steps, sim.cmd_filtered[:], sim.vel_cmd = 0, 0.0, np.zeros(3)
                engaged = True
                obs = sim.observations()
                w0 = np.linalg.norm(sim.wrist_poses_b()[:, :3] - sim.believed[:, :3], axis=1).mean()
            prev = sim.applied.copy()
            prev_policy = getattr(sim, "policy_target", sim.applied[:12]).copy()
            if engaged:
                obs = sim.step(obs)
            else:
                # the pre-pose at the policy's gains: the default pose, eased to the start pose once on the feet
                t0 = PP_HANG_S + PP_LOWER_S + 0.5
                wgt = 0.0 if start in ("settled", "hung") else float(np.clip((t - t0) / PP_EASE_S, 0.0, 1.0))
                sim.applied[:] = sim.default + wgt * (pose - sim.default)
                sim.physics_step()
            jump = np.abs(sim.applied - prev)
            q = sim.measured_q()
            tilt = sim.tilt()
            rec.append({"t": t - PP_ENGAGE_S, "engaged": engaged, "jump_legs": jump[:12].copy(), "jump_arms": jump[13:].copy(),
                        "jump_policy": np.abs(sim.policy_target - prev_policy) if engaged else np.zeros(12),
                        "tau_share": float(np.max(np.abs(sim.last_tau[:12]) / sim.effort[:12])),
                        "knee_off": float(np.abs(q[KNEE_IDX] - sim.default[KNEE_IDX]).mean()),
                        "target": sim.applied[:12].copy(), "tilt": tilt, "pitch": sim.pitch(), "z": float(d.qpos[2]),
                        "wrist_err": float(np.linalg.norm(sim.wrist_poses_b()[:, :3] - sim.believed[:, :3], axis=1).mean())})
            if engaged and fell_t is None and (tilt > FALL_TILT or d.qpos[2] < FALL_HEIGHT):
                fell_t = t - PP_ENGAGE_S
            if writer is not None and k % max(1, int(round(1 / 50 / DT))) == 0:
                import cv2
                frames = []
                for c in cams:
                    renderer.update_scene(d, c)
                    frames.append(renderer.render().copy())
                img = np.concatenate(frames, axis=1)
                if not engaged:
                    phase = ("PRE-POSE: on gantry" if not on_feet and t < lowered else "PRE-POSE: lowering" if not on_feet
                             else f"PRE-POSE: rope {f['rope']:.0%}" + ("" if start in ("settled", "hung") else f", easing to the {start} pose"))
                else:
                    phase = f"POLICY ENGAGED +{t - PP_ENGAGE_S:.2f} s" + (" (rope released)" if f["release"] else f" (rope {f['rope']:.0%})")
                lines = [f"{cond}  {os.path.basename(os.path.normpath(export))}", f"t = {t:5.2f} s   {phase}",
                         f"leg target jump this step: max {jump[:12].max():.2f} rad   max leg torque {rec[-1]['tau_share']:.0%} of limit",
                         f"knees off default {rec[-1]['knee_off']:.2f} rad   tilt {math.degrees(tilt):4.1f} deg" + ("   FELL" if fell_t is not None else "")]
                for i, line in enumerate(lines):
                    y = 26 + 24 * i
                    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                writer.append_data(img)
            k += 1
            if fell_t is not None and t - PP_ENGAGE_S > fell_t + 1.5:
                break
    if writer is not None:
        writer.close()
    eng = [r for r in rec if r["engaged"]]
    first = [r for r in eng if r["t"] < 0.5]
    after = [r for r in eng if r["t"] >= 2.0]
    T = np.array([r["target"] for r in after]) if after else np.zeros((0, 12))
    lo, hi = sim.lower[:12], sim.upper[:12]
    at_limit = float(((np.abs(T - lo) < 2e-3) | (np.abs(T - hi) < 2e-3)).any(1).mean()) if len(T) else None
    err = lambda s: next((r["wrist_err"] for r in eng if r["t"] >= s), None)  # noqa: E731
    return {"cond": cond, "profile": profile, "plant": plant, "seed": seed, "prepose": True,
            "fell": fell_t is not None, "fall_t": fell_t,
            "first_mean_jump": float(np.mean([r["jump_legs"].mean() for r in first])),
            "first_max_jump": float(np.max([r["jump_legs"].max() for r in first])),
            # 0.5-1 s after the engage (a blend must not just move the jump past the first 0.5 s), and the policy's own
            # targets before the engage blend (what the robot would get without it)
            "second_mean_jump": float(np.mean([r["jump_legs"].mean() for r in eng if 0.5 <= r["t"] < 1.0])),
            "policy_mean_jump": float(np.mean([r["jump_policy"].mean() for r in first])),
            "policy_max_jump": float(np.max([r["jump_policy"].max() for r in first])),
            "first_torque_share": float(np.max([r["tau_share"] for r in first])),
            "knee_off": float(np.mean([r["knee_off"] for r in after])) if after else None,
            "at_limit": at_limit,
            "pre_knee_off": float(np.mean([r["knee_off"] for r in rec if not r["engaged"] and r["t"] > -0.5])),
            "wrist_err_mm": [1000 * x if x is not None else None for x in (w0, err(0.25), err(0.5), err(1.0))],
            "arm_max_jump": float(np.max([r["jump_arms"].max() for r in eng])),
            "max_tilt_deg": float(math.degrees(max(r["tilt"] for r in eng))),
            # the stance's mean pitch from 2 s on (deg, + nose down)
            "stance_pitch_deg": float(math.degrees(np.mean([r["pitch"] for r in after]))) if after and fell_t is None else None}


def run_job(job):
    torch.set_num_threads(1)
    export, deploy, cond, profile, plant, seed, seconds = job
    f = dict(NOMINAL, **CONDITIONS[cond])
    if f["prepose"]:
        return run_prepose_job(job)
    actors = {a: Recurrent(torch.jit.load(os.path.join(export, f"{a}.pt"), map_location="cpu").eval())
              for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = StressMujoco(actors, f, deploy, plant, rng)
    sim.reset(rng)
    DT = sim.policy_dt
    obs = sim.observations()
    for a, net in actors.items():
        if net.net.mean.numel() != len(obs[a]):
            raise SystemExit(f"{a}: the network takes {net.net.mean.numel()} inputs, the sim gives {len(obs[a])}")
    steps = int(seconds / DT)
    lower = sim.lower[ANKLE_IDX]
    rec = {k: [] for k in ("target", "dq", "q_knee", "tau_knee", "fz", "roll", "z", "tilt", "lstm")}
    t0 = f["hang_s"] + LOWER_S if f["hang_s"] > 0 else 0.0  # on its feet from here
    fell_at = steps
    with torch.no_grad():
        for k in range(steps):
            sim.vel_cmd = command(k * DT - t0, profile)
            period = int(round(f["lstm_reset_s"] / DT)) if f["lstm_reset_s"] > 0 else 0
            if period and k > 0 and k % period == 0:
                for a in ("legs", "arms"):
                    if f["lstm_reset_which"] in ("both", a):
                        actors[a].reset_state()
            obs = sim.step(obs)
            d = sim.d
            rec["target"].append(sim.applied[:12].copy())
            rec["dq"].append(d.qvel[sim.vadr][:12].copy())
            rec["q_knee"].append(d.qpos[sim.qadr][KNEE_IDX].copy())
            rec["tau_knee"].append(sim.last_tau[KNEE_IDX].copy())
            rec["fz"].append(sim.foot_forces())
            rec["roll"].append(sim.torso_roll())
            rec["z"].append(d.qpos[2])
            rec["tilt"].append(sim.tilt())
            rec["lstm"].append(actors["legs"].state_norm())
            if rec["tilt"][-1] > FALL_TILT or rec["z"][-1] < FALL_HEIGHT:
                fell_at = k
                break
    A = {k: np.array(v) for k, v in rec.items()}
    n = len(A["z"])
    t = np.arange(n) * DT - t0
    # ankle at its floor for over FLOOR_HOLD_S
    at_floor = (A["target"][:, [4, 10]] <= lower + FLOOR_MARGIN).any(1)
    run, floor_hold = 0.0, False
    for v in at_floor:
        run = run + DT if v else 0.0
        floor_hold |= run > FLOOR_HOLD_S
    W = int(1.0 / DT)
    settled = t >= STAND_WINDOW[0]
    dq2 = np.where(settled, (A["dq"] ** 2).mean(1), 0.0)
    if n > W:
        c = np.concatenate([[0.0], np.cumsum(dq2)])
        win = np.sqrt((c[W:] - c[:-W]) / W)
        worst = int(win.argmax())
        seg = A["target"][worst:worst + W] - A["target"][worst:worst + W].mean(0)
        p = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).sum(1)
        freq = float(np.fft.rfftfreq(len(seg), DT)[1:][p[1:].argmax()])
        worst_rms = float(win.max())
    else:
        freq, worst_rms = float("nan"), float("nan")
    s0, s1 = (int((x + t0) / DT) for x in STAND_WINDOW)
    stand = {}
    if n > s1:
        fz = A["fz"][s0:s1].clip(min=0)
        share = float(fz[:, 0].sum() / max(fz.sum(), 1e-6))
        stand = {"knee_q": A["q_knee"][s0:s1].mean(0).tolist(), "knee_tau": A["tau_knee"][s0:s1].mean(0).tolist(),
                 "left_share": share, "imbalance": abs(share - 0.5) * 2,
                 "ankle_target": A["target"][s0:s1][:, [4, 10]].mean(0).tolist(),
                 "roll_deg": float(np.degrees(A["roll"][s0:s1].mean()))}
    lstm = A["lstm"]
    late = lstm[int(t0 / DT) + int(5 / DT):]
    return {"cond": cond, "profile": profile, "plant": plant, "seed": seed, "fell": fell_at < steps,
            "fall_t": float(fell_at * DT - t0) if fell_at < steps else None, "ankle_floor": bool(floor_hold),
            "lstm_c_5s": float(lstm[min(int((t0 + 5) / DT), n - 1)]), "lstm_c_end": float(lstm[-1]),
            "lstm_c_max_after_5s": float(late.max()) if len(late) else None,
            "last_ankle_target": A["target"][-int(2 / DT):][:, [4, 10]].mean(0).tolist(),
            "worst_dq_rms": worst_rms, "freq": freq, **stand}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--profiles", default="zero,small")
    parser.add_argument("--plants", default="robocasa,isaac")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 4, 1))
    parser.add_argument("--out", default=None)
    parser.add_argument("--video", default=None, help="render one job of this pp_* condition (seed 0) to an mp4")
    args = parser.parse_args()
    deploy = load_deploy_cfg(args.export)
    out = Path(args.out or REPO / "logs/hw_stress_mujoco" / args.label)
    if args.video:
        out.mkdir(parents=True, exist_ok=True)
        path = str(out / f"{args.video}.mp4")
        plant = args.plants.split(",")[0]
        r = run_prepose_job((os.path.abspath(args.export), deploy, args.video, "zero", plant, 0, 0.0), video=path)
        print(json.dumps(r), flush=True)
        print(f"[stress-mj] wrote {path}", flush=True)
        return
    head = deploy.get("imu", {}).get("frame", "imu_link") != "imu_link"
    names = args.conditions.split(",") if args.conditions else [c for c in CONDITIONS if not c.startswith("pp_")]
    if "pp" in names:  # the pre-pose grid
        names = [c for c in names if c != "pp"] + [c for c in CONDITIONS if c.startswith("pp_")]
    if not head:
        names = [c for c in names if not (CONDITIONS[c].get("gyro_k") and not CONDITIONS[c].get("prepose"))]
    jobs = [(os.path.abspath(args.export), deploy, c, p, pl, s, args.seconds) for c in names
            for p in (["zero"] if CONDITIONS[c].get("prepose") else args.profiles.split(","))
            for pl in args.plants.split(",") for s in range(args.seeds)]
    print(f"[stress-mj] {args.label}: {len(jobs)} jobs on {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"[stress-mj] {i + 1}/{len(jobs)}", flush=True)

    cells, pp_cells = {}, {}
    for c in [c for c in names if CONDITIONS[c].get("prepose")]:
        for pl in args.plants.split(","):
            rs = [r for r in results if r["cond"] == c and r["plant"] == pl]
            med = lambda k: float(np.median([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None  # noqa: E731
            worst = lambda k: float(np.max([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None  # noqa: E731
            cell = {"runs": len(rs), "fell_pct": 100 * float(np.mean([r["fell"] for r in rs])),
                    "first_mean_jump": med("first_mean_jump"), "first_mean_jump_worst": worst("first_mean_jump"),
                    "first_max_jump": worst("first_max_jump"), "first_torque_share": worst("first_torque_share"),
                    "second_mean_jump_worst": worst("second_mean_jump"),
                    "policy_mean_jump": med("policy_mean_jump"), "policy_mean_jump_worst": worst("policy_mean_jump"),
                    "policy_max_jump": worst("policy_max_jump"),
                    "knee_off": med("knee_off"), "knee_off_worst": worst("knee_off"), "at_limit": worst("at_limit"),
                    "pre_knee_off": med("pre_knee_off"), "arm_max_jump": worst("arm_max_jump"),
                    "max_tilt_deg": worst("max_tilt_deg"),
                    "stance_pitch_deg": med("stance_pitch_deg"),
                    "stance_pitch_range": ([float(np.min(sp)), float(np.max(sp))] if (sp := [r["stance_pitch_deg"] for r in rs
                                           if r["stance_pitch_deg"] is not None]) else None),
                    "wrist_err_mm": np.median([r["wrist_err_mm"] for r in rs], axis=0).tolist()}
            cell["pass"] = bool(cell["fell_pct"] == 0 and cell["first_mean_jump_worst"] <= PP_MAX_MEAN_JUMP
                                and (cell["second_mean_jump_worst"] or 0.0) <= PP_MAX_MEAN_JUMP
                                and cell["first_torque_share"] <= PP_MAX_TORQUE_SHARE
                                and (cell["knee_off_worst"] or 0.0) <= PP_MAX_KNEE_OFFSET
                                and (cell["at_limit"] or 0.0) < PP_MAX_AT_LIMIT
                                and cell["stance_pitch_range"] is not None
                                and max(map(abs, cell["stance_pitch_range"])) <= PP_MAX_STANCE_PITCH_DEG)
            pp_cells[f"{c}/{pl}"] = cell
    for c in [c for c in names if not CONDITIONS[c].get("prepose")]:
        for pl in args.plants.split(","):
            for p in args.profiles.split(","):
                rs = [r for r in results if r["cond"] == c and r["plant"] == pl and r["profile"] == p]
                st = [r for r in rs if "knee_q" in r]
                mean = lambda k: np.mean([r[k] for r in st], axis=0).tolist() if st else None  # noqa: E731
                falls = [r["fall_t"] for r in rs if r["fell"]]
                cells[f"{c}/{pl}/{p}"] = {
                    "runs": len(rs), "fell_pct": 100 * len(falls) / len(rs),
                    "fall_t_median": float(np.median(falls)) if falls else None,
                    "ankle_floor_pct": 100 * float(np.mean([r["ankle_floor"] for r in rs])),
                    "worst_dq_rms_median": float(np.nanmedian([r["worst_dq_rms"] for r in rs])),
                    "freq_median": float(np.nanmedian([r["freq"] for r in rs])),
                    "knee_q": mean("knee_q"), "knee_tau": mean("knee_tau"), "left_share": mean("left_share"),
                    "imbalance": mean("imbalance"), "ankle_target": mean("ankle_target"), "roll_deg": mean("roll_deg"),
                    "lstm_c_5s": float(np.mean([r["lstm_c_5s"] for r in rs])),
                    "lstm_c_end": float(np.mean([r["lstm_c_end"] for r in rs])),
                    "last_ankle_target": np.mean([r["last_ankle_target"] for r in rs], axis=0).tolist(),
                }
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"label": args.label, "export": os.path.abspath(args.export), "seeds": args.seeds,
               "seconds": args.seconds, "cells": cells, "pp_cells": pp_cells, "runs": results},
              open(out / "summary.json", "w"), indent=1, default=float)

    def fmt(v, f="{:.2f}"):
        if v is None:
            return "-"
        if isinstance(v, list):
            return "/".join(f.format(x) for x in v)
        return f.format(v)

    md = [f"# MuJoCo stress test: {args.label}", "", f"`{os.path.abspath(args.export)}`, {args.seeds} seeds per cell,"
          f" {args.seconds:g} s, CL_Assets h1_2_magpie.xml at 500 Hz.", "",
          "| cell | fell % | fall t s | ankle floor % | worst dq rms | f Hz | knee q L/R | knee tau L/R | left load |"
          " imbalance | ankle tgt L/R | roll deg | LSTM |c| 5 s / end | last 2 s ankle tgt |", "|" + "---|" * 14]
    for key, r in cells.items():
        md.append(f"| {key} | {r['fell_pct']:.0f} | {fmt(r['fall_t_median'], '{:.1f}')} | {r['ankle_floor_pct']:.0f} |"
                  f" {r['worst_dq_rms_median']:.2f} | {r['freq_median']:.1f} | {fmt(r['knee_q'])} |"
                  f" {fmt(r['knee_tau'], '{:.0f}')} | {fmt(r['left_share'])} | {fmt(r['imbalance'])} |"
                  f" {fmt(r['ankle_target'])} | {fmt(r['roll_deg'], '{:.1f}')} |"
                  f" {r['lstm_c_5s']:.3f} / {r['lstm_c_end']:.3f} | {fmt(r['last_ankle_target'])} |")
    if pp_cells:
        md += ["", f"Pre-pose engage (zero command; the first 0.5 s after the engage, the stance from 2 s on). Pass: no"
               f" fall, mean leg target jump <= {PP_MAX_MEAN_JUMP} rad (worst seed) in 0-0.5 s and in 0.5-1 s, leg torque"
               f" <= {PP_MAX_TORQUE_SHARE}"
               f" of the limit, knees within {PP_MAX_KNEE_OFFSET} rad of default, targets at a limit"
               f" < {PP_MAX_AT_LIMIT:.0%} of steps, the stance's pelvis pitch from 2 s on within"
               f" +-{PP_MAX_STANCE_PITCH_DEG} deg (every seed; + nose down).", "",
               "The policy's own jump: its leg targets before lmg_deploy's engage blend (export.yaml engage_blend_s; the"
               " same as the sent ones without it), reported only.", "",
               "| cell | pass | fell % | 0.5 s mean jump med / worst | 0.5-1 s worst | 0.5 s max jump | policy's own"
               " 0.5 s mean jump med / worst / max | 0.5 s torque/limit | knee off"
               " med / worst | at limit | knee off pre-pose | wrist err mm at 0/0.25/0.5/1 s | arm max jump | max tilt deg |"
               " stance pitch deg med (min / max) |",
               "|" + "---|" * 15]
        for key, r in pp_cells.items():
            md.append(f"| {key} | {'yes' if r['pass'] else 'NO'} | {r['fell_pct']:.0f} | {fmt(r['first_mean_jump'], '{:.3f}')} /"
                      f" {fmt(r['first_mean_jump_worst'], '{:.3f}')} | {fmt(r.get('second_mean_jump_worst'), '{:.3f}')} |"
                      f" {fmt(r['first_max_jump'])} | {fmt(r.get('policy_mean_jump'), '{:.3f}')} /"
                      f" {fmt(r.get('policy_mean_jump_worst'), '{:.3f}')} / {fmt(r.get('policy_max_jump'))} |"
                      f" {fmt(r['first_torque_share'])} | {fmt(r['knee_off'])} / {fmt(r['knee_off_worst'])} |"
                      f" {fmt(r['at_limit'], '{:.1%}')} | {fmt(r['pre_knee_off'])} | {fmt(r['wrist_err_mm'], '{:.0f}')} |"
                      f" {fmt(r['arm_max_jump'])} | {fmt(r['max_tilt_deg'], '{:.1f}')} |"
                      f" {fmt(r.get('stance_pitch_deg'), '{:+.1f}')} ({fmt(r.get('stance_pitch_range'), '{:+.1f}')}) |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    print(f"[stress-mj] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
