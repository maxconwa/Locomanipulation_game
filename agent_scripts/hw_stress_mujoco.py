"""The hardware-perturbation stress test (hw_stress_test.py) in MuJoCo: an exported policy (GOLEM lmg_deploy's legs.pt /
arms.pt, TorchScript, LSTM state carried) on CL_Assets' H1-2 MJCF, the model GOLEM's RoboCasa sim uses, at 500 Hz.

Builds on mujoco_evals.mujoco_sim.MarlMujoco (the env's action bounds, PD, IK and observations) and adds what the
Mid360 policies need and the perturbations:
  IMU            the run's: the MID-360 chip (livox_imu_site) or the torso (imu), with its gyro low-pass
  gait clock     each leg's phase sin (and cos) from the episode time, standing (phase 0) under a 0.1 command
  plant          robocasa: the MJCF's joint dynamics (damping 10, frictionloss 0.2, armature 0.1, as RoboCasa);
                 isaac: the training plant's (none, armature 0.01)
  strength, gain, dq_lag, gyro_k, delay_ms, torso_kg, enc   as in hw_stress_test.py (CONDITIONS there)
Per job (one condition, profile, plant and seed) one robot for --seconds; jobs run in parallel processes.

    python agent_scripts/hw_stress_mujoco.py --export <policies/name> --label <name> [--seeds 8] [--plants robocasa,isaac]
        [--conditions ...] [--profiles zero,small] [--workers 24]

Writes logs/hw_stress_mujoco/<label>/summary.json and summary.md. Any python with mujoco, torch, numpy and pyyaml.
"""

from __future__ import annotations

import argparse
import glob
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
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mujoco_evals import common as C  # noqa: E402
from mujoco_evals.mujoco_sim import (  # noqa: E402
    ALL_NAMES, MarlMujoco, axis_angle, mjcf_path, qconj, qmul, qrot, qunique, xyzw)

# the same conditions as the Isaac stress test
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
    # the actors' LSTM state zeroed every 10 s (both, the legs' only, the arms' only): does the policy go wrong
    # past the training episode's 20 s because its state has run longer than any episode?
    "lstm_reset": {"lstm_reset_s": 10.0},
    "lstm_reset_legs": {"lstm_reset_s": 10.0, "lstm_reset_which": "legs"},
    "lstm_reset_arms": {"lstm_reset_s": 10.0, "lstm_reset_which": "arms"},
}
NOMINAL = {"strength": 1.0, "gain": 1.0, "dq_lag": 0.0, "gyro_k": 0.0, "delay_ms": 10.0, "torso_kg": 0.0, "enc": 0.0,
           "hang_s": 0.0, "lstm_reset_s": 0.0, "lstm_reset_which": "both"}
HANG_LIFT, LOWER_S = 0.08, 1.0
GYRO_RMS_PER_DQ_RMS = 3.4
LAG_IDX = [ALL_NAMES.index(f"{s}_{j}_joint") for s in ("left", "right") for j in ("hip_pitch", "knee")]
ANKLE_IDX = [ALL_NAMES.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
KNEE_IDX = [ALL_NAMES.index(n) for n in ("left_knee_joint", "right_knee_joint")]
FALL_TILT, FALL_HEIGHT = 0.8, 0.55
FLOOR_MARGIN, FLOOR_HOLD_S = 0.1, 0.5
STAND_WINDOW = (2.0, 10.0)
DT = C.POLICY_DT


class Recurrent:
    """An exported actor called as actor(obs[None]) -> action[None], carrying its LSTM state."""

    def __init__(self, net):
        self.net = net
        self.recurrent = hasattr(net, "hidden_size")
        if self.recurrent:
            self.h = torch.zeros(net.num_layers, 1, net.hidden_size)
            self.c = torch.zeros_like(self.h)

    def __call__(self, x):
        if not self.recurrent:
            return self.net(x)
        a, self.h, self.c = self.net(x, self.h, self.c)
        return a

    def reset_state(self):
        if self.recurrent:
            self.h.zero_()
            self.c.zero_()

    def state_norm(self) -> float:
        return float(self.c.abs().mean()) if self.recurrent else 0.0


class StressMujoco(MarlMujoco):
    def __init__(self, actors, f: dict, deploy: dict, physics: str, rng: np.random.Generator):
        imu = deploy.get("imu", {})
        self.head = imu.get("frame", "imu_link") != "imu_link"
        self.gait_cos = bool(deploy.get("gait_phase_cos", False))
        # the gait clock and the stand/walk gate as the run trained them (export.yaml over the template's 0.8 s, 0.1)
        self.gait_period, self.zero_cmd_eps = float(deploy["gait_period"]), float(deploy["zero_cmd_eps"])
        model =mujoco.MjModel.from_xml_path(str(mjcf_path()))
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        model.body_mass[torso] += f["torso_kg"]
        rest = np.array(deploy["rest_pose"], float)
        limits = {k: tuple(v) for k, v in (deploy.get("target_limits") or {}).items()}
        super().__init__(actors, None, physics=physics, rest_pose_b=rest, model=model, target_limits=limits,
                         gyro_lowpass_hz=imu.get("gyro_lowpass_hz") or 0.0)
        m = self.m
        if self.head:
            sens = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]  # noqa: E731
            self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "livox_imu_site")
            self.s_acc, self.s_gyro = sens("livox_imu_acc"), sens("livox_imu_gyro")
        self.f = f
        self.enc = np.zeros(len(ALL_NAMES))
        self.enc[:12] = rng.uniform(-f["enc"], f["enc"], 12)
        self.delay = int(round(f["delay_ms"] / 1000 / m.opt.timestep))
        self.lag = int(round(f["dq_lag"] / 1000 / m.opt.timestep))
        self.res_freq = 20.0 + rng.uniform()
        self.res_phase = 2 * math.pi * rng.uniform()
        self.torso_body = torso

    def held_z(self):
        """The gantry's pelvis height now, or None once let go."""
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
        self.steps = 0
        super().reset(rng)
        self.hold_pose = self.d.qpos[0:7].copy()
        if self.f["hang_s"] > 0:
            self.d.qpos[2] += HANG_LIFT
            mujoco.mj_forward(self.m, self.d)
        self.qd_hist.append(self.d.qvel[self.vadr].copy())

    # the motor: what it measures (q + encoder offset) and how strong it is
    def pd_torque(self, target):
        q, qd = self.d.qpos[self.qadr] + self.enc, self.d.qvel[self.vadr]
        k = self.f["strength"] * self.f["gain"]
        return np.clip(k * self.kp * (target - q) - k * self.kd * qd, -self.f["strength"] * self.effort,
                       self.f["strength"] * self.effort)

    def _torque_bounds(self, idx):
        # the deploy's bound: nominal gains on the measured state
        q, qd = (self.d.qpos[self.qadr] + self.enc)[idx], self.d.qvel[self.vadr][idx]
        h = 0.85 * self.effort[idx]
        return q + (self.kd[idx] * qd - h) / self.kp[idx], q + (self.kd[idx] * qd + h) / self.kp[idx]

    def physics_step(self):
        m, d = self.m, self.d
        for _ in range(self.substeps):
            self.targets.append(self.applied.copy())
            target = self.targets[0]  # delay substeps old once the deque is full
            d.ctrl[self.act_id] = self.pd_torque(target)
            mujoco.mj_step(m, d)
            held = self.held_z()
            if held is not None:
                # the gantry: the pelvis kinematically held at its start pose, at height held
                d.qpos[0:2], d.qpos[2], d.qpos[3:7] = self.hold_pose[0:2], held, self.hold_pose[3:7]
                d.qvel[0:6] = 0.0
            qd = d.qvel[self.vadr].copy()
            self.qd_hist.append(qd)
            self.sub += 1
            if self.gyro_lowpass is not None:
                u = self._raw_gyro()
                if self.f["gyro_k"] > 0 and self.head:
                    alpha = m.opt.timestep / 0.1
                    self.leg_rms += alpha * (math.sqrt(np.mean(qd[:12] ** 2)) - self.leg_rms)
                    amp = self.f["gyro_k"] * GYRO_RMS_PER_DQ_RMS * math.sqrt(2) * self.leg_rms
                    u[1] += amp * math.sin(2 * math.pi * self.res_freq * self.sub * m.opt.timestep + self.res_phase)
                self._gyro = self.gyro_lowpass(u, m.opt.timestep)
        self.last_tau = self.pd_torque(self.applied)

    def gait(self):
        if np.linalg.norm(self.vel_cmd) < self.zero_cmd_eps:
            phase = np.zeros(2)
        else:
            p = (self.steps * DT % self.gait_period) / self.gait_period
            phase = np.array([p, (p + 0.5) % 1.0])
        return np.sin(2 * math.pi * phase), np.cos(2 * math.pi * phase)

    def observations(self):
        acc, gyro, grav = self.imu()
        q = self.d.qpos[self.qadr] + self.enc
        qd = self.d.qvel[self.vadr].copy()
        qd[LAG_IDX] = self.qd_hist[0][LAG_IDX]  # lag substeps old once the deque is full
        n, r = self.NOISE_POLICY, self.rng
        u = lambda x, s: x + r.uniform(-s, s, np.shape(x))  # noqa: E731
        believed = np.concatenate([xyzw(b) for b in self.believed])
        sin, cos = self.gait()
        common = [u(acc, n["lin_acc"]), u(gyro, n["ang_vel"]), u(grav, n["grav"]), self.vel_cmd,
                  np.array([1.0 if self.arm_mode else 0.0]), believed, u(q - self.default, n["jpos"]),
                  u(qd, n["jvel"]), sin] + ([cos] if self.gait_cos else [])
        legs_applied = (self.applied[self.leg_idx] - self.default[self.leg_idx]) / 0.25
        cur = self.wrist_poses_b()
        wrist_pose = np.concatenate([xyzw(np.r_[c[:3], qunique(c[3:])]) for c in cur])
        errs = np.concatenate([np.r_[b[:3] - c[:3], axis_angle(qmul(b[3:], qconj(c[3:])))]
                               for b, c in zip(self.believed, cur)])
        return {"legs": np.concatenate(common + [legs_applied]),
                "arms": np.concatenate(common + [self.residual, u(wrist_pose, n["wrist"]), u(errs, n["wrist"])])}

    def step(self, obs, **kw):
        self.steps += 1
        return super().step(obs, **kw)

    def torso_roll(self):
        w, x, y, z = self.d.xquat[self.torso_body]
        return math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def tilt(self):
        p, q = self.root()
        zb = qrot(q, np.array([0.0, 0.0, 1.0]))
        return math.acos(max(-1.0, min(1.0, zb[2])))


def command(t: float, profile: str) -> np.ndarray:
    if profile == "zero" or t < 10.0:
        return np.zeros(3)
    vx = 0.04 if t < 12.0 else 0.04 + (0.3 - 0.04) * min((t - 12.0) / 2.0, 1.0)
    return np.array([vx, 0.01, 0.04])


def run_job(job):
    torch.set_num_threads(1)
    export, deploy, cond, profile, plant, seed, seconds = job
    f = dict(NOMINAL, **CONDITIONS[cond])
    actors = {a: Recurrent(torch.jit.load(os.path.join(export, f"{a}.pt"), map_location="cpu").eval())
              for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = StressMujoco(actors, f, deploy, plant, rng)
    sim.reset(rng)
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
    args = parser.parse_args()
    cfg_path = [p for p in glob.glob(os.path.join(args.export, "*.yaml")) if not p.endswith("export.yaml")][0]
    deploy = yaml.safe_load(open(cfg_path))
    deploy.update(yaml.safe_load(open(os.path.join(args.export, "export.yaml"))) or {})
    head = deploy.get("imu", {}).get("frame", "imu_link") != "imu_link"
    names = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    if not head:
        names = [c for c in names if not CONDITIONS[c].get("gyro_k")]
    jobs = [(os.path.abspath(args.export), deploy, c, p, pl, s, args.seconds) for c in names
            for p in args.profiles.split(",") for pl in args.plants.split(",") for s in range(args.seeds)]
    print(f"[stress-mj] {args.label}: {len(jobs)} jobs on {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"[stress-mj] {i + 1}/{len(jobs)}", flush=True)

    cells = {}
    for c in names:
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
    out = Path(args.out or REPO / "logs/hw_stress_mujoco" / args.label)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"label": args.label, "export": os.path.abspath(args.export), "seeds": args.seeds,
               "seconds": args.seconds, "cells": cells, "runs": results}, open(out / "summary.json", "w"), indent=1)

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
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    print(f"[stress-mj] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
