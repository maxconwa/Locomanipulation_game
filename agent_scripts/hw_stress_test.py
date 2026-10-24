"""Hardware-perturbation stress test: does an exported policy keep standing (and take a small command) when the
plant differs from training the way the real H1-2 does?

Runs GOLEM lmg_deploy's exported networks (legs.pt / arms.pt, TorchScript; LSTM state carried per env) in the
training env at the export's physics (500 Hz, decimation 10), on a flat plane, no pushes or domain randomization,
sensor noise as in training. The envs are split into cells: one perturbation condition x one command profile.

Conditions (CONDITIONS below; each a set of factors, the rest nominal):
  strength   motor torque x s (kp, kd and the effort limit together)
  gain       kp and kd x g (effort limit unchanged)
  dq_lag     the observed joint velocity of hip pitch and knee (LowState dq) delayed by this many ms
  gyro_k     the head's 20-21 Hz pitch resonance added to the IMU gyro before the 8 Hz low-pass, its amplitude
             gyro_k x 3.4 sqrt(2) x the legs' joint-velocity rms (EMA over 0.1 s): 3.5-4.7 rad/s rms when the legs
             move at 1.0-1.4 rad/s rms, the real bags' (MID-360 policies only; the torso IMU doesn't see it)
  delay_ms   joint targets reach the motors this late (nominal 10; training drew 0-20)
  torso_kg   added to torso_link's mass
  enc        per joint encoder offset U(+-enc) rad on the legs: the policy and the PD both see q + offset
  passive    RoboCasa MJCF joint dynamics: damping 10 N m s/rad, friction 0.2 N m, armature 0.1
Profiles: zero (0 command for the whole run) and small (bag 1's: 10 s zero, 2 s of (0.04, 0.01, 0.04), vx ramped to
0.3 over 2 s, then held).

Per cell: falls (pelvis tilt > 0.8 rad or pelvis under 0.55 m) and when, GOLEM e-stop trips, the share of envs with
an ankle-pitch target held within 0.1 rad of its floor for over 0.5 s, the worst 1 s window of leg dq rms (and its
dominant frequency), and over the 2-10 s stand: knee angles and torques, the left foot's share of the vertical load,
torso roll.

    python agent_scripts/hw_stress_test.py --export <policies/name> --label <name> --headless \
        [--envs_per_cell 32] [--conditions nominal,strength085,...] [--seconds 30]

Writes logs/hw_stress/<label>/summary.json, summary.md and traces.npz (4 envs per cell).
"""

import argparse
import glob
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--export", required=True, help="Folder with legs.pt, arms.pt, export.yaml and the deploy yaml.")
parser.add_argument("--label", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--envs_per_cell", type=int, default=32)
parser.add_argument("--conditions", default=None, help="Comma-separated CONDITIONS keys; default: all that apply.")
parser.add_argument("--profiles", default="zero,small")
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--tables", default=None, help="Arm target tables; default: export.yaml's source.target_tables.")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.imu_filter import GyroLowpass  # noqa: E402
from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES, LOWER_JOINT_NAMES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the hardware-like combination: the report's measured effects (dq lag 12 ms, the head resonance) and plausible
# actuator and latency mismatch
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
    "passive": {"passive": True},
    "hw": dict(_HW),
    "hw_no_gyro": {k: v for k, v in _HW.items() if k != "gyro_k"},
    "hw_no_dqlag": {k: v for k, v in _HW.items() if k != "dq_lag"},
    "hw_no_strength": {k: v for k, v in _HW.items() if k != "strength"},
    "hw_no_delay": {k: v for k, v in _HW.items() if k != "delay_ms"},
    "hw_no_enc": {k: v for k, v in _HW.items() if k != "enc"},
}
NOMINAL = {"strength": 1.0, "gain": 1.0, "dq_lag": 0.0, "gyro_k": 0.0, "delay_ms": 10.0, "torso_kg": 0.0, "enc": 0.0,
           "passive": False}
GYRO_RMS_PER_DQ_RMS = 3.4   # real bags: 3.5-4.7 rad/s raw pitch-rate rms at 1.0-1.4 rad/s leg dq rms
LAG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "knee")]
FALL_TILT, FALL_HEIGHT = 0.8, 0.55
FLOOR_MARGIN, FLOOR_HOLD_S = 0.1, 0.5
STAND_WINDOW = (2.0, 10.0)

# -- the export
cfg_path = [p for p in glob.glob(os.path.join(args.export, "*.yaml")) if not p.endswith("export.yaml")][0]
deploy = yaml.safe_load(open(cfg_path))
deploy.update(yaml.safe_load(open(os.path.join(args.export, "export.yaml"))) or {})
imu_frame = deploy.get("imu", {}).get("frame", "imu_link")
head_imu = imu_frame != "imu_link"
lowpass_hz = deploy.get("imu", {}).get("gyro_lowpass_hz")
gait_cos = bool(deploy.get("gait_phase_cos", False))
tables = args.tables or deploy.get("source", {}).get("target_tables")
if tables and not os.path.isfile(tables):
    tables = os.path.join(REPO, "logs/skrl/locomanip_marl/golemlim120k/arm_target_tables_v3.pt")

names = [c for c in (args.conditions.split(",") if args.conditions else CONDITIONS)]
if not head_imu:
    # the torso IMU sees no head resonance
    names = [n for n in names if not CONDITIONS[n].get("gyro_k")]
profiles = args.profiles.split(",")
cells = [(c, p) for c in names for p in profiles]
E = args.envs_per_cell
N = E * len(cells)
print(f"[stress] {args.label}: IMU {imu_frame} (low-pass {lowpass_hz}), gait cos {gait_cos}, {len(cells)} cells x {E}"
      f" envs = {N}", flush=True)

# -- env config
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = N
cfg.seed = args.seed
cfg.sim.dt = round(deploy["control_dt"] / 10, 6)
cfg.decimation = 10
cfg.action_delay_range = (0, 20)  # substeps; set per env below
cfg.action_delay_substeps = True
cfg.episode_length_s = 1.0e4
cfg.estimator.train = False
cfg.commands.arm_targets.arm_goal_prob = 0.0
if tables:
    cfg.commands.arm_targets.table_file = tables
cfg.scene.terrain.terrain_type = "plane"
cfg.scene.terrain.terrain_generator = None
cfg.curriculum.terrain_levels = None
cfg.curriculum.push_levels = None
cfg.events.push_robot = None
cfg.events.scale_link_masses = None
cfg.events.add_torso_mass = None
cfg.events.shift_pelvis_com = None
cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
cfg.events.imu_mount.params.update(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0, acc_bias_g=0.0)
cfg.terminations.fell = None
cfg.terminations.terrain_out_of_bounds = None
cfg.gyro_lowpass_hz = lowpass_hz
# the gait clock and the stand/walk gate as the run trained them (export.yaml over the template's 0.8 s and 0.1)
cfg.gait_period = float(deploy["gait_period"])
cfg.zero_cmd_eps = float(deploy["zero_cmd_eps"])
if not head_imu:
    cfg.scene.imu.prim_path = "{ENV_REGEX_NS}/Robot/imu_link"
if not gait_cos:
    cfg.observations.legs.gait_phase_cos = None
    cfg.observations.arms.gait_phase_cos = None

env = gym.make(args.task, cfg=cfg)
base = env.unwrapped
dev = base.device
robot = base.scene["robot"]
dt, pdt = base.step_dt, base.physics_dt
torch.manual_seed(args.seed)

# per env factors
fac = {k: torch.full((N,), float(v), device=dev) for k, v in NOMINAL.items()}
for i, (c, _) in enumerate(cells):
    for k, v in CONDITIONS[c].items():
        fac[k][i * E:(i + 1) * E] = float(v)
profile_small = torch.tensor([p == "small" for _, p in cells for _ in range(E)], device=dev)

# -- networks
legs_net = torch.jit.load(os.path.join(args.export, "legs.pt"), map_location=dev).eval()
arms_net = torch.jit.load(os.path.join(args.export, "arms.pt"), map_location=dev).eval()


def zero_state(net):
    if not hasattr(net, "hidden_size"):
        return None
    return [torch.zeros(net.num_layers, N, net.hidden_size, device=dev) for _ in range(2)]


nets = {"legs": legs_net, "arms": arms_net}
for agent, net in nets.items():
    want = net.mean.numel()
    have = base.observation_manager.group_obs_dim[agent][0]
    if want != have:
        raise SystemExit(f"{agent}: the network takes {want} inputs, the env gives {have}")

# -- actuators: strength, gain, encoder offsets, passive dynamics
leg_ids = robot.find_joints(LOWER_JOINT_NAMES, preserve_order=True)[0]
enc_offset = torch.zeros(N, robot.num_joints, device=dev)
enc_offset[:, leg_ids] = (2 * torch.rand(N, len(leg_ids), device=dev) - 1) * fac["enc"].unsqueeze(1)
passive = fac["passive"] > 0.5
for name, act in robot.actuators.items():
    if not hasattr(act, "stiffness") or name == "grippers":
        continue
    s, g = fac["strength"].unsqueeze(1), fac["gain"].unsqueeze(1)
    act.stiffness[:] = act.stiffness * s * g
    act.damping[:] = act.damping * s * g
    act.effort_limit[:] = act.effort_limit * s
    idx = act.joint_indices
    idx = torch.arange(robot.num_joints, device=dev)[idx] if isinstance(idx, slice) else torch.as_tensor(idx, device=dev)

    def compute(control_action, joint_pos, joint_vel, _act=act, _orig=act.compute, _idx=idx):
        out = _orig(control_action, joint_pos + enc_offset[:, _idx], joint_vel)
        tau = torch.where(passive.unsqueeze(1),
                          -10.0 * joint_vel - 0.2 * torch.tanh(joint_vel / 0.05), torch.zeros_like(joint_vel))
        out.joint_efforts = _act.applied_effort + tau
        return out

    act.compute = compute
if passive.any():
    ids = passive.nonzero().flatten()
    arm = torch.full((len(ids), len(leg_ids)), 0.1, device=dev)
    robot.write_joint_armature_to_sim(arm, joint_ids=leg_ids, env_ids=ids)

# torso mass
if (fac["torso_kg"] != 0).any():
    torso = robot.find_bodies("torso_link")[0][0]
    masses = robot.root_physx_view.get_masses()
    masses[:, torso] += fac["torso_kg"].cpu()
    robot.root_physx_view.set_masses(masses, torch.arange(N))

# -- joint-velocity lag: a ring of the state after each substep (pushed before the next one is applied)
RING = 24
ring = torch.zeros(RING, N, robot.num_joints, device=dev)
ring_pos = [0]
lag_ids = robot.find_joints(LAG_JOINTS, preserve_order=True)[0]
lag_sub = torch.round(fac["dq_lag"] / (1000 * pdt)).long().clamp(0, RING - 1)
_apply = base._apply_action


def apply_action():
    ring[ring_pos[0] % RING] = robot.data.joint_vel
    ring_pos[0] += 1
    _apply()


base._apply_action = apply_action
rows = torch.arange(N, device=dev)


def lagged_joint_vel(env_, asset_cfg):
    dq = robot.data.joint_vel.clone()
    slot = (ring_pos[0] - lag_sub) % RING
    old = ring[slot, rows]  # (N, J): lag_sub substeps back (lag 0 -> overwritten below)
    use = (lag_sub > 0).unsqueeze(1)
    dq[:, lag_ids] = torch.where(use, old[:, lag_ids], dq[:, lag_ids])
    return dq[:, asset_cfg.joint_ids] - robot.data.default_joint_vel[:, asset_cfg.joint_ids]


def offset_joint_pos(env_, asset_cfg):
    q = robot.data.joint_pos + enc_offset
    return q[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]


om = base.observation_manager
for group in ("legs", "arms"):
    for name, term in zip(om._group_obs_term_names[group], om._group_obs_term_cfgs[group]):
        if name == "joint_vel":
            term.func = lagged_joint_vel
        elif name == "joint_pos":
            term.func = offset_joint_pos


# -- the head resonance in the gyro, before the low-pass
class ResonantGyro:
    def __init__(self, inner):
        self.inner = inner
        self.t = 0
        self.freq = 20.0 + torch.rand(N, device=dev)
        self.phase = 2 * math.pi * torch.rand(N, device=dev)
        self.rms = torch.zeros(N, device=dev)
        self.alpha = pdt / 0.1
        # the injection alone through the same low-pass, to report what reaches the policy
        self.filtered = GyroLowpass(inner.cutoff_hz, pdt, N, dev, axes=1)
        self.raw_sq = torch.zeros(N, device=dev)
        self.filt_sq = torch.zeros(N, device=dev)
        self.peak_filt = torch.zeros(N, device=dev)

    @property
    def output(self):
        return self.inner.output

    def reset(self, env_ids=None):
        self.inner.reset(env_ids)

    def __call__(self, u):
        dq = robot.data.joint_vel[:, leg_ids]
        self.rms += self.alpha * (dq.pow(2).mean(1).sqrt() - self.rms)
        amp = fac["gyro_k"] * GYRO_RMS_PER_DQ_RMS * math.sqrt(2) * self.rms
        inj = amp * torch.sin(2 * math.pi * self.freq * self.t * pdt + self.phase)
        f = self.filtered(inj.unsqueeze(1))[:, 0]
        self.raw_sq += inj ** 2
        self.filt_sq += f ** 2
        self.peak_filt = torch.maximum(self.peak_filt, f.abs())
        u = u.clone()
        u[:, 1] += inj
        self.t += 1
        return self.inner(u)


if base.gyro_lowpass is not None and (fac["gyro_k"] > 0).any():
    base.gyro_lowpass = ResonantGyro(base.gyro_lowpass)

# -- commands
vel = base.command_manager.get_term("base_velocity")


def command(t: float) -> torch.Tensor:
    small = torch.zeros(3, device=dev)
    if t >= 10.0:
        vx = 0.04 if t < 12.0 else 0.04 + (0.3 - 0.04) * min((t - 12.0) / 2.0, 1.0)
        small = torch.tensor([vx, 0.01, 0.04], device=dev)
    return torch.where(profile_small.unsqueeze(1), small.expand(N, 3), torch.zeros(N, 3, device=dev))


def hold(cmd):
    vel.vel_command_b[:] = cmd
    vel.time_left[:] = 1.0e6
    vel.is_standing_env[:] = cmd.norm(dim=1) == 0


# -- run
legs_term = base.action_manager.get_term("joint_pos")
leg_names = list(legs_term._joint_names)
ank = [leg_names.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
knee = [leg_names.index(n) for n in ("left_knee_joint", "right_knee_joint")]
low = legs_term._lower
low = (low if low.dim() == 1 else low[0])[ank]
floor = low + FLOOR_MARGIN
foot_sensor = base.scene.sensors["contact_forces"]
foot_ids = foot_sensor.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
torso_body = robot.find_bodies("torso_link")[0][0]
steps = int(args.seconds / dt)
T = {k: [] for k in ("target", "dq", "q_knee", "tau_knee", "fz", "roll", "z", "tilt", "estop")}

with torch.no_grad():
    obs, _ = base.reset()
    base._delay_steps[:] = torch.round(fac["delay_ms"] / (1000 * pdt)).long().clamp(0, 20)
    states = {a: zero_state(n) for a, n in nets.items()}
    # what each cell's first env got
    kn = robot.find_joints("left_knee_joint")[0][0]
    legs_act = robot.actuators["legs"]
    col = list(legs_act.joint_names).index("left_knee_joint")
    for i, (c, p) in enumerate(cells):
        e = i * E
        print(f"[stress] cell {c}/{p}: knee kp {legs_act.stiffness[e, col].item():.0f} kd {legs_act.damping[e, col].item():.2f}"
              f" limit {legs_act.effort_limit[e, col].item():.0f}, delay {base._delay_steps[e].item()} substeps, dq lag"
              f" {lag_sub[e].item()} substeps, enc |max| {enc_offset[e].abs().max().item():.3f}, torso"
              f" {robot.root_physx_view.get_masses()[e, robot.find_bodies('torso_link')[0][0]].item():.1f} kg,"
              f" passive {bool(passive[e])}, gyro k {fac['gyro_k'][e].item():g}", flush=True)
    for k in range(steps):
        hold(command(k * dt))
        actions = {}
        for agent, net in nets.items():
            if states[agent] is None:
                actions[agent] = net(obs[agent])
            else:
                a, h, c = net(obs[agent], *states[agent])
                states[agent] = [h, c]
                actions[agent] = a
        obs, _, _, _, _ = base.step(actions)
        data = robot.data
        T["target"].append(legs_term._applied.half())
        T["dq"].append(data.joint_vel[:, leg_ids].half())
        T["q_knee"].append(data.joint_pos[:, leg_ids][:, knee])
        T["tau_knee"].append(data.applied_torque[:, leg_ids][:, knee])
        T["fz"].append(foot_sensor.data.net_forces_w[:, foot_ids, 2])
        roll, _, _ = euler_xyz_from_quat(data.body_quat_w[:, torso_body])
        T["roll"].append(torch.atan2(torch.sin(roll), torch.cos(roll)))
        T["z"].append(data.root_pos_w[:, 2] - base.scene.env_origins[:, 2])
        T["tilt"].append(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1)))
        T["estop"].append(base.golem_tripped.clone())
        if k % 250 == 0:
            print(f"[stress] t {k * dt:5.1f} s", flush=True)

A = {k: torch.stack(v).float().cpu().numpy() for k, v in T.items()}  # (steps, N, ...)
gyro = None
if isinstance(base.gyro_lowpass, ResonantGyro):
    g = base.gyro_lowpass
    gyro = {"raw": (g.raw_sq / g.t).sqrt().cpu().numpy(), "filt": (g.filt_sq / g.t).sqrt().cpu().numpy(),
            "peak": g.peak_filt.cpu().numpy()}
t = np.arange(steps) * dt
fell_step = np.where(((A["tilt"] > FALL_TILT) | (A["z"] < FALL_HEIGHT)).any(0),
                     ((A["tilt"] > FALL_TILT) | (A["z"] < FALL_HEIGHT)).argmax(0), steps)
alive = np.arange(steps)[:, None] < fell_step[None]
# ankle at its floor for over FLOOR_HOLD_S (before any fall)
at_floor = (A["target"][:, :, ank] <= floor.cpu().numpy()).any(-1) & alive
run = np.zeros(N)
floor_hold = np.zeros(N, bool)
floor_first = np.full(N, np.nan)
for k in range(steps):
    run = np.where(at_floor[k], run + dt, 0.0)
    new = (run > FLOOR_HOLD_S) & ~floor_hold
    floor_first[new] = t[k]
    floor_hold |= new
# worst 1 s window of leg dq rms, and its dominant frequency (leg targets)
W = int(1.0 / dt)
dq2 = (A["dq"] ** 2).mean(-1)
settled = alive & (t[:, None] >= STAND_WINDOW[0])
csum = np.concatenate([np.zeros((1, N)), np.cumsum(np.where(settled, dq2, 0.0), 0)])
win = np.sqrt((csum[W:] - csum[:-W]) / W)
worst = win.argmax(0)
freq = np.zeros(N)
for i in range(N):
    seg = A["target"][worst[i]:worst[i] + W, i].astype(np.float64)
    seg = seg - seg.mean(0)
    p = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).sum(-1)
    f = np.fft.rfftfreq(W, dt)
    freq[i] = f[1:][p[1:].argmax()]
target_step = np.sqrt((np.diff(A["target"], axis=0) ** 2).mean(-1))  # rad per tick
s0, s1 = (int(x / dt) for x in STAND_WINDOW)
stand_ok = fell_step > s1
estop_step = np.where(A["estop"].any(0), A["estop"].argmax(0), steps)


def cell_summary(sel):
    ok = sel & stand_ok
    fz = A["fz"][s0:s1, ok].clip(min=0)
    share_left = fz[..., 0].sum(0) / np.maximum(fz.sum((0, 2)), 1e-6) if ok.any() else np.array([np.nan])
    falls = fell_step[sel] < steps
    out = {
        "envs": int(sel.sum()),
        "fell_pct": 100 * float(falls.mean()),
        "fall_time_s_median": float(np.median(t[fell_step[sel][falls]])) if falls.any() else None,
        "estop_pct": 100 * float((estop_step[sel] < steps).mean()),
        "ankle_floor_pct": 100 * float(floor_hold[sel].mean()),
        "worst_dq_rms_median": float(np.median(win.max(0)[sel])),
        "worst_dq_rms_p90": float(np.percentile(win.max(0)[sel], 90)),
        "worst_freq_hz_median": float(np.median(freq[sel])),
        "target_step_rms_p90": float(np.percentile(np.where(alive[1:], target_step, 0)[:, sel].max(0), 90)),
        "stand_knee_q": [float(x) for x in A["q_knee"][s0:s1, ok].mean((0, 1))] if ok.any() else None,
        "stand_knee_tau": [float(x) for x in A["tau_knee"][s0:s1, ok].mean((0, 1))] if ok.any() else None,
        "stand_left_load_share_median": float(np.median(share_left)),
        "stand_load_imbalance_median": float(np.median(np.abs(share_left - 0.5) * 2)),
        "stand_ankle_target": [float(x) for x in A["target"][s0:s1][:, ok][..., ank].mean((0, 1))] if ok.any() else None,
        "stand_roll_deg": float(np.degrees(A["roll"][s0:s1, ok].mean())) if ok.any() else None,
    }
    if gyro is not None:
        out["gyro_inj_raw_rms"] = float(np.median(gyro["raw"][sel]))
        out["gyro_inj_filtered_rms"] = float(np.median(gyro["filt"][sel]))
        out["gyro_inj_filtered_peak"] = float(np.median(gyro["peak"][sel]))
    return out


results = {"label": args.label, "export": os.path.abspath(args.export), "imu": imu_frame, "gyro_lowpass_hz": lowpass_hz,
           "envs_per_cell": E, "seconds": args.seconds, "conditions": {c: CONDITIONS[c] for c in names}, "cells": {}}
idx = np.arange(N)
for i, (c, p) in enumerate(cells):
    sel = (idx >= i * E) & (idx < (i + 1) * E)
    results["cells"][f"{c}/{p}"] = cell_summary(sel)

out = args.out or os.path.join(REPO, "logs/hw_stress", args.label)
os.makedirs(out, exist_ok=True)
json.dump(results, open(os.path.join(out, "summary.json"), "w"), indent=1)
keep = np.concatenate([np.arange(i * E, i * E + min(4, E)) for i in range(len(cells))])
np.savez_compressed(os.path.join(out, "traces.npz"), t=t, cells=np.array([f"{c}/{p}" for c, p in cells]),
                    envs_per_cell=min(4, E), **{k: v[:, keep] for k, v in A.items()})

md = [f"# Stress test: {args.label}", "", f"`{os.path.abspath(args.export)}`, IMU {imu_frame}, low-pass {lowpass_hz},"
      f" {E} envs per cell, {args.seconds:g} s, deterministic networks, training sensor noise.", "",
      "| cell | fell % | fall t s | e-stop % | ankle floor % | worst dq rms (med / p90) | f Hz | knee q L/R | knee tau L/R |"
      " left load | imbalance | ankle tgt L/R | roll deg |", "|" + "---|" * 13]


def fmt(v, f="{:.2f}"):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "/".join(f.format(x) for x in v)
    return f.format(v)


for key, r in results["cells"].items():
    md.append(f"| {key} | {r['fell_pct']:.0f} | {fmt(r['fall_time_s_median'], '{:.1f}')} | {r['estop_pct']:.0f} |"
              f" {r['ankle_floor_pct']:.0f} | {r['worst_dq_rms_median']:.2f} / {r['worst_dq_rms_p90']:.2f} |"
              f" {r['worst_freq_hz_median']:.1f} | {fmt(r['stand_knee_q'])} | {fmt(r['stand_knee_tau'], '{:.0f}')} |"
              f" {fmt(r['stand_left_load_share_median'])} | {fmt(r['stand_load_imbalance_median'])} |"
              f" {fmt(r['stand_ankle_target'])} | {fmt(r['stand_roll_deg'], '{:.1f}')} |")
open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md), flush=True)
print(f"[stress] wrote {out}", flush=True)
sys.stdout.flush()
os._exit(0)
