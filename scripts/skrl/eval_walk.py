"""Does a LocoManip-Marl flat-curriculum policy pair walk, and are its moves slow enough to be safe?

Holds every env at one walking stage (no gate moves) with deterministic (mean) actions and the run's own
estimator and target table:

  --stage walk       navigation only, arms at the zero-joint-angle rest pose
  --stage alternate  walking and arm goals 50/50 with settles between, reach levels held at
                     --levels spread:drop

It measures velocity tracking by command type, falls (direction, when, in which mode), and the speeds
the user set as safe (wrist <= 0.5 m/s and pelvis up/down <= 0.3 m/s outside walking; joint speeds
against the rate caps), and films one robot.

Writes to <run>/eval_walk/<checkpoint>_<stage>/ (or --out):
    report.md         the numbers, with pass/fail against the stage-0 criteria
    tracking.png      velocity error and tracked-path ratio per command type
    speeds.png        joint speeds (99th percentile) against the caps, wrist and pelvis speeds per mode
    timeline.png      one robot: commanded vs actual speed and yaw rate, pelvis height, mode shading
    walk.mp4          (--video) that robot on camera
    walk_stills.png   (--video) six evenly spaced frames
    samples.npz       the raw samples

    python scripts/skrl/eval_walk.py --checkpoint <run>/checkpoints/agent_<N>.pt
    python scripts/skrl/eval_walk.py --checkpoint ... --stage alternate --levels 4:4
    python scripts/skrl/eval_walk.py --checkpoint ... --video --num_envs 16 --out <folder>
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Flat-Curriculum-Direct-v0")
parser.add_argument("--algorithm", default="mappo")
parser.add_argument("--stage", choices=("walk", "alternate"), default="walk")
parser.add_argument("--levels", default="4:4", help="spread:drop reach levels held in --stage alternate.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000, help="Policy steps; 50 per second.")
parser.add_argument("--out", default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_seconds", type=float, default=30.0)
parser.add_argument("--gui", action="store_true", help="Open the Isaac Sim window and run at real time.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import STANDING_PELVIS_HEIGHT  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.mdp.commands import ground_height  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

SAFE_WRIST = 0.5  # m/s, outside walking
SAFE_PELVIS_VZ = 0.3  # m/s, outside walking
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
NAME = os.path.splitext(os.path.basename(args.checkpoint))[0]
OUT = args.out or os.path.join(RUN_DIR, "eval_walk", f"{NAME}_{args.stage}")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- environment and policies
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, f"skrl_{args.algorithm}_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
# the stage is set by the arm-goal share alone (a run trained with the walking gate is evaluated the same way)
env_cfg.commands.arm_targets.walk_gate = False
env_cfg.commands.arm_targets.arm_goal_prob = 0.0 if args.stage == "walk" else 0.5
if args.video or args.gui:
    if args.num_envs > 20:
        print(f"[eval_walk] WARNING: {args.num_envs} envs share tiles and overlap on camera; use --num_envs 16 to watch.")
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (2.6, 2.0, 0.1)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.2)
    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.scene.imu.debug_vis = False
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None))
base = env.unwrapped
arm = base.command_manager.get_term("arm_targets")
vel = base.command_manager.get_term("base_velocity")
robot = base.scene["robot"]
scanner = base.scene.sensors["height_scanner"]
# hold the reach levels: no curriculum moves
STAGE = 0 if args.stage == "walk" else 1
spread, drop = (int(x) for x in args.levels.split(":"))


def _hold_gate(env_ids, fell=None):
    vel.seg_commanded[env_ids] = 0.0
    vel.seg_tracked[env_ids] = 0.0


arm._judge_nav_segments = _hold_gate
arm._record_outcomes = lambda reached, missed: None
arm.update_levels = lambda env_ids, fell: None

GROUPS = {
    "shoulders": ".*_shoulder_.*_joint", "elbows": ".*_elbow_joint", "wrists": ".*_wrist_.*_joint",
    "knee": ".*_knee_joint", "hip_pitch": ".*_hip_pitch_joint", "ankle_pitch": ".*_ankle_pitch_joint",
    "hip_roll_yaw": ".*_hip_(roll|yaw)_joint", "ankle_roll": ".*_ankle_roll_joint",
}
group_ids = {g: robot.find_joints(p)[0] for g, p in GROUPS.items()}
wrist_ids = robot.find_bodies(arm.cfg.body_names, preserve_order=True)[0]


def caps(term_name: str, navigation: bool) -> dict[str, float]:
    """The applied-target rate cap (rad/s) per joint name of an action term, in one mode."""
    term = base.action_manager.get_term(term_name)
    rates = getattr(term, "_rate_navigation" if navigation else "_rate", None)
    if rates is None:  # no rate caps on this term
        return {name: float("inf") for name in term._joint_names}
    return dict(zip(term._joint_names, rates.tolist()))


cap_arm_goal = {**caps("joint_pos", False), **caps("arm_pos", False)}
cap_walk = {**caps("joint_pos", True), **caps("arm_pos", True)}
group_cap = {
    mode: {g: max(c[robot.joint_names[i]] for i in ids) for g, ids in group_ids.items()}
    for mode, c in (("walking", cap_walk), ("arm goal / settle", cap_arm_goal))
}

runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


# ---------------------------------------------------------------- rollout
keys = ("mode", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "pelvis_h", "pelvis_vz", "wrist_speed") + tuple(
    f"qd_{g}" for g in GROUPS
)
samples = {k: [] for k in keys}
falls = {k: [] for k in ("mode", "tilt_x", "tilt_y", "episode_time")}
timeline = {k: [] for k in ("mode", "cmd_speed", "along", "cmd_wz", "wz", "pelvis_h", "fell")}
frames = []
totals = dict(reached=0.0, missed=0.0, falls=0)
video_steps = int(args.video_seconds / base.step_dt) if args.video else 0
print(f"EVAL {args.checkpoint}  stage {args.stage}  envs {args.num_envs}  steps {args.steps}  out {OUT}")
with torch.inference_mode():
    arm.spread_level[:] = spread
    arm.drop_level[:] = drop
    obs, _ = base.reset()
    states = base.state()
    for step in range(args.steps):
        start = time.time()
        # the mode a step runs in, and the robot before it (a fall resets the env within the step)
        mode0 = torch.where(arm.arm_mode, 2, torch.where(arm.settling, 1, 0))
        gravity0 = robot.data.projected_gravity_b.clone()
        episode_time0 = base.episode_length_buf.float() * base.step_dt
        reached0, missed0 = arm.metrics["goals_reached"].clone(), arm.metrics["goals_missed"].clone()
        obs, _, terminated, _, _ = base.step(act(obs, states))
        states = base.state()
        totals["reached"] += (arm.metrics["goals_reached"] - reached0).clamp(min=0).sum().item()
        totals["missed"] += (arm.metrics["goals_missed"] - missed0).clamp(min=0).sum().item()
        fell = terminated["legs"]
        totals["falls"] += int(fell.sum())
        if fell.any():
            falls["mode"].append(mode0[fell])
            falls["tilt_x"].append(gravity0[fell, 0])
            falls["tilt_y"].append(gravity0[fell, 1])
            falls["episode_time"].append(episode_time0[fell])
        ground = ground_height(scanner)
        pelvis_h = robot.data.root_pos_w[:, 2] - ground
        if step >= 100:
            ok = torch.isfinite(ground) & ~fell & (base.episode_length_buf > 25)
            cmd = vel.vel_command_b
            v = robot.data.root_lin_vel_b
            samples["mode"].append(mode0[ok])
            samples["cmd_vx"].append(cmd[ok, 0])
            samples["cmd_vy"].append(cmd[ok, 1])
            samples["cmd_wz"].append(cmd[ok, 2])
            samples["vx"].append(v[ok, 0])
            samples["vy"].append(v[ok, 1])
            samples["wz"].append(robot.data.root_ang_vel_b[ok, 2])
            samples["pelvis_h"].append(pelvis_h[ok])
            samples["pelvis_vz"].append(robot.data.root_lin_vel_w[ok, 2])
            samples["wrist_speed"].append(robot.data.body_lin_vel_w[ok][:, wrist_ids].norm(dim=-1).max(dim=1)[0])
            for g, ids in group_ids.items():
                samples[f"qd_{g}"].append(robot.data.joint_vel[ok][:, ids].abs().max(dim=1)[0])
        cmd0 = vel.vel_command_b[0]
        speed0 = cmd0[:2].norm().item()
        along0 = (robot.data.root_lin_vel_b[0, :2] * cmd0[:2]).sum().item() / max(speed0, 1e-6) if speed0 > 1e-3 else 0.0
        timeline["mode"].append(int(mode0[0]))
        timeline["cmd_speed"].append(speed0)
        timeline["along"].append(along0 if speed0 > 1e-3 else robot.data.root_lin_vel_b[0, :2].norm().item())
        timeline["cmd_wz"].append(cmd0[2].item())
        timeline["wz"].append(robot.data.root_ang_vel_b[0, 2].item())
        timeline["pelvis_h"].append(pelvis_h[0].item())
        timeline["fell"].append(bool(fell[0]))
        if step < video_steps:
            frame = base.render()
            if frame is not None:
                frames.append(frame)
        if args.gui:
            time.sleep(max(0.0, base.step_dt - (time.time() - start)))
        if not app.is_running():
            break

to_np = lambda xs: torch.cat(xs).float().cpu().numpy() if xs else np.zeros(0)  # noqa: E731
S = {k: to_np(v) for k, v in samples.items()}
F = {k: to_np(v) for k, v in falls.items()}
T = {k: np.array(v) for k, v in timeline.items()}
minutes = args.num_envs * args.steps * base.step_dt / 60.0
MODES = {0: "walking", 1: "settle", 2: "arm goal"}

# ---------------------------------------------------------------- tracking by command type
walking = S["mode"] == 0
cvx, cvy, cwz = S["cmd_vx"], S["cmd_vy"], S["cmd_wz"]
cspeed = np.hypot(cvx, cvy)
err_xy = np.hypot(cvx - S["vx"], cvy - S["vy"])
err_wz = np.abs(cwz - S["wz"])
along = np.where(cspeed > 1e-3, (S["vx"] * cvx + S["vy"] * cvy) / np.maximum(cspeed, 1e-6), 0.0)
classes = {
    "standing": walking & (cspeed < 0.05) & (np.abs(cwz) < 0.05),
    "forward": walking & (cvx >= 0.05) & (np.abs(cvx) >= np.abs(cvy)),
    "backward": walking & (cvx <= -0.05) & (np.abs(cvx) >= np.abs(cvy)),
    "sideways": walking & (np.abs(cvy) > np.abs(cvx)) & (cspeed >= 0.05),
    "turning only": walking & (cspeed < 0.05) & (np.abs(cwz) >= 0.05),
}
tracking = []
for name, m in classes.items():
    moving = m & (cspeed > 1e-3)
    ratio = (np.minimum(np.clip(along[moving], 0, None), cspeed[moving]).sum() / cspeed[moving].sum()) if moving.any() else np.nan
    tracking.append(dict(name=name, share=m.mean() / max(walking.mean(), 1e-9) if walking.any() else 0.0,
                         err_xy=err_xy[m].mean() if m.any() else np.nan,
                         err_wz=err_wz[m].mean() if m.any() else np.nan, ratio=ratio))
moving_all = walking & (cspeed > 1e-3)
overall_ratio = np.minimum(np.clip(along[moving_all], 0, None), cspeed[moving_all]).sum() / max(cspeed[moving_all].sum(), 1e-9)
overall_err = err_xy[walking].mean() if walking.any() else np.nan

# ---------------------------------------------------------------- speeds
speed_rows = []
for mode, label in ((0, "walking"), (1, "settle"), (2, "arm goal")):
    m = S["mode"] == mode
    if not m.any():
        continue
    row = dict(mode=label, n=int(m.sum()),
               wrist=np.percentile(S["wrist_speed"][m], [50, 95, 99]), pelvis=np.percentile(np.abs(S["pelvis_vz"][m]), [50, 95, 99]))
    for g in GROUPS:
        row[g] = np.percentile(S[f"qd_{g}"][m], 99)
    speed_rows.append(row)
not_walking = S["mode"] > 0
wrist99 = np.percentile(S["wrist_speed"][not_walking], 99) if not_walking.any() else np.nan
pelvis99 = np.percentile(np.abs(S["pelvis_vz"][not_walking]), 99) if not_walking.any() else np.nan
if args.stage == "walk":
    # walk-only has no settles or arm goals: the safe-speed check reads standing commands instead
    standing = classes["standing"]
    wrist99 = np.percentile(S["wrist_speed"][standing], 99) if standing.any() else np.nan
    pelvis99 = np.percentile(np.abs(S["pelvis_vz"][standing]), 99) if standing.any() else np.nan

# ---------------------------------------------------------------- plots
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a96"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True,
})

# 1. tracking per command type: two measures, two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
xs = np.arange(len(tracking))
ax1.bar(xs, [r["err_xy"] for r in tracking], width=0.55, color=SERIES[0])
ax1.axhline(0.25, color=REF, linewidth=1.0)
ax1.text(xs[-1] + 0.3, 0.25, "pass 0.25", color=INK_2, fontsize=8, va="bottom", ha="right")
ax1.set_title("Velocity error, xy (m/s)")
ax2.bar(xs, [100 * r["ratio"] for r in tracking], width=0.55, color=SERIES[0])
ax2.axhline(80, color=REF, linewidth=1.0)
ax2.text(xs[-1] + 0.3, 80, "pass 80%", color=INK_2, fontsize=8, va="bottom", ha="right")
ax2.set_ylim(0, 105)
ax2.set_title("Commanded path covered (%)")
for ax in (ax1, ax2):
    ax.set_xticks(xs, [r["name"] for r in tracking], fontsize=9)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tracking.png"), dpi=150)
plt.close(fig)

# 2. speeds: joint p99 against the caps (walking vs arm goal / settle), then wrist and pelvis per mode
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [2.2, 1, 1]})
groups = list(GROUPS)
xs = np.arange(len(groups))
rows_by = {r["mode"]: r for r in speed_rows}
for k, (label, color) in enumerate((("walking", SERIES[0]), ("arm goal", SERIES[1]))):
    if label in rows_by:
        ax1.bar(xs + (k - 0.5) * 0.36, [rows_by[label][g] for g in groups], width=0.34, color=color, label=label)
for k, (mode, color) in enumerate((("walking", SERIES[0]), ("arm goal / settle", SERIES[1]))):
    for x, g in zip(xs, groups):
        c = group_cap[mode][g]
        if not np.isfinite(c):
            continue
        ax1.plot([x + (k - 0.5) * 0.36 - 0.17, x + (k - 0.5) * 0.36 + 0.17], [c, c], color=INK, linewidth=1.2)
ax1.set_xticks(xs, groups, fontsize=8, rotation=20)
ax1.set_title("Joint speed, 99th percentile (rad/s)" + ("; black ticks: target-rate caps" if any(
    np.isfinite(c) for m in group_cap.values() for c in m.values()) else ""))
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="x", visible=False)
labels = [r["mode"] for r in speed_rows]
xm = np.arange(len(labels))
colors = {"walking": SERIES[0], "settle": SERIES[2], "arm goal": SERIES[1]}
ax2.bar(xm, [r["wrist"][2] for r in speed_rows], width=0.55, color=[colors[l] for l in labels])
ax2.axhline(SAFE_WRIST, color=REF, linewidth=1.0)
ax2.set_title("Wrist speed p99 (m/s)")
ax3.bar(xm, [r["pelvis"][2] for r in speed_rows], width=0.55, color=[colors[l] for l in labels])
ax3.axhline(SAFE_PELVIS_VZ, color=REF, linewidth=1.0)
ax3.set_title("Pelvis |v_z| p99 (m/s)")
for ax in (ax2, ax3):
    ax.set_xticks(xm, labels, fontsize=9)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "speeds.png"), dpi=150)
plt.close(fig)

# 3. timeline of env 0: shared time axis, mode shading
t = np.arange(len(T["mode"])) * base.step_dt
window = t <= 30.0
fig, axes = plt.subplots(3, 1, figsize=(10, 6.8), sharex=True)
axes[0].plot(t[window], T["cmd_speed"][window], color=REF, linewidth=1.5, label="commanded speed")
axes[0].plot(t[window], T["along"][window], color=SERIES[0], linewidth=1.5, label="speed along the command")
axes[0].set_ylabel("m/s")
axes[0].legend(fontsize=8, loc="upper right", ncol=2)
axes[1].plot(t[window], T["cmd_wz"][window], color=REF, linewidth=1.5, label="commanded")
axes[1].plot(t[window], T["wz"][window], color=SERIES[0], linewidth=1.5, label="actual")
axes[1].set_ylabel("yaw rate (rad/s)")
axes[1].legend(fontsize=8, loc="upper right", ncol=2)
axes[2].plot(t[window], T["pelvis_h"][window], color=SERIES[0], linewidth=1.5)
axes[2].axhline(STANDING_PELVIS_HEIGHT, color=REF, linewidth=1.0)
axes[2].set_ylabel("pelvis height (m)")
axes[2].set_xlabel("time (s)")
shade = {1: "#ececea", 2: "#dbe8f7"}
for ax in axes:
    for mode, color in shade.items():
        m = (T["mode"] == mode) & window
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, linewidth=0, zorder=0)
    for tf in t[window][T["fell"][window]]:
        ax.axvline(tf, color="#e34948", linewidth=1.2)
axes[0].set_title(f"One robot, stage {args.stage}: blue shading arm goals, gray settles, red lines falls", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timeline.png"), dpi=150)
plt.close(fig)

if frames:
    import imageio.v2 as imageio

    imageio.mimwrite(os.path.join(OUT, "walk.mp4"), frames, fps=int(round(1.0 / base.step_dt)), quality=8)
    picks = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 6.4))
    for ax, i in zip(axes.flat, picks):
        ax.imshow(frames[i])
        ax.axis("off")
        ax.set_title(f"t = {i * base.step_dt:.1f} s, {MODES[T['mode'][i]]}, pelvis {T['pelvis_h'][i]:.2f} m", fontsize=9)
    fig.suptitle(f"One robot, stage {args.stage}", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "walk_stills.png"), dpi=110)
    plt.close(fig)

np.savez_compressed(os.path.join(OUT, "samples.npz"), **{f"s_{k}": v for k, v in S.items()}, **{f"f_{k}": v for k, v in F.items()})

# ---------------------------------------------------------------- report
def pct(x):
    return f"{100 * x:.0f}%" if np.isfinite(x) else "-"


falls_per_min = totals["falls"] / minutes
finished = totals["reached"] + totals["missed"]
lines = [f"# Walking evaluation: `{args.checkpoint}`", "",
         f"Stage **{args.stage}**" + (f" (reach levels {spread}:{drop} held)" if STAGE else "") +
         f", {args.num_envs} envs, {args.steps} steps ({args.steps * base.step_dt:.0f} s), deterministic actions,"
         " the run's estimator, flat ground.", ""]
checks = [
    ("xy velocity error while walking <= 0.25 m/s", overall_err <= 0.25, f"{overall_err:.3f} m/s"),
    ("commanded path covered >= 80%", overall_ratio >= 0.8, pct(overall_ratio)),
    ("falls < 0.1 per env-minute", falls_per_min < 0.1, f"{falls_per_min:.2f}"),
    (f"wrist speed p99 <= {SAFE_WRIST} m/s " + ("(standing commands)" if not STAGE else "(arm goals, settles)"),
     wrist99 <= SAFE_WRIST, f"{wrist99:.2f} m/s"),
    (f"pelvis |v_z| p99 <= {SAFE_PELVIS_VZ} m/s " + ("(standing commands)" if not STAGE else "(arm goals, settles)"),
     pelvis99 <= SAFE_PELVIS_VZ, f"{pelvis99:.2f} m/s"),
]
lines += ["## Pass criteria", "", "| check | result | value |", "|---|---|---|"]
lines += [f"| {name} | {'pass' if ok else '**fail**'} | {value} |" for name, ok, value in checks]
lines += ["", "## Velocity tracking by command type (walking steps)", "",
          "| command | share of walking | xy error | yaw-rate error | path covered |", "|---|---|---|---|---|"]
for r in tracking:
    lines.append(f"| {r['name']} | {pct(r['share'])} | {r['err_xy']:.3f} m/s | {r['err_wz']:.3f} rad/s | {pct(r['ratio'])} |")
lines += ["", "## Speeds (99th percentile unless noted)", "",
          "| mode | samples | wrist p50 / p95 / p99 | pelvis abs v_z p50 / p95 / p99 | "
          + " | ".join(GROUPS) + " |", "|---" * (4 + len(GROUPS)) + "|"]
for r in speed_rows:
    lines.append(
        f"| {r['mode']} | {r['n']} | {r['wrist'][0]:.2f} / {r['wrist'][1]:.2f} / {r['wrist'][2]:.2f} m/s |"
        f" {r['pelvis'][0]:.2f} / {r['pelvis'][1]:.2f} / {r['pelvis'][2]:.2f} m/s | "
        + " | ".join(f"{r[g]:.1f}" for g in GROUPS) + " |"
    )
cap_cell = lambda c: f"{c:.1f}" if np.isfinite(c) else "none"  # noqa: E731
lines.append("| target-rate cap, walking | | | | " + " | ".join(cap_cell(group_cap["walking"][g]) for g in GROUPS) + " |")
lines.append("| target-rate cap, arm goal / settle | | | | "
             + " | ".join(cap_cell(group_cap["arm goal / settle"][g]) for g in GROUPS) + " |")
lines += ["", "Joint speeds are measured, not commanded: the cap limits the target, and the PD response, steps"
          " and impacts can briefly exceed it."]
lines += ["", "## Falls", ""]
if len(F["mode"]):
    fwd = (np.abs(F["tilt_x"]) >= np.abs(F["tilt_y"])) & (F["tilt_x"] > 0)
    bwd = (np.abs(F["tilt_x"]) >= np.abs(F["tilt_y"])) & (F["tilt_x"] <= 0)
    lines += [
        f"- {totals['falls']} falls, {falls_per_min:.2f} per env-minute. Direction: forward {pct(fwd.mean())},"
        f" backward {pct(bwd.mean())}, sideways {pct((~fwd & ~bwd).mean())}.",
        "- In mode: " + ", ".join(f"{MODES[m]} {pct((F['mode'] == m).mean())}" for m in MODES) + ".",
        f"- Within 1 s of an episode start: {pct((F['episode_time'] < 1.0).mean())}.",
    ]
else:
    lines.append("- No falls.")
if STAGE:
    lines += ["", "## Manipulation", "", f"- Arm goals reached: {pct(totals['reached'] / finished) if finished else '-'}"
              f" of {int(finished)} ended goals; time in each mode: "
              + ", ".join(f"{MODES[m]} {pct((S['mode'] == m).mean())}" for m in MODES) + "."]
lines += ["", "Figures: `tracking.png`, `speeds.png`, `timeline.png`" + (", video `walk.mp4`, stills `walk_stills.png`." if frames else ".")]
report = "\n".join(lines)
with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write(report + "\n")
print("\n" + report)
print(f"\n[eval_walk] wrote {OUT}")
sys.stdout.flush()
os._exit(0)
