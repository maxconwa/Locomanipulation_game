"""Does a trained LocoManip-Marl policy pair crouch for low arm targets? Plots, a report, and a video.

Every env gets arm goals only, with the arm curriculum held at fixed
(spread, drop) level pairs, deterministic (mean) actions, and the run's own
estimator and target table. For each pair it measures how far the pelvis
drops against how far the targets were lowered (the crouch), how the legs and
pelvis pose change, how many goals are reached, and how often the robot falls.

Writes to <run>/eval_crouch/<checkpoint>/ (or --out):
    report.md                 the numbers that matter, with a verdict per question
    crouch_response.png       target drop vs pelvis drop, per spread level
    posture.png               knee / hip / ankle / pelvis pitch vs target drop
    outcomes.png              goals reached (%) and falls per env-minute, per pair
    reach_by_height.png       goals reached and pelvis drop by how high the lower target is above the ground
    timeline.png              one robot through a run of goals at the --focus pair
    rewards_by_crouch.md/.png every legs and arms reward term, by how low the pelvis is and how it moves
    crouch.mp4                (--video) that robot on camera, goal frames visible
    crouch_stills.png, stills (--video) its deepest crouches and the moments before its falls
    samples.npz               the raw samples behind the plots

Use many envs for the numbers and few for anything you watch: with more than
~20 envs several robots share each terrain tile and overlap on camera.

    # numbers and plots
    python scripts/skrl/eval_crouch.py --checkpoint <run>/checkpoints/agent_<N>.pt
    # a video of one robot (headless, needs cameras), into its own folder
    python scripts/skrl/eval_crouch.py --checkpoint ... --video --num_envs 16 --grid 4:10 --out <folder>
    # live in the Isaac Sim window, real time
    python scripts/skrl/eval_crouch.py --checkpoint ... --gui --num_envs 16 --grid 4:10 --steps 3000
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True, help="skrl agent checkpoint; its run's estimator and target table are used.")
parser.add_argument("--task", default="LocoManip-Marl-Stand-Direct-v0")
parser.add_argument("--algorithm", default="mappo", help="skrl config the checkpoint was trained with: ippo or mappo.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500, help="Policy steps per (spread, drop) pair; 50 per second.")
parser.add_argument("--grid", default="4:0,4:2,4:4,4:5,4:6,4:8,4:10,8:0,8:5,8:10", help="spread:drop level pairs to hold.")
parser.add_argument("--focus", default=None, help="spread:drop pair for the timeline and video; default: the grid's deepest drop.")
parser.add_argument("--out", default=None, help="Output folder; default <run>/eval_crouch/<checkpoint name>.")
parser.add_argument("--video", action="store_true", help="Record crouch.mp4 of env 0 at the focus pair (enables cameras).")
parser.add_argument("--video_seconds", type=float, default=20.0)
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

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import STANDING_PELVIS_HEIGHT  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.mdp.commands import ground_height  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

PAIRS = [tuple(int(x) for x in pair.split(":")) for pair in args.grid.split(",")]
FOCUS = tuple(int(x) for x in args.focus.split(":")) if args.focus else max(PAIRS, key=lambda p: (p[1], -p[0]))
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
OUT = args.out or os.path.join(RUN_DIR, "eval_crouch", os.path.splitext(os.path.basename(args.checkpoint))[0])
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- environment and policies
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, f"skrl_{args.algorithm}_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR  # its saved target table and estimator (with curriculum state) are reused
env_cfg.commands.arm_targets.arm_goal_prob = 1.0
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
if args.video or args.gui:
    if args.num_envs > 20:
        print(f"[eval_crouch] WARNING: {args.num_envs} envs share terrain tiles and overlap on camera; use --num_envs 16 to watch.")
    # follow env 0's robot from the front-left, framing feet to hands
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (2.6, 2.0, 0.1)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.2)
    # keep the arm goal frames on screen, drop the rest of the debug drawing
    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.scene.imu.debug_vis = False
    env_cfg.commands.base_velocity.debug_vis = False
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None))
base = env.unwrapped
arm = base.command_manager.get_term("arm_targets")
robot = base.scene["robot"]
scanner = base.scene.sensors["height_scanner"]
# hold each test level exactly: no curriculum moves
arm._record_outcomes = lambda reached, missed: None
arm.update_levels = lambda env_ids, fell: None

JOINTS = {
    "knee": ["left_knee_joint", "right_knee_joint"],
    "hip_pitch": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    "ankle_pitch": ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
}
joint_ids = {name: robot.find_joints(names, preserve_order=True)[0] for name, names in JOINTS.items()}
soft_limits = {name: robot.data.soft_joint_pos_limits[0, ids].mean(dim=0).tolist() for name, ids in joint_ids.items()}
# the lowest wrist target reachable standing, above the ground: below it a goal needs a crouch
STANDING_REACH = arm._standing_min_z.min().item() + env_cfg.commands.arm_targets.standing_height

runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


def pelvis_state():
    """(pelvis drop below standing height, valid) per env, in m."""
    ground = ground_height(scanner)
    valid = torch.isfinite(ground)
    return STANDING_PELVIS_HEIGHT - (robot.data.root_pos_w[:, 2] - ground), valid


# ---------------------------------------------------------------- one (spread, drop) pair
def evaluate(spread: int, drop: int, focus: bool) -> dict:
    arm.spread_level[:] = spread
    arm.drop_level[:] = drop
    obs, _ = base.reset()
    states = base.state()
    totals = dict(reached=0.0, missed=0.0, falls=0.0)
    samples = {k: [] for k in ("target_drop", "pelvis_drop", "knee", "hip_pitch", "ankle_pitch", "pelvis_pitch", "pos_err", "rot_err",
                               "pelvis_vz", "target_height", "rew_legs", "rew_arms")}
    goals = {k: [] for k in ("height", "reached", "needs_crouch")}  # one entry per goal that ended
    falls = {k: [] for k in ("tilt_x", "tilt_y", "pelvis_drop", "target_height", "needs_crouch", "episode_time")}  # one per fall
    timeline = {k: [] for k in ("target_drop", "target_height", "pelvis_drop", "knee", "pos_err", "reached", "fell")}
    frames = []
    video_steps = int(args.video_seconds / base.step_dt) if (focus and args.video) else 0
    for step in range(args.steps):
        start = time.time()
        reached0 = arm.metrics["goals_reached"].clone()
        missed0 = arm.metrics["goals_missed"].clone()
        # the goal a step ends is replaced within that step: read its targets first
        height0, needs_crouch0 = arm.lowest_target_height.clone(), arm.needs_crouch.clone()
        # and the robot's last state before a fall: the step resets a fallen env
        gravity0, pelvis_drop0 = robot.data.projected_gravity_b.clone(), pelvis_state()[0]
        episode_time0 = base.episode_length_buf.clone().float() * base.step_dt
        obs, _, terminated, _, _ = base.step(act(obs, states))
        states = base.state()
        reached = (arm.metrics["goals_reached"] - reached0).clamp(min=0)
        # per-episode counters; a reset zeroes them, so clamp that step's difference
        missed = (arm.metrics["goals_missed"] - missed0).clamp(min=0)
        totals["reached"] += reached.sum().item()
        totals["missed"] += missed.sum().item()
        ended = (reached > 0) | (missed > 0)
        if step >= 100 and ended.any():
            goals["height"].append(height0[ended])
            goals["reached"].append(reached[ended] > 0)
            goals["needs_crouch"].append(needs_crouch0[ended])
        totals["falls"] += terminated["legs"].float().sum().item()
        fell = terminated["legs"]
        if fell.any():
            # gravity in the pelvis frame: +x when pitched nose-down (toppling forward), +y when tipped left
            falls["tilt_x"].append(gravity0[fell, 0])
            falls["tilt_y"].append(gravity0[fell, 1])
            falls["pelvis_drop"].append(pelvis_drop0[fell])
            falls["target_height"].append(height0[fell])
            falls["needs_crouch"].append(needs_crouch0[fell])
            falls["episode_time"].append(episode_time0[fell])
        pos_err, rot_err = arm.errors()
        pelvis_drop, valid = pelvis_state()

        if step >= 100 and step % 5 == 0:  # past the reset transient
            ok = valid
            samples["target_drop"].append(arm.height_drop[ok])
            samples["pelvis_drop"].append(pelvis_drop[ok])
            for name, ids in joint_ids.items():
                samples[name].append(robot.data.joint_pos[ok][:, ids].mean(dim=1))
            samples["pelvis_pitch"].append(euler_xyz_from_quat(robot.data.root_quat_w[ok])[1])
            samples["pos_err"].append(pos_err[ok].mean(dim=1))
            samples["rot_err"].append(rot_err[ok].mean(dim=1))
            samples["pelvis_vz"].append(robot.data.root_lin_vel_w[ok][:, 2])
            samples["target_height"].append(arm.lowest_target_height[ok])
            # each term's weighted reward this step, per second (RewardManager._step_reward)
            samples["rew_legs"].append(base.reward_managers["legs"]._step_reward[ok].clone())
            samples["rew_arms"].append(base.reward_managers["arms"]._step_reward[ok].clone())
        if focus:
            timeline["target_drop"].append(arm.height_drop[0].item())
            timeline["target_height"].append(arm.lowest_target_height[0].item())
            timeline["pelvis_drop"].append(pelvis_drop[0].item())
            timeline["knee"].append(robot.data.joint_pos[0, joint_ids["knee"]].mean().item())
            timeline["pos_err"].append(pos_err[0].mean().item())
            timeline["reached"].append(reached[0].item() > 0)
            timeline["fell"].append(bool(terminated["legs"][0].item()))
            if step < video_steps:
                frame = base.render()
                if frame is not None:
                    frames.append(frame)
        if args.gui:
            time.sleep(max(0.0, base.step_dt - (time.time() - start)))

    to_np = lambda xs: torch.cat(xs).float().cpu().numpy()  # noqa: E731
    data = {k: to_np(v) for k, v in samples.items()}
    data.update({f"goal_{k}": to_np(v) if v else np.zeros(0) for k, v in goals.items()})
    data.update({f"fall_{k}": to_np(v) if v else np.zeros(0) for k, v in falls.items()})
    minutes = args.num_envs * args.steps * base.step_dt / 60.0
    finished = totals["reached"] + totals["missed"]
    cmd, act_drop = data["target_drop"], data["pelvis_drop"]
    var = np.var(cmd)
    slope = float(np.cov(cmd, act_drop, bias=True)[0, 1] / var) if var > 1e-8 else float("nan")
    r2 = float(np.corrcoef(cmd, act_drop)[0, 1] ** 2) if var > 1e-8 else float("nan")
    result = dict(
        spread=spread, drop=drop, data=data, slope=slope, r2=r2,
        success=totals["reached"] / finished if finished else float("nan"),
        reached_per_min=totals["reached"] / minutes, falls_per_min=totals["falls"] / minutes,
        timeline={k: np.array(v) for k, v in timeline.items()} if focus else None, frames=frames,
    )
    needs_crouch = data["goal_needs_crouch"] > 0
    result["crouch_goals"] = int(needs_crouch.sum())
    result["crouch_success"] = float(data["goal_reached"][needs_crouch].mean()) if needs_crouch.any() else float("nan")
    print(
        f"EVAL {spread:3d}:{drop:<3d} | success {100 * result['success']:5.1f}%"
        f" (needing a crouch {100 * result['crouch_success']:5.1f}% of {result['crouch_goals']}) |"
        f" falls/env-min {result['falls_per_min']:.2f} |"
        f" target drop {100 * cmd.mean():4.1f} cm -> pelvis drop {100 * act_drop.mean():4.1f} cm, slope {slope:5.2f} |"
        f" knee {data['knee'].mean():.2f} rad | err {100 * data['pos_err'].mean():.1f} cm {data['rot_err'].mean():.2f} rad"
    )
    sys.stdout.flush()
    return result


print(f"EVAL {args.checkpoint}  envs {args.num_envs}  steps {args.steps}  focus {FOCUS}  out {OUT}")
results = []
with torch.inference_mode():  # one block, resets included: every env buffer is then an inference tensor
    for spread, drop in PAIRS:
        results.append(evaluate(spread, drop, focus=(spread, drop) == FOCUS))
        if not app.is_running():
            break

# ---------------------------------------------------------------- plots
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill): categorical slots in fixed order, recessive chrome
SURFACE, INK, INK_2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a96"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True,
})
spreads = sorted({r["spread"] for r in results})
color_of = {s: SERIES[i % len(SERIES)] for i, s in enumerate(spreads)}  # color follows the spread, not its rank
BINS = np.arange(0.0, env_cfg.commands.arm_targets.max_height_drop + 0.011, 0.02)


def binned(x, y):
    idx = np.digitize(x, BINS) - 1
    centres, means, lo, hi = [], [], [], []
    for b in range(len(BINS) - 1):
        sel = idx == b
        if sel.sum() >= 50:
            centres.append(0.5 * (BINS[b] + BINS[b + 1]))
            means.append(np.mean(y[sel]))
            lo.append(np.percentile(y[sel], 25))
            hi.append(np.percentile(y[sel], 75))
    return map(np.array, (centres, means, lo, hi))


def pooled(spread, key):
    rs = [r for r in results if r["spread"] == spread]
    return np.concatenate([r["data"]["target_drop"] for r in rs]), np.concatenate([r["data"][key] for r in rs])


def fit(x, y):
    var = np.var(x)
    if var < 1e-8:
        return float("nan"), float(np.mean(y))
    slope = float(np.cov(x, y, bias=True)[0, 1] / var)
    return slope, float(np.mean(y) - slope * np.mean(x))


# 1. crouch response: the headline
fig, ax = plt.subplots(figsize=(7.2, 4.8))
lim = 100 * (BINS[-1])
ax.plot([0, lim], [0, lim], color=REF, linewidth=1.0, zorder=1)
ax.text(lim * 0.97, lim * 0.97, "pelvis drops as far\nas the targets", color=INK_2, fontsize=8, ha="right", va="top")
summary = {}
for k, s in enumerate(spreads):
    x, y = pooled(s, "pelvis_drop")
    c, m, lo, hi = binned(x, y)
    slope, icpt = fit(x, y)
    summary[s] = (slope, icpt)
    ax.fill_between(100 * c, 100 * lo, 100 * hi, color=color_of[s], alpha=0.10, linewidth=0)
    ax.plot(100 * c, 100 * m, color=color_of[s], linewidth=1.5, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"spread level {s}", zorder=3)
    if len(c):
        # stacked by spread, so lines that end together don't print over each other
        ax.annotate(f"slope {slope:.2f}", (100 * c[-1], 100 * m[-1]), xytext=(6, 7 - 14 * k), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
best = max(summary.values(), key=lambda v: v[0] if not np.isnan(v[0]) else -1)[0]
ax.set_title(f"The pelvis follows lowered targets {best:.2f} cm per cm" if not np.isnan(best) else "Pelvis drop vs target drop")
ax.set_xlabel("target drop below the standing workspace (cm)")
ax.set_ylabel("pelvis drop below standing height (cm)")
ax.set_xlim(0, lim * 1.15)
ax.set_ylim(min(0, ax.get_ylim()[0]), lim)
ax.legend(loc="upper left")
fig.text(0.01, 0.01, "points: binned mean; band: interquartile range", color=INK_2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "crouch_response.png"), dpi=150)
plt.close(fig)

# 2. posture: small multiples, one measure each (never two y-scales)
panels = [("knee", "knee flexion (rad)"), ("hip_pitch", "hip pitch (rad)"), ("ankle_pitch", "ankle pitch (rad)"),
          ("pelvis_pitch", "pelvis pitch (rad)")]
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharex=True)
for ax, (key, label) in zip(axes, panels):
    for s in spreads:
        x, y = pooled(s, key)
        c, m, lo, hi = binned(x, y)
        ax.fill_between(100 * c, lo, hi, color=color_of[s], alpha=0.10, linewidth=0)
        ax.plot(100 * c, m, color=color_of[s], linewidth=1.5, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=f"spread {s}")
    if key in soft_limits:
        for bound in soft_limits[key]:
            if ax.get_ylim()[0] - 0.3 < bound < ax.get_ylim()[1] + 0.3:
                ax.axhline(bound, color=REF, linewidth=1.0)
                ax.text(ax.get_xlim()[0], bound, " soft limit", color=INK_2, fontsize=7, va="bottom")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("target drop (cm)")
axes[0].legend(loc="best", fontsize=8)
fig.suptitle("How the legs get the pelvis down", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "posture.png"), dpi=150)
plt.close(fig)

# 3. outcomes per pair: two measures, two panels
labels = [f"{r['spread']}:{r['drop']}" for r in results]
colors = [color_of[r["spread"]] for r in results]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
xs = np.arange(len(results))
ax1.bar(xs, [100 * r["success"] for r in results], width=0.55, color=colors)
ax1.set_title("Arm goals reached (%)")
ax1.set_ylim(0, 100)
ax2.bar(xs, [r["falls_per_min"] for r in results], width=0.55, color=colors)
ax2.set_title("Falls per env-minute")
for ax in (ax1, ax2):
    ax.set_xticks(xs, labels)
    ax.set_xlabel("spread : drop level")
    ax.grid(axis="x", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=color_of[s]) for s in spreads]
ax1.legend(handles, [f"spread {s}" for s in spreads], loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "outcomes.png"), dpi=150)
plt.close(fig)

# 3b. by target height above the ground, all pairs pooled: is a goal that needs a crouch reached?
HEIGHT_EDGES = [np.inf, STANDING_REACH, 0.75, 0.60, 0.45, -np.inf]  # descending: lower targets to the right
height_labels = [f">= {STANDING_REACH:.2f} m\n(standing reach)"] + [
    f"{lo:.2f}-{hi:.2f} m" for hi, lo in zip(HEIGHT_EDGES[1:-2], HEIGHT_EDGES[2:-1])
] + [f"< {HEIGHT_EDGES[-2]:.2f} m"]
g_height = np.concatenate([r["data"]["goal_height"] for r in results])
g_reached = np.concatenate([r["data"]["goal_reached"] for r in results])
s_height = np.concatenate([r["data"]["target_height"] for r in results])
s_drop = np.concatenate([r["data"]["pelvis_drop"] for r in results])
s_err = np.concatenate([r["data"]["pos_err"] for r in results])
by_height = []
for hi, lo, label in zip(HEIGHT_EDGES[:-1], HEIGHT_EDGES[1:], height_labels):
    g, s = (g_height < hi) & (g_height >= lo), (s_height < hi) & (s_height >= lo)
    by_height.append(dict(
        label=label, goals=int(g.sum()), success=float(g_reached[g].mean()) if g.any() else float("nan"),
        pelvis_drop=float(s_drop[s].mean()) if s.any() else float("nan"),
        pos_err=float(s_err[s].mean()) if s.any() else float("nan"),
    ))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0))
xs = np.arange(len(by_height))
ax1.bar(xs, [100 * b["success"] for b in by_height], width=0.55, color=SERIES[0])
for x, b in zip(xs, by_height):
    if b["goals"]:
        ax1.text(x, 100 * b["success"] + 1.5, f"{100 * b['success']:.0f}%\nn={b['goals']}", ha="center", fontsize=8, color=INK_2)
ax1.set_ylim(0, 110)
ax1.set_title("Arm goals reached (%)")
ax2.bar(xs, [100 * b["pelvis_drop"] for b in by_height], width=0.55, color=SERIES[0])
ax2.set_title("Pelvis drop below standing (cm)")
for ax in (ax1, ax2):
    ax.set_xticks(xs, [b["label"] for b in by_height], fontsize=8)
    ax.set_xlabel("lower wrist target above the ground")
    ax.grid(axis="x", visible=False)
fig.suptitle("Targets below the standing reach can only be reached by crouching", x=0.01, ha="left",
             fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reach_by_height.png"), dpi=150)
plt.close(fig)

# 4. timeline of one robot at the focus pair: shared time axis, one measure per panel
focus_result = next((r for r in results if (r["spread"], r["drop"]) == FOCUS), None)
if focus_result is not None:
    tl = focus_result["timeline"]
    t = np.arange(len(tl["pelvis_drop"])) * base.step_dt
    window = t <= 30.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    axes[0].plot(t[window], 100 * tl["target_drop"][window], color=REF, linewidth=1.5, label="target drop")
    axes[0].plot(t[window], 100 * tl["pelvis_drop"][window], color=SERIES[0], linewidth=1.5, label="pelvis drop")
    axes[0].set_ylabel("cm below standing")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].plot(t[window], tl["knee"][window], color=SERIES[0], linewidth=1.5)
    axes[1].set_ylabel("knee (rad)")
    axes[2].plot(t[window], 100 * tl["pos_err"][window], color=SERIES[0], linewidth=1.5)
    axes[2].axhline(100 * env_cfg.commands.arm_targets.reach_pos_tol, color=REF, linewidth=1.0)
    axes[2].text(0, 100 * env_cfg.commands.arm_targets.reach_pos_tol, " reach tolerance", color=INK_2, fontsize=7, va="bottom")
    axes[2].set_ylabel("wrist error (cm)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        for tr in t[window][tl["reached"][window]]:
            ax.axvline(tr, color=SERIES[2], linewidth=0.8, alpha=0.6)
        for tf in t[window][tl["fell"][window]]:
            ax.axvline(tf, color="#e34948", linewidth=1.2)
    axes[0].set_title(
        f"One robot at spread {FOCUS[0]}, drop {FOCUS[1]}: green lines are reached goals, red a fall", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "timeline.png"), dpi=150)
    plt.close(fig)
    if focus_result["frames"]:
        import imageio.v2 as imageio

        fps = int(round(1.0 / base.step_dt))
        imageio.mimwrite(os.path.join(OUT, "crouch.mp4"), focus_result["frames"], fps=fps, quality=8)

        # stills: the deepest crouch moments (at least 1 s apart), and the moment before each fall
        n = len(focus_result["frames"])
        depth = tl["pelvis_drop"][:n]
        falls_at = np.nonzero(tl["fell"][:n])[0]
        # a squat, not a fall: no frame in the 2 s before a fall (a fall can start as a sideways split a second
        # earlier), nor deeper than the legs can fold (0.415 m with the feet flat at the hard knee and ankle
        # limits; a little slack for the scanned ground)
        squatting = depth < 0.45
        for i in falls_at:
            squatting[max(i - int(2.0 / base.step_dt), 0): i + 1] = False
        picks = []
        for i in np.argsort(-np.where(squatting, depth, -np.inf)):
            if not squatting[i]:
                break
            if len(picks) == 6:
                break
            if all(abs(i - j) >= int(1.0 / base.step_dt) for j, _ in picks):
                picks.append((int(i), "deepest"))
        picks += [(max(int(i) - int(0.3 / base.step_dt), 0), "before a fall") for i in falls_at[:3]]
        picks.sort()
        stills_dir = os.path.join(OUT, "stills")
        os.makedirs(stills_dir, exist_ok=True)
        for i, kind in picks:
            imageio.imwrite(os.path.join(stills_dir, f"t{i * base.step_dt:05.2f}s_{kind.replace(' ', '_')}.png"),
                            focus_result["frames"][i])
        cols = 3
        rows = (len(picks) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (i, kind) in zip(axes.flat, picks):
            ax.imshow(focus_result["frames"][i])
            ax.set_title(f"t = {i * base.step_dt:.1f} s, pelvis {100 * depth[i]:.0f} cm down,"
                         f" target {tl['target_height'][i]:.2f} m up" + (" (before a fall)" if kind != "deepest" else ""),
                         fontsize=9, color="#e34948" if kind != "deepest" else INK)
        fig.suptitle(f"One robot at spread {FOCUS[0]}, drop {FOCUS[1]}: its deepest crouches", x=0.01, ha="left",
                     fontweight="bold", color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "crouch_stills.png"), dpi=110)
        plt.close(fig)

# 5. rewards by crouch depth and motion: every term, per second
BUCKETS = [("standing (< 3 cm)", -1.0, 0.03), ("shallow (3-8 cm)", 0.03, 0.08), ("mid (8-13 cm)", 0.08, 0.13),
           ("deep (13-25 cm)", 0.13, 0.25), ("very deep (> 25 cm)", 0.25, 2.0)]
pool = lambda key: np.concatenate([r["data"][key] for r in results])  # noqa: E731
p_drop, p_vz = pool("pelvis_drop"), pool("pelvis_vz")
masks = [(name, (p_drop >= lo) & (p_drop < hi)) for name, lo, hi in BUCKETS]
masks += [("lowering (v_z < -0.1)", p_vz < -0.1), ("holding (|v_z| < 0.05)", np.abs(p_vz) < 0.05), ("rising (v_z > 0.1)", p_vz > 0.1)]
reward_lines = ["# Rewards during a crouch", "",
                "Each term's weighted reward per second (what the reward manager adds, before the per-agent clip),"
                " averaged over samples grouped by how far the pelvis is below standing height, and by its vertical"
                " velocity. Arm goals only, all pairs pooled.", ""]
deltas = {}
for agent in ("legs", "arms"):
    names = base.reward_managers[agent].active_terms
    rew = pool(f"rew_{agent}")
    header = "| term | " + " | ".join(n for n, _ in masks) + " |"
    reward_lines += [f"## {agent}", "", header, "|---" * (len(masks) + 1) + "|",
                     "| samples | " + " | ".join(f"{int(m.sum())}" for _, m in masks) + " |"]
    stand, deep = masks[0][1], masks[3][1]
    order = sorted(range(len(names)), key=lambda i: (rew[deep, i].mean() - rew[stand, i].mean()) if deep.any() and stand.any() else 0)
    for i in order:
        cells = [f"{rew[m, i].mean():+.3f}" if m.any() else "-" for _, m in masks]
        reward_lines.append(f"| {names[i]} | " + " | ".join(cells) + " |")
    totals = [f"**{rew[m].sum(axis=1).mean():+.3f}**" if m.any() else "-" for _, m in masks]
    reward_lines += ["| **total** | " + " | ".join(totals) + " |", ""]
    if deep.any() and stand.any():
        deltas[agent] = {names[i]: rew[deep, i].mean() - rew[stand, i].mean() for i in range(len(names))}
with open(os.path.join(OUT, "rewards_by_crouch.md"), "w") as f:
    f.write("\n".join(reward_lines) + "\n")
if "legs" in deltas:
    # what a deep crouch gains or costs the legs, term by term: diverging blue (gain) / red (cost)
    items = sorted(deltas["legs"].items(), key=lambda kv: kv[1])
    items = [kv for kv in items if abs(kv[1]) >= 0.005]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1.4))
    ys = np.arange(len(items))
    ax.barh(ys, [v for _, v in items], height=0.6, color=["#2a78d6" if v > 0 else "#e34948" for _, v in items])
    ax.set_yticks(ys, [k for k, _ in items])
    ax.axvline(0, color=REF, linewidth=1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("reward per second: deep crouch (13-25 cm) minus standing (< 3 cm)")
    ax.set_title("What a deep crouch gains (blue) or costs (red) the legs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rewards_by_crouch.png"), dpi=150)
    plt.close(fig)
print("\n".join(reward_lines))

np.savez_compressed(
    os.path.join(OUT, "samples.npz"),
    **{f"{r['spread']}_{r['drop']}_{k}": v for r in results for k, v in r["data"].items()},
)

# ---------------------------------------------------------------- report
def pct(x):
    return f"{100 * x:.0f}%"


lines = [f"# Crouch evaluation: `{args.checkpoint}`", "",
         f"{args.num_envs} envs, {args.steps} steps ({args.steps * base.step_dt:.0f} s) per pair, arm goals only,"
         " deterministic actions, the run's estimator. Terrain levels as saved with the run.", ""]
lines += ["## Does the pelvis follow the targets down?", ""]
for s in spreads:
    slope, icpt = summary[s]
    deepest = max((r for r in results if r["spread"] == s), key=lambda r: r["drop"])
    tgt, pel = deepest["data"]["target_drop"].mean(), deepest["data"]["pelvis_drop"].mean()
    zero = next((r for r in results if r["spread"] == s and r["drop"] == 0), None)
    base_drop = zero["data"]["pelvis_drop"].mean() if zero else icpt
    verdict = "yes" if slope >= 0.6 else "partly" if slope >= 0.15 else "no"
    lines.append(
        f"- **Spread {s}: {verdict}**, slope {slope:.2f}. At drop level {deepest['drop']} the targets sit"
        f" {100 * tgt:.1f} cm lower on average and the pelvis {100 * pel:.1f} cm lower; with no drop it already sits"
        f" {100 * base_drop:.1f} cm low. So the legs take {pct(max(pel - base_drop, 0) / tgt) if tgt > 0 else 'n/a'}"
        f" of the extra drop and the arms reach for the rest."
    )
lines += ["", "## Can the robot do the task, and does it stay up?", "",
          "| spread:drop | goals reached | needing a crouch: reached (goals) | falls / env-min | wrist error | rotation error |",
          "|---|---|---|---|---|---|"]
for r in results:
    crouch = f"{pct(r['crouch_success'])} ({r['crouch_goals']})" if r["crouch_goals"] else "-"
    lines.append(
        f"| {r['spread']}:{r['drop']} | {pct(r['success'])} | {crouch} | {r['falls_per_min']:.2f} |"
        f" {100 * r['data']['pos_err'].mean():.1f} cm | {r['data']['rot_err'].mean():.2f} rad |"
    )
worst_falls = max(r["falls_per_min"] for r in results)
lines += ["", f"- Falls: worst pair {worst_falls:.2f} per env-minute"
          + (" (**unstable**: a robot falls every couple of minutes or sooner)" if worst_falls > 0.5 else " (stable)") + "."]
fall = {k: np.concatenate([r["data"][f"fall_{k}"] for r in results]) for k in
        ("tilt_x", "tilt_y", "pelvis_drop", "target_height", "needs_crouch", "episode_time")}
lines += ["", "## Falls: which way, and when", ""]
if len(fall["tilt_x"]):
    forward = (np.abs(fall["tilt_x"]) >= np.abs(fall["tilt_y"])) & (fall["tilt_x"] > 0)
    backward = (np.abs(fall["tilt_x"]) >= np.abs(fall["tilt_y"])) & (fall["tilt_x"] <= 0)
    sideways = ~(forward | backward)
    lines += [
        f"- {len(fall['tilt_x'])} falls over {sum(args.num_envs * args.steps for _ in results) * base.step_dt / 60:.0f}"
        f" env-minutes. Direction (the pelvis tilt the step before): forward {pct(forward.mean())},"
        f" backward {pct(backward.mean())}, sideways {pct(sideways.mean())}.",
        f"- {pct(fall['needs_crouch'].mean())} happened during goals that need a crouch; the pelvis was on average"
        f" {100 * fall['pelvis_drop'].mean():.1f} cm down, the lower target {fall['target_height'].mean():.2f} m above the ground.",
        f"- {pct((fall['episode_time'] < 1.0).mean())} came within 1 s of an episode start (a reset to standing),"
        f" {pct(((fall['episode_time'] >= 1.0) & (fall['pelvis_drop'] >= 0.2)).mean())} later from a crouch deeper than 20 cm,"
        f" {pct(((fall['episode_time'] >= 1.0) & (fall['pelvis_drop'] < 0.2)).mean())} later from higher.",
    ]
else:
    lines.append("- No falls.")
lines += ["", "## Goals below the standing reach", "",
          f"No arm pose reaches a wrist target below {STANDING_REACH:.2f} m above the ground standing (the lowest pose in"
          " the target table), so reaching one takes a crouch. All pairs pooled, by the lower target's height:", "",
          "| lower target above the ground | goals | reached | pelvis drop | wrist error |", "|---|---|---|---|---|"]
for b in by_height:
    lines.append(
        f"| {b['label'].replace(chr(10), ' ')} | {b['goals']} | {pct(b['success']) if b['goals'] else '-'} |"
        f" {100 * b['pelvis_drop']:.1f} cm | {100 * b['pos_err']:.1f} cm |"
    )
all_drop = np.concatenate([r["data"]["pelvis_drop"] for r in results])
lines += ["", f"- Deepest crouch: pelvis drop {100 * np.percentile(all_drop, 95):.1f} cm at the 95th percentile,"
          f" {100 * all_drop.max():.1f} cm at most (pelvis {STANDING_PELVIS_HEIGHT - all_drop.max():.2f} m above the ground)."]
lines += ["", "## How the legs do it", ""]
for s in spreads:
    rs = sorted((r for r in results if r["spread"] == s), key=lambda r: r["drop"])
    first, last = rs[0]["data"], rs[-1]["data"]
    lines.append(
        f"- Spread {s}, drop {rs[0]['drop']} -> {rs[-1]['drop']}: knee {first['knee'].mean():.2f} -> {last['knee'].mean():.2f} rad,"
        f" hip pitch {first['hip_pitch'].mean():.2f} -> {last['hip_pitch'].mean():.2f}, ankle pitch"
        f" {first['ankle_pitch'].mean():.2f} -> {last['ankle_pitch'].mean():.2f}, pelvis pitch"
        f" {first['pelvis_pitch'].mean():.2f} -> {last['pelvis_pitch'].mean():.2f} rad."
    )
all_ankle = np.concatenate([r["data"]["ankle_pitch"] for r in results])
all_knee = np.concatenate([r["data"]["knee"] for r in results])
lines.append(
    f"- Joint headroom: ankle pitch within 0.1 rad of its soft limit {pct(np.mean(np.min(np.abs(all_ankle[:, None] - np.array(soft_limits['ankle_pitch'])[None]), axis=1) < 0.1))}"
    f" of the time; knee max {all_knee.max():.2f} rad (soft limit {soft_limits['knee'][1]:.2f})."
)
lines += ["", "Figures: `crouch_response.png`, `posture.png`, `outcomes.png`, `reach_by_height.png`, `timeline.png`"
          + (", video `crouch.mp4`, stills `crouch_stills.png` and `stills/`." if focus_result and focus_result["frames"] else ".")]
report = "\n".join(lines)
with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write(report + "\n")
print("\n" + report)
print(f"\n[eval_crouch] wrote {OUT}")
sys.stdout.flush()
# Kit can hang in app.close(); everything is written
os._exit(0)
