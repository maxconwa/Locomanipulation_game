"""One robot through a scripted sequence: walk -> standing reach -> crouch reach -> walk.

The trained command draws modes and targets at random; this script schedules them instead, every
env in step with the same sequence, deterministic (mean) actions, the run's estimator and target
table. The ArmTargetsCommand still runs its own mechanics: world-anchored targets, reach detection
(5 cm / 0.35 rad held 0.2 s, or the goal's timeout), and settle segments at mode switches (0.75 s
stopped before an arm goal, 1.5 s to stand up before walking).

  walk:S         navigation for S seconds at --walk_speed m/s straight ahead, arms at the zero pose
  stand_reach    an arm goal from the reach table at --spread, not lowered
  crouch_reach   an arm goal from the low, in-front part of the table, lowered by --crouch_drop m; with squat
                 tables, a low goal from the squat depth closest to --crouch_drop m of pelvis drop

Writes to <run>/eval_sequence/<checkpoint>/ (or --out): sequence.mp4 (with --video), timeline.png
(phase-shaded: pelvis height, wrist error, commanded vs actual speed), stills.png (one frame per
phase, at its deepest or closest moment) and report.md (per phase: reached?, time, pelvis drop).

    python scripts/skrl/eval_sequence.py --checkpoint <run>/checkpoints/agent_<N>.pt --video
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Flat-Curriculum-Direct-v0")
parser.add_argument("--algorithm", default="mappo")
parser.add_argument("--schedule", default="walk:6,stand_reach,crouch_reach,walk:6")
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--spread", type=int, default=4, help="Reach-table spread level for both reaches.")
parser.add_argument("--crouch_drop", type=float, default=0.35, help="How far the crouch reach's targets are lowered (m).")
parser.add_argument("--reach_timeout", type=float, default=6.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", default=None)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
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
from locomanipulation_game.tasks.direct.locomanip_marl.mdp.commands import _apply, _relative, ground_height  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

SCHEDULE = []
for item in args.schedule.split(","):
    kind, _, seconds = item.partition(":")
    SCHEDULE.append((kind, float(seconds) if seconds else args.reach_timeout))
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
NAME = os.path.splitext(os.path.basename(args.checkpoint))[0]
OUT = args.out or os.path.join(RUN_DIR, "eval_sequence", NAME)
os.makedirs(OUT, exist_ok=True)

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, f"skrl_{args.algorithm}_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
env_cfg.episode_length_s = 120.0  # the whole sequence in one episode
env_cfg.commands.arm_targets.walk_gate = False
env_cfg.commands.arm_targets.resampling_time_range = (args.reach_timeout, args.reach_timeout)
env_cfg.events.push_robot = None  # no random shoves in a demonstration
env_cfg.viewer.origin_type = "asset_root"
env_cfg.viewer.asset_name = "robot"
env_cfg.viewer.env_index = 0
env_cfg.viewer.eye = (2.6, 2.0, 0.1)
env_cfg.viewer.lookat = (0.0, 0.0, -0.2)
env_cfg.scene.height_scanner.debug_vis = False
env_cfg.scene.imu.debug_vis = False
env_cfg.commands.base_velocity.debug_vis = False  # the arm goal frames stay on screen
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None))
base = env.unwrapped
arm = base.command_manager.get_term("arm_targets")
vel = base.command_manager.get_term("base_velocity")
robot = base.scene["robot"]
scanner = base.scene.sensors["height_scanner"]
arm._record_outcomes = lambda reached, missed: None
arm.update_levels = lambda env_ids, fell: None
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)

# ---------------------------------------------------------------- the schedule, applied at each goal event
segment = torch.zeros(args.num_envs, dtype=torch.long, device=base.device)  # next schedule entry per env
phase = torch.full((args.num_envs,), -1, dtype=torch.long, device=base.device)  # running entry (-1 before start)
original_resample = arm._resample_command


def apply_drop(env_ids: torch.Tensor, drop: float):
    """Lower the just-drawn arm goals of env_ids so their total drop is `drop`."""
    extra = drop - arm.height_drop[env_ids]
    arm.anchor_w[env_ids, :, 2] -= extra.unsqueeze(1)
    arm.believed_b[env_ids] = _relative(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], arm.anchor_w[env_ids])
    arm.shadow_b[env_ids] = arm.believed_b[env_ids]
    arm.height_drop[env_ids] = drop
    arm.lowest_target_height[env_ids] -= extra


def apply_squat_goal(env_ids: torch.Tensor, drop: float):
    """Replace the just-drawn arm goals of env_ids with low targets from the squat depth closest to `drop`."""
    j = int(torch.argmin((arm._squat_drops - drop).abs()))
    level = arm.spread_level[env_ids]
    targets = torch.empty(len(env_ids), arm.num_arms, 7, device=base.device)
    for a in range(arm.num_arms):
        pick = (torch.rand(len(env_ids), device=base.device) * arm._squat_counts[a][j, level]).long()
        targets[:, a] = arm._squat_tables[a][j, pick]
    origin, quat = arm.standing_frame_w()
    arm.anchor_w[env_ids] = _apply(origin[env_ids], quat[env_ids], targets)
    arm.believed_b[env_ids] = _relative(robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], arm.anchor_w[env_ids])
    arm.shadow_b[env_ids] = arm.believed_b[env_ids]
    arm.height_drop[env_ids] = arm._squat_drops[j]
    arm.goal_low[env_ids] = True
    arm.lowest_target_height[env_ids] = targets[..., 2].min(dim=1)[0] + arm.cfg.standing_height
    arm.needs_crouch[env_ids] = (targets[..., 2] < arm._standing_min_z).any(dim=1)


SQUAT = arm.cfg.squat_tables
if SQUAT:
    # every drawn goal is a standing one, settle ends included; a crouch reach's is replaced by a squat-table goal
    arm.cfg.low_goal_prob = 0.0


def scheduled_resample(env_ids):
    env_ids = torch.as_tensor(env_ids, device=base.device)
    settling = arm.settling[env_ids].clone()
    # an ending settle applies the mode chosen when it began: same schedule entry
    fresh = env_ids[~settling]
    nxt = segment[fresh].clamp(max=len(SCHEDULE) - 1)
    kinds = [SCHEDULE[i][0] for i in nxt.tolist()]
    groups = {
        "walk": fresh[torch.tensor([k == "walk" for k in kinds], dtype=torch.bool, device=base.device)],
        "stand_reach": fresh[torch.tensor([k == "stand_reach" for k in kinds], dtype=torch.bool, device=base.device)],
        "crouch_reach": fresh[torch.tensor([k == "crouch_reach" for k in kinds], dtype=torch.bool, device=base.device)],
    }
    phase[fresh] = nxt
    segment[fresh] += 1
    if settling.any():
        original_resample(env_ids[settling])
    cfg = arm.cfg
    for kind, ids in groups.items():
        if len(ids) == 0:
            continue
        cfg.arm_goal_prob = 0.0 if kind == "walk" else 1.0
        arm.spread_level[ids] = args.spread
        arm.drop_level[ids] = 0 if kind != "crouch_reach" else cfg.drop_levels
        # the crouch reach draws from the low, in-front part of the table (any drop counts as lowered)
        cfg.low_target_min_drop = -1.0 if kind == "crouch_reach" else 0.02
        original_resample(ids)
        now_walking = ids[~arm.arm_mode[ids] & ~arm.settling[ids]]
        now_reaching = ids[arm.arm_mode[ids]]
        if kind == "walk" and len(now_walking):
            seconds = torch.tensor([SCHEDULE[int(phase[i])][1] for i in now_walking.tolist()], device=base.device)
            arm.time_left[now_walking] = seconds
        if kind == "crouch_reach" and len(now_reaching):
            (apply_squat_goal if SQUAT else apply_drop)(now_reaching, args.crouch_drop)
    cfg.low_target_min_drop = 0.02


arm._resample_command = scheduled_resample


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


# ---------------------------------------------------------------- rollout
max_steps = int((sum(s for _, s in SCHEDULE) + 3.0 * len(SCHEDULE) + 2.0) / base.step_dt)
rec = {k: [] for k in ("phase", "settling", "arm", "pelvis_h", "pos_err", "cmd_speed", "speed", "reached", "fell", "target_h")}
frames = []
with torch.inference_mode():
    obs, _ = base.reset()
    states = base.state()
    for step in range(max_steps):
        # the walk command: straight ahead at a fixed speed, never resampled mid-segment
        walking = ~arm.arm_mode & ~arm.settling
        vel.vel_command_b[walking] = torch.tensor([args.walk_speed, 0.0, 0.0], device=base.device)
        vel.time_left[:] = 1.0e6
        vel.is_standing_env[:] = False
        reached0 = arm.metrics["goals_reached"].clone()
        obs, _, terminated, _, _ = base.step(act(obs, states))
        states = base.state()
        pos_err, _ = arm.errors()
        ground = ground_height(scanner)
        rec["phase"].append(int(phase[0]))
        rec["settling"].append(bool(arm.settling[0]))
        rec["arm"].append(bool(arm.arm_mode[0]))
        rec["pelvis_h"].append((robot.data.root_pos_w[0, 2] - ground[0]).item())
        rec["pos_err"].append(pos_err[0].mean().item() if arm.arm_mode[0] else float("nan"))
        rec["cmd_speed"].append(vel.vel_command_b[0, :2].norm().item())
        rec["speed"].append(robot.data.root_lin_vel_b[0, :2].norm().item())
        rec["reached"].append(bool((arm.metrics["goals_reached"][0] - reached0[0]) > 0))
        rec["fell"].append(bool(terminated["legs"][0]))
        rec["target_h"].append(arm.lowest_target_height[0].item() if arm.arm_mode[0] else float("nan"))
        if args.video:
            frame = base.render()
            if frame is not None:
                frames.append(frame)
        if int(segment[0]) >= len(SCHEDULE) and not arm.settling[0] and rec["phase"][-1] == len(SCHEDULE) - 1 \
                and arm.time_left[0] < base.step_dt:
            break

R = {k: np.array(v) for k, v in rec.items()}
t = np.arange(len(R["phase"])) * base.step_dt
labels = {"walk": "walk", "stand_reach": "standing reach", "crouch_reach": "crouch reach"}

# per-phase summary (env 0) and over all envs at the crouch
rows = []
for i, (kind, seconds) in enumerate(SCHEDULE):
    m = (R["phase"] == i) & ~R["settling"]
    if not m.any():
        rows.append(dict(i=i, kind=kind, start=np.nan, dur=0.0, reached="-", drop=np.nan, err=np.nan, speed=np.nan, target=np.nan))
        continue
    idx = np.flatnonzero(m)
    reached = R["reached"][idx[0]: idx[-1] + 2].any() if kind != "walk" else None
    rows.append(dict(
        i=i, kind=kind, start=t[idx[0]], dur=len(idx) * base.step_dt,
        reached=("yes" if reached else "no") if kind != "walk" else "-",
        drop=STANDING_PELVIS_HEIGHT - np.nanmin(R["pelvis_h"][idx]),
        err=np.nanmin(R["pos_err"][idx]) if kind != "walk" else np.nan,
        speed=np.mean(R["speed"][idx]) if kind == "walk" else np.nan,
        target=np.nanmin(R["target_h"][idx]) if kind != "walk" else np.nan,
    ))

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
shade = {"walk": "#fcfcfb", "stand_reach": "#dbe8f7", "crouch_reach": "#fde3d4"}
fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), sharex=True)
axes[0].plot(t, R["pelvis_h"], color=SERIES[0], linewidth=1.6)
axes[0].axhline(STANDING_PELVIS_HEIGHT, color=REF, linewidth=1.0)
axes[0].set_ylabel("pelvis height (m)")
axes[1].plot(t, 100 * R["pos_err"], color=SERIES[0], linewidth=1.6)
axes[1].axhline(100 * env_cfg.commands.arm_targets.reach_pos_tol, color=REF, linewidth=1.0)
axes[1].set_ylabel("wrist error (cm)")
axes[2].plot(t, R["cmd_speed"], color=REF, linewidth=1.5, label="commanded")
axes[2].plot(t, R["speed"], color=SERIES[0], linewidth=1.5, label="actual")
axes[2].set_ylabel("speed (m/s)")
axes[2].set_xlabel("time (s)")
axes[2].legend(fontsize=8, loc="upper right", ncol=2)
for ax in axes:
    for r in rows:
        if np.isfinite(r["start"]):
            ax.axvspan(r["start"], r["start"] + r["dur"], color=shade[r["kind"]], linewidth=0, zorder=0)
    settle = R["settling"]
    edges = np.flatnonzero(np.diff(np.concatenate([[0], settle.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#ececea", linewidth=0, zorder=0)
    for tr in t[R["reached"]]:
        ax.axvline(tr, color=SERIES[2], linewidth=1.0)
    for tf in t[R["fell"]]:
        ax.axvline(tf, color="#e34948", linewidth=1.2)
for r in rows:
    if np.isfinite(r["start"]):
        axes[0].text(r["start"] + 0.1, axes[0].get_ylim()[1], labels[r["kind"]], fontsize=8, color=INK_2, va="top")
axes[0].set_title("Walk, standing reach, crouch reach, walk: gray settles, green a reached goal, red a fall", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timeline.png"), dpi=150)
plt.close(fig)

if frames:
    import imageio.v2 as imageio

    imageio.mimwrite(os.path.join(OUT, "sequence.mp4"), frames, fps=int(round(1.0 / base.step_dt)), quality=8)
    picks = []
    for r in rows:
        if not np.isfinite(r["start"]):
            continue
        idx = np.flatnonzero((R["phase"] == r["i"]) & ~R["settling"])
        if r["kind"] == "walk":
            i = idx[len(idx) // 2]
        elif r["kind"] == "crouch_reach":
            i = idx[np.nanargmin(R["pelvis_h"][idx])]
        else:
            e = R["pos_err"][idx]
            i = idx[np.nanargmin(e)] if np.isfinite(e).any() else idx[len(idx) // 2]
        picks.append((min(i, len(frames) - 1), labels[r["kind"]]))
    fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 3.4), squeeze=False)
    for ax, (i, label) in zip(axes.flat, picks):
        ax.imshow(frames[i])
        ax.axis("off")
        ax.set_title(f"{label}: t = {t[i]:.1f} s, pelvis {R['pelvis_h'][i]:.2f} m", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stills.png"), dpi=110)
    plt.close(fig)
    for i, label in picks:
        imageio.imwrite(os.path.join(OUT, f"still_{label.replace(' ', '_')}_{t[i]:05.1f}s.png"), frames[i])

lines = [f"# Sequence: `{args.checkpoint}`", "",
         f"Schedule `{args.schedule}`, walk at {args.walk_speed} m/s, reach targets at spread level {args.spread},"
         f" crouch reach lowered {args.crouch_drop} m; env 0 of {args.num_envs}, deterministic actions, no pushes.", "",
         "| # | phase | starts | lasts | reached | lowest target above ground | closest wrist error | deepest pelvis drop | mean speed |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(
        f"| {r['i']} | {labels[r['kind']]} | {r['start']:.1f} s | {r['dur']:.1f} s | {r['reached']} |"
        f" {r['target']:.2f} m | {100 * r['err']:.1f} cm | {100 * r['drop']:.1f} cm | {r['speed']:.2f} m/s |"
        .replace("nan m/s", "-").replace("nan cm", "-").replace("nan m", "-")
    )
lines += ["", f"- Falls: {int(R['fell'].sum())}.", "",
          "Figures: `timeline.png`" + (", `stills.png`, video `sequence.mp4`." if frames else ".")]
with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[eval_sequence] wrote {OUT}")
sys.stdout.flush()
os._exit(0)
