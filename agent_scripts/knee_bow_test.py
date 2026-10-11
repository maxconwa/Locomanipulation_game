"""Knee-bow test: do the knees collapse inward (valgus) when the policies crouch?

    python agent_scripts/knee_bow_test.py --checkpoint <run>/checkpoints/agent_<N>.pt [--headless] \
        [--goals rest|grid] [--num_envs 32] [--max_depth 1.0] [--goal_s 6]

Each env stands for SETTLE_S (zero velocity command), then gets one arm goal, placed in the standing frame the way
ArmTargetsCommand places one, and held for --goal_s:
    rest  both wrists at their navigation rest pose, lowered by a depth swept evenly over [0, --max_depth] across the
          envs: a pure, left-right symmetric squat (default)
    grid  the reach evaluation's goal grid (logs/reach/goals/eval_goals_v1.csv) without extension: real reach goals at
          depths 0..1 m, the base goals from --first_base on, one goal per env (11 depths per base goal)
The task (two-agent or whole-body) follows from the checkpoint's agents, as in paper/reach/tools/eval_isaac.py.
Physics is nominal (friction at the training midpoint, no torso mass offset, no pushes); the curriculum is frozen.

Recorded every policy step, in the robot's heading frame (pelvis position, yaw only, z up), per leg: the hip
(hip_roll_link), knee (knee_link) and ankle (ankle_pitch_link) origins, and the 12 leg joint angles. Derived:
    knee_gap     knee width minus ankle width (m): < 0 when the knees sit inside the ankles
    valgus       per leg, how far the knee sits medial of the hip-ankle line in the frontal (y-z) plane (m):
                 > 0 is the knee collapsing in, < 0 bowing out
    pelvis_drop  standing height minus the pelvis height above the ground (m)

Writes <out>/knee_bow.npz (traces), <out>/summary.txt and <out>/knee_bow.png. With --video, also a camera in front
of env --view_env (default the last, the deepest rest goal) at knee height: <out>/knee_bow_front.mp4 over the stand
and the goal, and <out>/knee_bow_front.png, the last standing frame beside the last goal frame.
Default out: logs/knee_bow/<run>_<N>_<goals>/.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mujoco_evals.common import DEFAULT_ASSETS  # noqa: E402

os.environ.setdefault("CL_ASSETS_DIR", str(DEFAULT_ASSETS))  # the task reads it when imported
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--goals", choices=["rest", "grid"], default="rest")
parser.add_argument("--num_envs", type=int, default=32, help="rest: envs in the depth sweep; grid: at most this many goals")
parser.add_argument("--max_depth", type=float, default=1.0, help="rest: the deepest wrist drop (m); the curriculum tops out at 1.0")
parser.add_argument("--first_base", type=int, default=0, help="grid: the first base goal")
parser.add_argument("--goal_s", type=float, default=6.0, help="s the goal is held")
parser.add_argument("--steady_s", type=float, default=1.0, help="per-env steady state: the mean over the goal's last this-many s")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None)
parser.add_argument("--video", action="store_true", help="render a front view of --view_env")
parser.add_argument("--view_env", type=int, default=-1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import LOWER_JOINT_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.mdp.commands import _apply, _relative, ground_height  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402
from mujoco_evals.common import GOALS_DIR, POLICY_DT, SETTLE_S, goal_pose, read_goals  # noqa: E402
from mujoco_evals.policy import WHOLE_BODY_TASK, checkpoint_agents, load_actors  # noqa: E402

checkpoint = Path(args.checkpoint).resolve()
run_dir = checkpoint.parent.parent
task = WHOLE_BODY_TASK if checkpoint_agents(str(checkpoint)) == ["whole"] else "LocoManip-Marl-Direct-v0"
step_n = checkpoint.stem.split("_")[-1]
out_dir = Path(args.out) if args.out else REPO / "logs" / "knee_bow" / f"{run_dir.name}_{step_n}_{args.goals}"
out_dir.mkdir(parents=True, exist_ok=True)
estimator = estimator_checkpoint_for(str(checkpoint))
if estimator is None:
    raise FileNotFoundError(f"no pelvis estimator beside {checkpoint} (<run>/estimator/estimator_<N>.pt)")

if args.goals == "grid":
    grid = [g for g in read_goals(GOALS_DIR / "eval_goals_v1.csv") if g["ext"] == 0.0 and g["base_id"] >= args.first_base]
    grid = grid[: args.num_envs]
    n = len(grid)
else:
    n = args.num_envs

# The env loads its arm target tables from log_dir: give it a copy of the run's, so a settings mismatch rebuilds them
# here rather than over the run's file.
env_dir = out_dir / "envdir"
env_dir.mkdir(exist_ok=True)
table = run_dir / "arm_target_tables_v3.pt"
if table.is_file() and not (env_dir / table.name).is_file():
    shutil.copy(table, env_dir / table.name)

cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = n
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.log_dir = str(env_dir)
cfg.estimator.train = False
cfg.estimator.checkpoint_path = estimator
cfg.episode_length_s = 1.0e4
cfg.commands.arm_targets.arm_goal_prob = 0.0  # the only goal is the one placed below
cfg.events.push_robot = None
cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
cfg.events.physics_material.params["dynamic_friction_range"] = (0.65, 0.65)
cfg.events.add_torso_mass.params["mass_distribution_params"] = (0.0, 0.0)
view_env = args.view_env % n
if args.video:
    cfg.commands.arm_targets.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.height_scanner.debug_vis = False
    cfg.viewer.origin_type = "asset_root"  # the camera follows the robot; view_front sets its offset
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = view_env

base = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
dev = base.device
actors = load_actors(str(checkpoint), device=dev, clip_actions=base.cfg.clip_actions)
robot, scanner = base.scene["robot"], base.scene.sensors["height_scanner"]
vel, arm = base.command_manager.get_term("base_velocity"), base.command_manager.get_term("arm_targets")
arm._record_outcomes = lambda ended: None  # the curriculum stays frozen
arm.update_levels = lambda env_ids, fell: None
leg_joints = robot.find_joints(LOWER_JOINT_NAMES, preserve_order=True)[0]
# (3 points: hip, knee, ankle) x (2 legs: left, right)
points = [robot.find_bodies([f"left_{p}_link", f"right_{p}_link"], preserve_order=True)[0]
          for p in ("hip_roll", "knee", "ankle_pitch")]
point_ids = [i for pair in points for i in pair]
first_agent = base.cfg.possible_agents[0]  # dones are shared (one body)


def act(obs: dict) -> dict:
    return {agent: actor(obs[agent]) for agent, actor in actors.items()}


def hold_still():
    """Zero velocity command and no command events: the robot stands, a placed goal stays."""
    vel.vel_command_b[:] = 0.0
    vel.time_left[:] = 1.0e6
    vel.is_standing_env[:] = True
    arm.time_left[:] = 1.0e6


def heading_frame(p_w: torch.Tensor) -> torch.Tensor:
    """(n, k, 3) world points in the pelvis's heading frame: origin at the pelvis, yaw only."""
    k = p_w.shape[1]
    q = yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    rel = (p_w - robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3)
    return quat_apply_inverse(q, rel).view(-1, k, 3)


frames = []


def view_front():
    """--video: a frame from 1.8 m in front of view_env's pelvis, at knee height, looking back at its knees."""
    if not args.video:
        return
    q = yaw_quat(robot.data.root_quat_w[view_env : view_env + 1])
    eye = quat_apply(q, torch.tensor([[1.8, 0.0, -0.5]], device=dev))[0].cpu().numpy()
    base.viewport_camera_controller.update_view_location(eye=eye, lookat=(0.0, 0.0, -0.5))
    frame = base.render()
    if frame is not None:
        frames.append(frame.copy())


settle_steps, goal_steps = round(SETTLE_S / POLICY_DT), round(args.goal_s / POLICY_DT)
log = {k: [] for k in ("points", "joint_pos", "pelvis_h", "wrist_err", "alive", "valgus_cost", "legs_reward")}
# the legs' (or the whole body's) knee_valgus reward term and its reward manager, when the task has one
valgus_term = next(((m, m.active_terms.index(name)) for m in base.reward_managers.values()
                    for name in m.active_terms if name.endswith("knee_valgus")), None)
print(f"[knee_bow] {checkpoint} ({task}): {n} envs, {args.goals} goals, {args.goal_s:.1f} s each", flush=True)

with torch.inference_mode():
    obs, _ = base.reset(seed=args.seed)
    for _ in range(settle_steps):
        hold_still()
        obs, _, _, _, _ = base.step(act(obs))
        view_front()
    standing_frames = len(frames)

    # -- the goal, in the standing frame now, as ArmTargetsCommand._resample_command places one
    if args.goals == "grid":
        poses = torch.tensor([[goal_pose(g, "left"), goal_pose(g, "right")] for g in grid], dtype=torch.float32, device=dev)
        depth = torch.tensor([g["depth"] for g in grid], device=dev)
    else:
        depth = torch.linspace(0.0, args.max_depth, n, device=dev)
        poses = arm.rest_pose_b.unsqueeze(0).repeat(n, 1, 1)
        poses[..., 2] -= depth.unsqueeze(1)
    arm.invalidate_errors()
    arm.arm_mode[:] = True
    arm.goal_best_error[:] = float("inf")
    arm.goal_reached[:] = False
    arm.hold_time[:] = 0.0
    arm.target_drop[:] = depth
    arm.needs_crouch[:] = (poses[..., 2] < arm._standing_min_z).any(dim=1)
    arm.use_estimate[:] = True
    origin, quat = arm.standing_frame_w()
    arm.anchor_w[:] = _apply(origin, quat, poses)
    arm.believed_b[:] = _relative(robot.data.root_pos_w, robot.data.root_quat_w, arm.anchor_w)
    arm.shadow_b[:] = arm.believed_b
    hold_still()
    obs = {a: base.observation_manager.compute_group(a) for a in base.cfg.possible_agents}
    needs_crouch = arm.needs_crouch.cpu().numpy().copy()

    alive = torch.ones(n, dtype=torch.bool, device=dev)
    for _ in range(goal_steps):
        hold_still()
        obs, _, terminated, truncated, _ = base.step(act(obs))
        alive &= ~(terminated[first_agent] | truncated[first_agent])
        view_front()
        d = robot.data
        step = {
            "points": heading_frame(d.body_pos_w[:, point_ids]).view(n, 3, 2, 3),
            "joint_pos": d.joint_pos[:, leg_joints],
            "pelvis_h": d.root_pos_w[:, 2] - ground_height(scanner),
            "wrist_err": arm.errors()[0].mean(dim=1),
            "alive": alive,
        }
        if valgus_term is not None:
            manager, idx = valgus_term
            step["valgus_cost"] = manager._step_reward[:, idx]  # per s, weighted
            step["legs_reward"] = manager._reward_buf / base.step_dt  # per s, before the floor at 0
        else:
            step["valgus_cost"] = step["legs_reward"] = torch.full((n,), float("nan"), device=dev)
        # copies: on a CPU device .cpu().numpy() is a view, and the sim's buffers change in place
        for k, v in step.items():
            log[k].append(v.cpu().numpy().copy())

T = {k: np.stack(v) for k, v in log.items()}  # (steps, n, ...)
depth = depth.cpu().numpy()
t = (np.arange(goal_steps) + 1) * POLICY_DT
hip, knee, ankle = T["points"][:, :, 0], T["points"][:, :, 1], T["points"][:, :, 2]  # (steps, n, leg, xyz)
knee_width = knee[..., 0, 1] - knee[..., 1, 1]
ankle_width = ankle[..., 0, 1] - ankle[..., 1, 1]
knee_gap = knee_width - ankle_width
# the hip-ankle line's y at the knee's height; medial is toward y = 0, so the right leg's sign flips
span = ankle[..., 2] - hip[..., 2]
s = (knee[..., 2] - hip[..., 2]) / np.where(np.abs(span) > 1e-3, span, np.nan)
y_line = hip[..., 1] + s * (ankle[..., 1] - hip[..., 1])
valgus = np.array([1.0, -1.0]) * (y_line - knee[..., 1])  # (steps, n, leg)
pelvis_drop = arm.cfg.standing_height - T["pelvis_h"]
q = np.degrees(T["joint_pos"])  # LOWER_JOINT_NAMES order
names = LOWER_JOINT_NAMES
hip_roll = q[..., [names.index("left_hip_roll_joint"), names.index("right_hip_roll_joint")]]
hip_yaw = q[..., [names.index("left_hip_yaw_joint"), names.index("right_hip_yaw_joint")]]
knee_q = q[..., [names.index("left_knee_joint"), names.index("right_knee_joint")]]
alive = T["alive"]
survived = alive[-1]

np.savez(out_dir / "knee_bow.npz", t=t, depth=depth, needs_crouch=needs_crouch, alive=alive, points=T["points"],
         joint_pos=T["joint_pos"], joint_names=np.array(names), pelvis_h=T["pelvis_h"], wrist_err=T["wrist_err"],
         knee_gap=knee_gap, valgus=valgus, pelvis_drop=pelvis_drop, valgus_cost=T["valgus_cost"],
         legs_reward=T["legs_reward"])

# -- per env: the steady state, the mean over the goal's last steady_s
tail = slice(-max(1, round(args.steady_s / POLICY_DT)), None)
ss = {k: v[tail].mean(axis=0) for k, v in (("pelvis_drop", pelvis_drop), ("knee_width", knee_width),
                                            ("ankle_width", ankle_width), ("knee_gap", knee_gap), ("valgus", valgus),
                                            ("hip_roll", hip_roll), ("hip_yaw", hip_yaw), ("knee_q", knee_q),
                                            ("wrist_err", T["wrist_err"]), ("valgus_cost", T["valgus_cost"]),
                                            ("legs_reward", T["legs_reward"]))}
lines = [f"knee-bow test: {checkpoint}", f"task {task}, goals {args.goals}, {n} envs, goal held {args.goal_s:.1f} s,"
         f" steady state = last {args.steady_s:.1f} s; {int((~survived).sum())} fell",
         "lengths in cm, angles in deg; valgus > 0 = knee medial of the hip-ankle line (collapsing in);"
         " knee_gap < 0 = knees inside the ankles; vCost = the knee_valgus reward term, rLegs = the agent's reward"
         " before its floor at 0, both per s (nan: the task has no knee_valgus)", "",
         "per env (steady state)",
         f"{'env':>4} {'depth':>6} {'crouch':>6} {'fell':>4} {'p_drop':>7} {'kneeW':>6} {'ankW':>6} {'gap':>6}"
         f" {'valgL':>6} {'valgR':>6} {'rollL':>6} {'rollR':>6} {'yawL':>6} {'yawR':>6} {'kneeL':>6} {'kneeR':>6} {'wrErr':>6} {'vCost':>6} {'rLegs':>6}"]
for i in np.argsort(depth, kind="stable"):
    lines.append(f"{i:>4} {100 * depth[i]:>6.0f} {'y' if needs_crouch[i] else '-':>6} {'y' if not survived[i] else '-':>4}"
                 f" {100 * ss['pelvis_drop'][i]:>7.1f} {100 * ss['knee_width'][i]:>6.1f} {100 * ss['ankle_width'][i]:>6.1f}"
                 f" {100 * ss['knee_gap'][i]:>6.1f} {100 * ss['valgus'][i, 0]:>6.1f} {100 * ss['valgus'][i, 1]:>6.1f}"
                 f" {ss['hip_roll'][i, 0]:>6.1f} {ss['hip_roll'][i, 1]:>6.1f} {ss['hip_yaw'][i, 0]:>6.1f}"
                 f" {ss['hip_yaw'][i, 1]:>6.1f} {ss['knee_q'][i, 0]:>6.1f} {ss['knee_q'][i, 1]:>6.1f}"
                 f" {100 * ss['wrist_err'][i]:>6.1f} {ss['valgus_cost'][i]:>6.2f} {ss['legs_reward'][i]:>6.2f}")

# -- every alive step, binned by pelvis drop
lines += ["", "all alive steps, by pelvis drop",
          f"{'p_drop':>12} {'steps':>6} {'kneeW':>6} {'ankW':>6} {'gap':>6} {'valgL':>6} {'valgR':>6} {'rollL':>6} {'rollR':>6}"]
bin_w = 0.05
edges = (np.arange(np.floor(np.nanmin(pelvis_drop[alive]) / bin_w) * bin_w, np.nanmax(pelvis_drop[alive]) + bin_w, bin_w)
         if alive.any() else np.zeros(0))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = alive & (pelvis_drop >= lo) & (pelvis_drop < hi)
    if m.sum() == 0:
        continue
    lines.append(f"{100 * lo:>5.0f}..{100 * hi:<5.0f} {int(m.sum()):>6} {100 * knee_width[m].mean():>6.1f}"
                 f" {100 * ankle_width[m].mean():>6.1f} {100 * knee_gap[m].mean():>6.1f} {100 * np.nanmean(valgus[..., 0][m]):>6.1f}"
                 f" {100 * np.nanmean(valgus[..., 1][m]):>6.1f} {hip_roll[..., 0][m].mean():>6.1f} {hip_roll[..., 1][m].mean():>6.1f}")
summary = "\n".join(lines)
(out_dir / "summary.txt").write_text(summary + "\n")
print(summary, flush=True)

# -- figure
LEFT, RIGHT = "#2a78d6", "#eb6834"  # categorical slots 1, 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
ok = survived
x = 100 * ss["pelvis_drop"][ok]
mk = dict(s=36, edgecolors="white", linewidths=1.0, zorder=3)

a = ax[0, 0]
a.scatter(x, 100 * ss["knee_width"][ok], color=LEFT, label="knee width", **mk)
a.scatter(x, 100 * ss["ankle_width"][ok], color=RIGHT, marker="s", label="ankle width", **mk)
a.set(xlabel="pelvis drop (cm)", ylabel="lateral distance (cm)", title="Knee and ankle width")
a.legend()

a = ax[0, 1]
a.axhline(0.0, color=MUTED, linewidth=0.8)
a.scatter(x, 100 * ss["valgus"][ok, 0], color=LEFT, label="left", **mk)
a.scatter(x, 100 * ss["valgus"][ok, 1], color=RIGHT, marker="s", label="right", **mk)
a.set(xlabel="pelvis drop (cm)", ylabel="knee medial of hip-ankle line (cm)", title="Valgus offset (> 0: collapsing in)")
a.legend()

a = ax[1, 0]
a.scatter(x, ss["hip_roll"][ok, 0], color=LEFT, label="hip roll, left", **mk)
a.scatter(x, ss["hip_roll"][ok, 1], color=RIGHT, label="hip roll, right", **mk)
a.scatter(x, ss["hip_yaw"][ok, 0], color=LEFT, marker="^", label="hip yaw, left", **mk)
a.scatter(x, ss["hip_yaw"][ok, 1], color=RIGHT, marker="^", label="hip yaw, right", **mk)
a.set(xlabel="pelvis drop (cm)", ylabel="joint angle (deg, raw sign)", title="Hip roll and yaw")
a.legend(ncols=2)

a = ax[1, 1]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#0d366b"])
norm = matplotlib.colors.Normalize(depth.min(), max(depth.max(), 1e-6))
for i in range(n):
    g = np.where(alive[:, i], 100 * knee_gap[:, i], np.nan)
    a.plot(t, g, color=cmap(norm(depth[i])), linewidth=1.2)
a.axhline(0.0, color=MUTED, linewidth=0.8)
a.set(xlabel="time since goal (s)", ylabel="knee width - ankle width (cm)", title="Knee gap over the goal")
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, label="wrist goal depth (m)")

fig.suptitle(f"Knee-bow test: {run_dir.name} / {checkpoint.name}, {args.goals} goals"
             f" ({int(ok.sum())}/{n} stood; fallen envs left out)", color=INK)
fig.savefig(out_dir / "knee_bow.png", dpi=150)
print(f"[knee_bow] wrote {out_dir}/knee_bow.npz, summary.txt, knee_bow.png", flush=True)

if frames:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    def caption(frame: np.ndarray, text: str) -> np.ndarray:
        image = Image.fromarray(frame)
        ImageDraw.Draw(image).text((12, 10), text, fill=(20, 20, 20))
        return np.asarray(image)

    v = view_env
    imageio.mimwrite(out_dir / "knee_bow_front.mp4", frames, fps=round(1.0 / POLICY_DT), quality=8)
    pair = [caption(frames[standing_frames - 1], f"env {v}: standing"),
            caption(frames[-1], f"env {v}: goal depth {depth[v]:.2f} m, pelvis drop {100 * ss['pelvis_drop'][v]:.0f} cm,"
                                f" knee gap {100 * ss['knee_gap'][v]:.0f} cm")]
    imageio.imwrite(out_dir / "knee_bow_front.png", np.concatenate(pair, axis=1))
    print(f"[knee_bow] wrote {out_dir}/knee_bow_front.mp4, knee_bow_front.png (env {v})", flush=True)
os._exit(0)  # Isaac's app.close() hangs
