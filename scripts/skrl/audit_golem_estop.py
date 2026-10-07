"""Would GOLEM's safety layer e-stop this policy? Checked at every physics substep in the sim.

Runs a checkpoint as trained (walking, settles and arm goals mixed, curriculum levels as saved) or at fixed
spread:drop levels, with deterministic actions. The env's own golem_estop termination is switched off, so the policy
runs as it would without it and every trip is counted. Against two threshold sets:
  exact    GOLEM's relax_safety_split e-stops, read from the GOLEM checkout
  margin   the same tightened by GolemEstopCfg (what the Golem task trains against)
it reports trips per robot-minute by cause (position / velocity / torque) and mode (walking / settle / arm goal), the
joints that trip, and per joint the closest approach to each threshold. It also reports the share of position targets
outside GOLEM's clip, which the safety layer would change.

Before running, it checks that golem_safety.py's vendored table matches the GOLEM files, and fails if it doesn't.

    python scripts/skrl/audit_golem_estop.py --checkpoint <agent.pt> [--task LocoManip-Marl-Flat-Golem-Direct-v0]
        [--levels 4:10] [--arm_goal_prob 1.0] [--num_envs 512] [--steps 1500] [--out <dir>]
"""

import argparse
import importlib.util
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Flat-Golem-Direct-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500, help="Policy steps (50 per second); the first 100 are not counted.")
parser.add_argument("--levels", default=None, help="spread:drop to hold; default: the run's saved levels.")
parser.add_argument("--arm_goal_prob", type=float, default=None, help="Share of events that are arm goals; default: the task's.")
parser.add_argument("--out", default=None, help="Output folder; default <run>/audit_golem_estop/<checkpoint name>.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.direct.locomanip_marl import golem_safety as gs  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

# -- the vendored table against the GOLEM checkout
spec = importlib.util.spec_from_file_location("golem_joint_limits", f"{gs.GOLEM_SAFETY_DIR}/h12_safety_layer/core/joint_limits.py")
jl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jl)
preset = yaml.safe_load(open(f"{gs.GOLEM_SAFETY_DIR}/config/{gs.GOLEM_PRESET}.yaml"))["limits"]


def per_joint(value):
    return list(value) if isinstance(value, list) else [value] * jl.MOTOR_COUNT


live = list(zip(jl.JOINT_NAMES, [p["low"] for p in jl.URDF_POSITION_LIMITS], [p["high"] for p in jl.URDF_POSITION_LIMITS],
                jl.URDF_VELOCITY_LIMITS, jl.URDF_TORQUE_LIMITS, per_joint(preset["clip"]["position_offset"]),
                per_joint(preset["estop"]["position_offset"]), per_joint(preset["estop"]["velocity_ratio"]),
                per_joint(preset["estop"]["torque_ratio"])))
diff = [(a, b) for a, b in zip(gs.GOLEM_LIMITS, live) if a[0] != b[0] or not np.allclose(a[1:], b[1:])]
if len(live) != len(gs.GOLEM_LIMITS) or diff:
    raise SystemExit(f"golem_safety.GOLEM_LIMITS differs from {gs.GOLEM_SAFETY_DIR} ({gs.GOLEM_PRESET}): {diff[:3]}")
print(f"[INFO] golem_safety.py matches {gs.GOLEM_SAFETY_DIR} ({gs.GOLEM_PRESET})")

torch.set_grad_enabled(False)
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
name = os.path.splitext(os.path.basename(args.checkpoint))[0]
out_dir = args.out or os.path.join(RUN_DIR, "audit_golem_estop", name)
os.makedirs(out_dir, exist_ok=True)
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
if getattr(env_cfg.terminations, "golem_estop", None) is not None:
    env_cfg.terminations.golem_estop = None  # count every trip; the monitor below does the checking
margin_cfg = env_cfg.golem_estop or gs.GolemEstopCfg()
env_cfg.golem_estop = None
if args.arm_goal_prob is not None:
    env_cfg.commands.arm_targets.arm_goal_prob = args.arm_goal_prob
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
robot = base.scene["robot"]
arm = base.command_manager.get_term("arm_targets")
if args.levels:
    spread, drop = (int(x) for x in args.levels.split(":"))
    arm._record_outcomes = lambda reached, missed: None
    arm._record_split_outcomes = lambda *a: None
    arm.update_levels = lambda env_ids, fell: None
    arm.spread_level[:] = spread
    arm.drop_level[:] = drop
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)
dev, n = base.device, base.num_envs

MODES = ("walking", "settle", "arm_goal")
SETS = {"exact": gs.GolemEstopCfg(velocity_margin=1.0, torque_margin=1.0, position_margin=0.0), "margin": margin_cfg}
monitors = {k: gs.GolemEstopMonitor(robot, c, n, dev) for k, c in SETS.items()}
ids = monitors["exact"].joint_ids
names = gs.GOLEM_JOINT_NAMES
J = len(names)
table = torch.tensor([row[1:] for row in gs.GOLEM_LIMITS], device=dev)
low, high = table[:, 0], table[:, 1]
clip_low, clip_high = low + table[:, 4], high - table[:, 4]
ex = monitors["exact"]
onsets = {k: {c: torch.zeros(len(MODES), device=dev) for c in gs.CAUSES} for k in SETS}
joint_onsets = {k: {c: torch.zeros(J, device=dev) for c in gs.CAUSES} for k in SETS}
was = {k: {c: torch.zeros(n, dtype=torch.bool, device=dev) for c in gs.CAUSES} for k in SETS}
substeps = torch.zeros(len(MODES), device=dev)
dq_ratio_max = torch.zeros(J, device=dev)
tau_ratio_max = torch.zeros(J, device=dev)
q_room_min = torch.full((J,), 10.0, device=dev)
target_out = torch.zeros(len(MODES), J, device=dev)
policy_steps = torch.zeros(len(MODES), device=dev)
recording = [False]
skip = torch.ones(n, dtype=torch.bool, device=dev)


def mode_index():
    m = torch.zeros(n, dtype=torch.long, device=dev)
    m[arm.settling] = 1
    m[arm.arm_mode] = 2
    return m


original_update = robot.update


def update(dt):
    """After each physics substep: every monitor's check, onsets counted per cause and mode."""
    original_update(dt)
    if not recording[0]:
        return
    mi = mode_index()
    substeps.add_(torch.bincount(mi, minlength=len(MODES)).float())
    data = robot.data
    q, dq, tau = data.joint_pos[:, ids], data.joint_vel[:, ids].abs(), ex._motor_torque().abs()
    dq_ratio_max.copy_(torch.maximum(dq_ratio_max, (dq / ex.dq_max).max(dim=0).values))
    tau_ratio_max.copy_(torch.maximum(tau_ratio_max, (tau[~skip] / ex.tau_max).max(dim=0).values if (~skip).any() else tau_ratio_max))
    q_room_min.copy_(torch.minimum(q_room_min, torch.minimum(q - ex.q_low, ex.q_high - q).min(dim=0).values))
    for k, mon in monitors.items():
        mon.clear()
        mon._skip_torque[:] = skip
        mon.update()
        for c in gs.CAUSES:
            now = mon.flags[c]
            onset = now & ~was[k][c]
            was[k][c] = now.clone()
            onsets[k][c].add_(torch.bincount(mi[onset], minlength=len(MODES)).float())
        for c, hits in mon.joint_hits.items():
            joint_onsets[k][c].add_(hits)
            hits.zero_()
    skip[:] = False


robot.update = update


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


obs, _ = base.reset()
states = base.state()
warmup = 100
for step in range(args.steps):
    recording[0] = step >= warmup
    obs, _, terminated, truncated, _ = base.step(act(obs, states))
    states = base.state()
    done = (terminated["legs"] | truncated["legs"]).view(-1)
    skip |= done  # computed_torque is stale for the first substep after a reset
    if step < warmup:
        continue
    mi = mode_index()
    policy_steps.add_(torch.bincount(mi, minlength=len(MODES)).float())
    tgt = robot.data.joint_pos_target[:, ids]
    outside = (tgt < clip_low - 1e-6) | (tgt > clip_high + 1e-6)
    target_out.index_add_(0, mi, outside.float())

minutes = substeps * base.physics_dt / 60.0
total_minutes = float(minutes.sum())
out = {"checkpoint": args.checkpoint, "task": args.task, "levels": args.levels or "saved", "robot_minutes": round(total_minutes, 1),
       "time_share": {m: round(float(minutes[i]) / total_minutes, 3) for i, m in enumerate(MODES)}, "sets": {}, "joints": {}}
for k in SETS:
    rates = {c: {m: round(float(onsets[k][c][i] / minutes[i].clamp(min=1e-9)), 3) for i, m in enumerate(MODES)} for c in gs.CAUSES}
    total = sum(float(onsets[k][c].sum()) for c in gs.CAUSES)
    out["sets"][k] = {
        "trips_per_robot_minute": round(total / total_minutes, 3),
        "by_cause_and_mode": rates,
        "joint_substeps_over": {c: {names[j]: int(v) for j, v in enumerate(joint_onsets[k][c].tolist()) if v > 0} for c in gs.CAUSES},
    }
for j, jn in enumerate(names):
    out["joints"][jn] = {
        "peak_velocity_of_estop": round(float(dq_ratio_max[j]), 3),
        "peak_torque_of_estop": round(float(tau_ratio_max[j]), 3),
        "closest_to_position_estop_rad": round(float(q_room_min[j]), 4),
        "targets_outside_clip": {m: round(float(target_out[i, j] / policy_steps[i].clamp(min=1)), 4) for i, m in enumerate(MODES)},
    }
json.dump(out, open(os.path.join(out_dir, "audit_golem_estop.json"), "w"), indent=1)

lines = [f"# GOLEM e-stop audit: `{args.checkpoint}`", "",
         f"{args.task}, {n} robots, {total_minutes:.0f} robot-minutes (walking {out['time_share']['walking']:.0%}, settle"
         f" {out['time_share']['settle']:.0%}, arm goals {out['time_share']['arm_goal']:.0%}), levels {out['levels']}, deterministic"
         f" actions, checked every physics substep against {gs.GOLEM_PRESET}.", "",
         "| thresholds | trips / robot-min | position (walk / settle / arm) | velocity | torque |", "|---|---|---|---|---|"]
for k, label in (("exact", "GOLEM exact"), ("margin", f"margin (x{margin_cfg.velocity_margin} / x{margin_cfg.torque_margin} / {margin_cfg.position_margin} rad)")):
    r = out["sets"][k]
    cells = [" / ".join(f"{r['by_cause_and_mode'][c][m]:.2f}" for m in MODES) for c in ("position", "velocity", "torque")]
    lines.append(f"| {label} | {r['trips_per_robot_minute']:.2f} | " + " | ".join(cells) + " |")
lines += ["", "Joints over GOLEM's exact thresholds (physics substeps):", ""]
for c in gs.CAUSES:
    js = sorted(out["sets"]["exact"]["joint_substeps_over"][c].items(), key=lambda x: -x[1])[:6]
    lines.append(f"- {c}: " + (", ".join(f"{a} {b}" for a, b in js) if js else "none"))
lines += ["", "| joint | peak velocity / e-stop | peak torque / e-stop | closest to position e-stop (rad) | targets outside clip, walk / settle / arm |",
          "|---|---|---|---|---|"]
for jn, r in out["joints"].items():
    lines.append(f"| {jn} | {r['peak_velocity_of_estop']:.2f} | {r['peak_torque_of_estop']:.2f} | {r['closest_to_position_estop_rad']:.3f} | "
                 + " / ".join(f"{100 * v:.1f}%" for v in r["targets_outside_clip"].values()) + " |")
open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
print(f"[INFO] wrote {out_dir}")
os._exit(0)
