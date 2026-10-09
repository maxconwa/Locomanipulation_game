"""Evaluate a checkpoint in Isaac: walking and turning, then reaching and squatting.

    python scripts/skrl/eval_final.py --checkpoint <run>/checkpoints/agent_<N>.pt --headless [--out <dir>]

RoboCasa's joint dynamics (passive damping 10, armature 0.1 on every motor joint), no pushes, deterministic actions.

Walking: every env holds each command for 6 s after 2 s of standing. Scored over the command's last 4.5 s: the mean
body-frame velocity and yaw rate, the 90th-percentile foot lift above standing, and the episodes ended by a fall or
a GOLEM e-stop (those envs are left out of the velocities).

Reaching: arm goals only (half standing, half low), the reach curriculum held at each --levels spread:drop pair.
Per 4 s goal: reached (both wrists held 1 s inside 5 cm / 0.35 rad), the closest approach (mean of both wrists), the
wrists' position and orientation error over the goal's last 2 s, the pelvis's deepest drop below standing height
(left out for goals that ended in a fall or an e-stop), the goal's squat depth, and whether it ended in a fall or
an e-stop. Summarized for standing and low goals and by
squat depth.

Writes <out>/eval.json and <out>/eval.md (default <out>: <run>/eval_<N>).
"""

import argparse
import json
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--out", default=None)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--levels", default="4:10,8:10", help="spread:drop pairs for the reach test")
parser.add_argument("--reach_seconds", type=float, default=24.0, help="per level pair")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import FOOT_LINK_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.mdp.commands import ground_height  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

WALKS = [("forward", [0.5, 0.0, 0.0]), ("backward", [-0.4, 0.0, 0.0]), ("left", [0.0, 0.25, 0.0]),
         ("right", [0.0, -0.25, 0.0]), ("turn left", [0.0, 0.0, 0.4]), ("turn right", [0.0, 0.0, -0.4]),
         ("forward + turn", [0.3, 0.0, 0.3]), ("slow forward", [0.15, 0.0, 0.0])]
STAND_S, WALK_S, SKIP_S = 2.0, 6.0, 1.5

checkpoint = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(checkpoint))
step = re.search(r"_(\d+)\.pt$", checkpoint)
out_dir = args.out or os.path.join(run_dir, f"eval_{step.group(1) if step else 'checkpoint'}")
os.makedirs(out_dir, exist_ok=True)

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.log_dir = run_dir
cfg.estimator.train = False
cfg.estimator.checkpoint_path = estimator_checkpoint_for(checkpoint)
cfg.episode_length_s = 1.0e4
cfg.commands.arm_targets.arm_goal_prob = 0.0
cfg.events.push_robot = None
cfg.events.passive_damping.params["damping_distribution_params"] = (10.0, 10.0)
cfg.events.joint_armature.params["armature_distribution_params"] = (0.1, 0.1)
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=cfg))
base = env.unwrapped
runner = Runner(env, agent_cfg)
runner.agent.load(checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)
robot, scanner = base.scene["robot"], base.scene.sensors["height_scanner"]
vel = base.command_manager.get_term("base_velocity")
arm = base.command_manager.get_term("arm_targets")
terms = base.termination_manager
foot_ids = robot.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
N, dt = base.num_envs, base.step_dt
state = {}


def act():
    outputs = runner.agent.act(state["obs"], {a: state["states"] for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


def step_env():
    state["obs"], _, _, _, _ = base.step(act())
    state["states"] = base.state()
    return terms.get_term("fell").cpu().numpy(), terms.get_term("golem_estop").cpu().numpy()


def hold(cmd):
    vel.vel_command_b[:] = torch.tensor(cmd, device=base.device)
    vel.time_left[:] = 1.0e6
    vel.is_standing_env[:] = not any(cmd)


def foot_height() -> np.ndarray:
    return (robot.data.body_pos_w[:, foot_ids, 2] - ground_height(scanner).unsqueeze(1)).cpu().numpy()


results = {"checkpoint": checkpoint, "task": args.task, "num_envs": N, "walking": {}, "reaching": {}}
with torch.no_grad():
    state["obs"], _ = base.reset()
    state["states"] = base.state()

    # -- walking
    rest = []
    for _ in range(int(STAND_S / dt)):
        hold([0.0, 0.0, 0.0])
        step_env()
        rest.append(foot_height())
    rest_z = np.median(np.concatenate(rest[-50:]), axis=0)  # (feet,)
    for name, cmd in WALKS:
        for _ in range(int(STAND_S / dt)):
            hold([0.0, 0.0, 0.0])
            step_env()
        ended, falls, estops, v, lift = np.zeros(N, bool), 0, 0, [], []
        for i in range(int(WALK_S / dt)):
            hold(cmd)
            fell, estop = step_env()
            falls += int(fell.sum())
            estops += int(estop.sum())
            ended |= fell | estop
            if i * dt >= SKIP_S:
                data = robot.data
                v.append(torch.cat([data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]], 1).cpu().numpy())
                lift.append(foot_height() - rest_z)
        v, lift, ok = np.array(v), np.array(lift), ~ended
        mean = v[:, ok].mean(axis=(0, 1)) if ok.any() else np.full(3, np.nan)
        results["walking"][name] = {
            "command": cmd, "vx": float(mean[0]), "vy": float(mean[1]), "yaw_rate": float(mean[2]),
            "foot_lift_p90_cm": float(100 * np.percentile(lift[:, ok], 90)) if ok.any() else None,
            "falls": falls, "estops": estops,
        }
        print(f"[eval_final] {name:15s} {cmd} -> vx {mean[0]:+.3f} vy {mean[1]:+.3f} yaw {mean[2]:+.3f}", flush=True)

    # -- reaching
    arm._record_outcomes = lambda reached, missed: None  # hold the levels
    arm.update_levels = lambda env_ids, fell: None
    arm.cfg.arm_goal_prob = 1.0
    for pair in args.levels.split(","):
        spread, drop = (int(x) for x in pair.split(":"))
        arm.spread_level[:] = spread
        arm.drop_level[:] = drop
        arm.time_left[:] = 0.0  # new goals at these levels now
        goals, cur, prev_anchor = [], [None] * N, arm.anchor_w.clone()
        for _ in range(int(args.reach_seconds / dt)):
            fell, estop = step_env()
            anchor = arm.anchor_w.clone()
            pos_err, rot_err = arm.errors()
            pe, re_ = pos_err.mean(1).cpu().numpy(), rot_err.mean(1).cpu().numpy()
            drop_now = (arm.cfg.standing_height - (robot.data.root_pos_w[:, 2] - ground_height(scanner))).cpu().numpy()
            new = (anchor - prev_anchor).abs().amax(dim=(1, 2)).cpu().numpy() > 1e-6
            mode = arm.arm_mode.cpu().numpy()
            for i in range(N):
                g = cur[i]
                if g is not None and (new[i] or fell[i] or estop[i] or not mode[i]):
                    g.update(fell=bool(fell[i]), estop=bool(estop[i]))
                    goals.append(g)
                    cur[i] = g = None
                if mode[i] and new[i]:
                    # the shallowest squat table's depth is 0 up to rounding (about -1e-7)
                    cur[i] = g = {"low": bool(arm.goal_low[i]), "depth": max(float(arm.height_drop[i]), 0.0), "pelvis_drop": 0.0,
                                  "closest": 10.0, "err": [], "rot": [], "reached": False}
                if g is not None:
                    g["pelvis_drop"] = max(g["pelvis_drop"], float(drop_now[i]))
                    g["closest"] = min(g["closest"], float(pe[i]))
                    g["err"].append(float(pe[i]))
                    g["rot"].append(float(re_[i]))
                    g["reached"] |= bool(arm.goal_reached[i])
            prev_anchor = anchor
        full = [g for g in goals if len(g["err"]) >= int(3.0 / dt) or g["fell"] or g["estop"]]
        for g in full:
            g["err_last2s"] = float(np.mean(g["err"][-int(2.0 / dt):]))
            g["rot_last2s"] = float(np.mean(g["rot"][-int(2.0 / dt):]))
            del g["err"], g["rot"]

        def summary(sel):
            if not sel:
                return None
            e = 100 * np.array([g["err_last2s"] for g in sel])
            return {
                "goals": len(sel), "reached_pct": 100 * float(np.mean([g["reached"] for g in sel])),
                "closest_cm_median": 100 * float(np.median([g["closest"] for g in sel])),
                "error_last2s_cm_median": float(np.median(e)), "error_last2s_cm_p90": float(np.percentile(e, 90)),
                "orientation_last2s_deg_median": float(np.degrees(np.median([g["rot_last2s"] for g in sel]))),
                "falls": sum(g["fell"] for g in sel), "estops": sum(g["estop"] for g in sel),
            }

        low = [g for g in full if g["low"]]
        upright = [g for g in low if not (g["fell"] or g["estop"])]  # a fall drops the pelvis too
        by_depth = {}
        for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)):
            sel = [g for g in low if lo <= g["depth"] < hi]
            kept = [g["pelvis_drop"] for g in upright if lo <= g["depth"] < hi]
            if sel:
                by_depth[f"{100 * lo:.0f}-{100 * hi:.0f} cm"] = summary(sel) | {
                    "pelvis_drop_cm_median": 100 * float(np.median(kept)) if kept else float("nan")}
        drops = 100 * np.array([g["pelvis_drop"] for g in upright]) if upright else np.zeros(1)
        results["reaching"][pair] = {
            "standing": summary([g for g in full if not g["low"]]), "low": summary(low),
            "deepest_pelvis_drop_cm": float(drops.max()), "pelvis_drop_cm_p95": float(np.percentile(drops, 95)),
            "squat_table_depth_cm": 100 * float(arm._squat_drops.max()), "by_depth": by_depth,
        }
        print(f"[eval_final] reach {pair}: {len(full)} goals", flush=True)

# -- write
json.dump(results, open(os.path.join(out_dir, "eval.json"), "w"), indent=1)
md = [f"# Evaluation of `{checkpoint}`", "",
      f"{args.task}, {N} envs, RoboCasa joint dynamics (damping 10, armature 0.1), deterministic actions.", "",
      "## Walking (mean over the last 4.5 s of 6 s per command)", "",
      "| command | vx m/s | vy m/s | yaw rad/s | foot lift p90 cm | falls | e-stops |", "|---|---|---|---|---|---|---|"]
for name, r in results["walking"].items():
    lift = "-" if r["foot_lift_p90_cm"] is None else f"{r['foot_lift_p90_cm']:.1f}"
    md.append(f"| {name} {r['command']} | {r['vx']:+.2f} | {r['vy']:+.2f} | {r['yaw_rate']:+.2f} | {lift} | {r['falls']} | {r['estops']} |")
for pair, r in results["reaching"].items():
    md += ["", f"## Reaching at spread:drop {pair} (4 s goals)", "",
           "| goals | n | reached | closest cm | error last 2 s cm (median / p90) | orientation deg | falls | e-stops |",
           "|---|---|---|---|---|---|---|---|"]
    for kind in ("standing", "low"):
        s = r[kind]
        if s:
            md.append(f"| {kind} | {s['goals']} | {s['reached_pct']:.0f}% | {s['closest_cm_median']:.1f} | "
                      f"{s['error_last2s_cm_median']:.1f} / {s['error_last2s_cm_p90']:.1f} | "
                      f"{s['orientation_last2s_deg_median']:.0f} | {s['falls']} | {s['estops']} |")
    md += ["", f"Deepest pelvis drop {r['deepest_pelvis_drop_cm']:.1f} cm (p95 {r['pelvis_drop_cm_p95']:.1f}); the squat "
           f"tables reach {r['squat_table_depth_cm']:.1f} cm.", "",
           "| squat depth | n | reached | pelvis drop median cm | error last 2 s median cm |", "|---|---|---|---|---|"]
    for depth, s in r["by_depth"].items():
        md.append(f"| {depth} | {s['goals']} | {s['reached_pct']:.0f}% | {s['pelvis_drop_cm_median']:.1f} | "
                  f"{s['error_last2s_cm_median']:.1f} |")
open(os.path.join(out_dir, "eval.md"), "w").write("\n".join(md) + "\n")
print(f"[eval_final] wrote {out_dir}/eval.json and eval.md", flush=True)
sys.stdout.flush()
os._exit(0)
