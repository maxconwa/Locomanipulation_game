"""Fixed-grid reach evaluation of a LocoManip-Marl-Direct-v0 checkpoint in Isaac Lab, the training simulator.

    python paper/reach/tools/eval_isaac.py --checkpoint <run>/checkpoints/agent_<N>.pt --label lambda1 \
        [--variant full|legs_blind|arms_ik] [--seeds 0,1,2] [--goals paper/reach/goals/eval_goals_v1.csv] \
        [--batch 1600] [--physics nominal|train] [--odometry learned|true] [--limit N]

One trial per env. The robot stands for SETTLE_S (navigation mode, zero velocity command), then one goal of the grid
is placed in the standing frame at that instant, the way ArmTargetsCommand places a goal, and held for GOAL_S. The
actors are the checkpoint's deterministic policies (reachlib.policy, no skrl). The depth curriculum is frozen. The
learned pelvis estimator moves the command the policies see, as on the robot (--odometry true: the true pelvis
motion). Every metric scores the true world target (reachlib.metrics).

Variants, all from the same checkpoint:
    full         both policies as trained
    legs_blind   the legs see navigation inputs (no arm goal, the arms' rest pose) while the arms chase the goal:
                 a lower body that balances but does not cooperate
    arms_ik      the arms' learned residual held at zero: the damped-least-squares IK step alone

Physics: nominal (default) pins the training randomization at its midpoint (friction 0.9 / 0.65, no torso mass
offset) and turns pushes off; train keeps the training distribution (pushes off).

Writes paper/reach/results/<label>/isaac[_<variant>][_<physics>][_<odometry>]/{trials.csv, traces.npz, meta.json}.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from reachlib.common import DEFAULT_ASSETS as _ASSETS  # noqa: E402
_os.environ.setdefault("CL_ASSETS_DIR", str(_ASSETS))   # the task reads it when imported
from isaaclab.app import AppLauncher

REACH = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--variant", choices=["full", "legs_blind", "arms_ik"], default="full")
parser.add_argument("--goals", default=str(REACH / "goals" / "eval_goals_v1.csv"))
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--batch", type=int, default=1600, help="envs per Isaac batch (one trial each)")
parser.add_argument("--physics", choices=["nominal", "train"], default="nominal")
parser.add_argument("--odometry", choices=["learned", "true"], default="learned")
parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N goals (smoke test)")
parser.add_argument("--ext0", action="store_true", help="only the goals without extension (learning curves)")
parser.add_argument("--trace_bases", type=int, default=12, help="full traces for these many base goals, ext 0, seed 0")
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--out", default=None)
parser.add_argument("--zero_gravity", action="store_true",
                    help="pipeline self-test: no gravity, so a zero-action robot stays up while its arms run the IK step"
                         " (not --fix_root: the task's IK indexes a floating base's Jacobian)")
parser.add_argument("--debug", action="store_true", help="print env 0's command and wrist state during the goal")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import ARM_JOINT_NAMES, FOOT_LINK_NAMES  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.mdp.commands import _apply, _relative, ground_height  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402
from reachlib.metrics import trial_metrics  # noqa: E402
from reachlib.policy import describe, load_actors  # noqa: E402

t_start = time.time()
checkpoint = Path(args.checkpoint).resolve()
run_dir = checkpoint.parent.parent
seeds = [int(s) for s in args.seeds.split(",")]
goals = C.read_goals(Path(args.goals))
if args.ext0:
    goals = [g for g in goals if g["ext"] == 0.0]
if args.limit:
    goals = goals[: args.limit]
suffix = "".join(s for s in (
    "" if args.variant == "full" else f"_{args.variant}",
    "" if args.physics == "nominal" else f"_{args.physics}",
    "" if args.odometry == "learned" else f"_{args.odometry}odom") if s)
out_dir = Path(args.out) if args.out else C.RESULTS_DIR / args.label / f"isaac{suffix}"
out_dir.mkdir(parents=True, exist_ok=True)
estimator = estimator_checkpoint_for(str(checkpoint))
if estimator is None and args.odometry == "learned":
    raise FileNotFoundError(f"no pelvis estimator beside {checkpoint} (<run>/estimator/estimator_<N>.pt);"
                            " evaluate with --odometry true or copy the run's estimator/ directory")

# The env loads its arm target tables from log_dir; give it the goal grid's tables so it does not rebuild them
# (they set only the navigation rest pose and the standing-reach floor, both deterministic).
env_dir = out_dir / "envdir"
env_dir.mkdir(exist_ok=True)
table = C.GOALS_DIR / "arm_target_tables_v3.pt"
if table.is_file() and not (env_dir / table.name).is_file():
    shutil.copy(table, env_dir / table.name)

n_envs = min(args.batch, len(goals))
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = n_envs
cfg.sim.device = args.device                       # --device cpu runs PhysX on the CPU when the GPU is taken
cfg.seed = seeds[0]
cfg.log_dir = str(env_dir)
cfg.estimator.train = False
cfg.estimator.checkpoint_path = estimator
cfg.episode_length_s = 1.0e4
cfg.commands.arm_targets.arm_goal_prob = 0.0      # goals come only from the grid
cfg.commands.arm_targets.debug_vis = False
cfg.commands.base_velocity.debug_vis = False
cfg.events.push_robot = None
if args.physics == "nominal":
    cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
    cfg.events.physics_material.params["dynamic_friction_range"] = (0.65, 0.65)
    cfg.events.add_torso_mass.params["mass_distribution_params"] = (0.0, 0.0)

if args.zero_gravity:
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.terminations.fell = None                    # bad_orientation reads tilt from the gravity vector
    cfg.terminations.terrain_out_of_bounds = None
    suffix += "_zerog"
    out_dir = Path(args.out) if args.out else C.RESULTS_DIR / args.label / f"isaac{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
env = gym.make(args.task, cfg=cfg)
base = env.unwrapped
dev = base.device
actors = load_actors(str(checkpoint), device=dev, clip_actions=base.cfg.clip_actions)
for agent, actor in actors.items():
    want = base.cfg.observation_spaces[agent]
    if actor.mean.shape[0] != want:
        raise RuntimeError(f"{agent}: checkpoint expects {actor.mean.shape[0]} observations, the env gives {want}")

robot, scanner, contact = base.scene["robot"], base.scene.sensors["height_scanner"], base.scene.sensors["contact_forces"]
vel, arm = base.command_manager.get_term("base_velocity"), base.command_manager.get_term("arm_targets")
arm._record_outcomes = lambda ended: None           # the curriculum stays frozen
arm.update_levels = lambda env_ids, fell: None
feet = robot.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
feet_sensor = contact.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]
masses = robot.root_physx_view.get_masses().to(dev)                      # (N, bodies)
total_mass = masses.sum(1, keepdim=True)
names = robot.joint_names
groups = {
    "hip": [i for i, n in enumerate(names) if "_hip_" in n],
    "knee": [i for i, n in enumerate(names) if "_knee_" in n],
    "ankle": [i for i, n in enumerate(names) if "_ankle_" in n],
    "arm": [names.index(n) for n in ARM_JOINT_NAMES],
}
effort = robot.data.joint_effort_limits

# legs_blind: where the arm-goal flag and the arm command sit in the legs' observation
term_names = base.observation_manager.active_terms["legs"]
term_dims = [int(np.prod(d)) for d in base.observation_manager.group_obs_term_dim["legs"]]
offsets = dict(zip(term_names, np.cumsum([0] + term_dims[:-1]).tolist()))
goal_flag = slice(offsets["arm_goal"], offsets["arm_goal"] + 1)
goal_cmd = slice(offsets["ee_targets"], offsets["ee_targets"] + 14)
rest = arm.rest_pose_b                                                     # (2, 7) wxyz
rest_xyzw = torch.cat([rest[:, :3], rest[:, 4:7], rest[:, 3:4]], dim=-1).reshape(1, 14)


def act(obs: dict) -> dict:
    legs_obs = obs["legs"]
    if args.variant == "legs_blind":
        legs_obs = legs_obs.clone()
        legs_obs[:, goal_flag] = 0.0
        legs_obs[:, goal_cmd] = rest_xyzw
    actions = {"legs": actors["legs"](legs_obs), "arms": actors["arms"](obs["arms"])}
    if args.variant == "arms_ik":
        actions["arms"] = torch.zeros_like(actions["arms"])
    return actions


def hold_still(goal_envs: torch.Tensor | None = None):
    """Zero velocity command, no command events: every env stands, goal envs keep their goal."""
    vel.vel_command_b[:] = 0.0
    vel.time_left[:] = 1.0e6
    vel.is_standing_env[:] = True
    keep = torch.ones(base.num_envs, dtype=torch.bool, device=dev) if goal_envs is None else ~goal_envs
    arm.time_left[keep] = 1.0e6


def fresh_obs() -> dict:
    return {a: base.observation_manager.compute_group(a) for a in base.cfg.possible_agents}


def pelvis_height() -> torch.Tensor:
    return robot.data.root_pos_w[:, 2] - ground_height(scanner)


settle_steps, goal_steps = int(round(C.SETTLE_S / C.POLICY_DT)), int(round(C.GOAL_S / C.POLICY_DT))
lam = C.read_reward_share(run_dir)
rows, traces, snaps = [], {}, []
trace_ids = {g["goal_id"] for g in goals if g["base_id"] < args.trace_bases and g["ext"] == 0.0}
print(f"[eval_isaac] {checkpoint} lambda={lam} variant={args.variant} physics={args.physics} odometry={args.odometry}:"
      f" {len(goals)} goals x {len(seeds)} seeds, batches of {n_envs}", flush=True)

with torch.inference_mode():
    for seed in seeds:
        for c0 in range(0, len(goals), n_envs):
            chunk = goals[c0:c0 + n_envs]
            m = len(chunk)
            ids = torch.arange(m, device=dev)
            obs, _ = base.reset(seed=seed * 1000 + c0 // n_envs)
            for _ in range(settle_steps):
                hold_still()
                obs, _, _, _, _ = base.step(act(obs))
            # -- the goal, in the standing frame now
            poses = torch.tensor([[C.goal_pose(g, "left"), C.goal_pose(g, "right")] for g in chunk],
                                 dtype=torch.float32, device=dev)                    # (m, 2, 7) wxyz
            h_start = pelvis_height()[:m].cpu().numpy()
            arm.invalidate_errors()
            arm.arm_mode[ids] = True
            arm.goal_best_error[ids] = float("inf")
            arm.goal_reached[ids] = False
            arm.hold_time[ids] = 0.0
            arm.target_drop[ids] = torch.tensor([g["depth"] for g in chunk], device=dev)
            arm.needs_crouch[ids] = (poses[..., 2] < arm._standing_min_z).any(dim=1)
            arm.use_estimate[ids] = args.odometry == "learned"
            origin, quat = arm.standing_frame_w()
            arm.anchor_w[ids] = _apply(origin[ids], quat[ids], poses)
            arm.believed_b[ids] = _relative(robot.data.root_pos_w[ids], robot.data.root_quat_w[ids], arm.anchor_w[ids])
            arm.shadow_b[ids] = arm.believed_b[ids]
            goal_envs = torch.zeros(base.num_envs, dtype=torch.bool, device=dev)
            goal_envs[ids] = True
            arm.time_left[ids] = C.GOAL_S + 1.0
            hold_still(goal_envs)
            obs = fresh_obs()
            needs_crouch = arm.needs_crouch[:m].cpu().numpy()

            def debug(tag):
                if not args.debug:
                    return
                d = robot.data
                wp = _relative(d.root_pos_w[:1], d.root_quat_w[:1], torch.cat([d.body_pos_w[:1, arm.body_ids], d.body_quat_w[:1, arm.body_ids]], -1))
                pe, re_ = arm.errors()
                aq = d.joint_pos[0, [names.index(n) for n in ARM_JOINT_NAMES[:7]]]
                print(f"[debug {tag}] root {d.root_pos_w[0].tolist()} quat {d.root_quat_w[0].tolist()}\n"
                      f"  goal_s {poses[0].tolist()}\n  anchor_w {arm.anchor_w[0].tolist()}\n  believed_b {arm.believed_b[0].tolist()}\n"
                      f"  wrists_b {wp[0].tolist()}\n  err {pe[0].tolist()} rot {re_[0].tolist()} mode {bool(arm.arm_mode[0])}"
                      f" time_left {float(arm.time_left[0]):.2f}\n  left arm q {aq.tolist()}\n"
                      f"  arm action applied {base.action_manager.get_term('arm_pos').processed_actions[0, :7].tolist()}", flush=True)
            debug("inject")

            log = {k: [] for k in ("pos_err", "rot_err", "alive", "root_pos", "root_quat", "root_ang_vel_b", "ground",
                                   "com", "com_vel", "feet_pos", "feet_quat", "feet_vel", "feet_force", "wrist_pos",
                                   "torque_ratio", "cmd_drift")}
            jp_trace, alive = [], torch.ones(m, dtype=torch.bool, device=dev)
            snap_q = robot.data.joint_pos[:m].clone()
            snap_root = torch.cat([robot.data.root_pos_w[:m], robot.data.root_quat_w[:m]], dim=1).clone()
            for t in range(goal_steps):
                hold_still(goal_envs)
                obs, _, terminated, truncated, _ = base.step(act(obs))
                ended = (terminated["legs"] | truncated["legs"])[:m]
                alive &= ~ended
                d = robot.data
                pos_err, rot_err = arm.errors()
                bodies_com = d.body_com_pos_w
                com = (masses.unsqueeze(-1) * bodies_com).sum(1) / total_mass
                com_vel = (masses.unsqueeze(-1) * d.body_com_lin_vel_w).sum(1) / total_mass
                believed_w = _apply(d.root_pos_w, d.root_quat_w, arm.believed_b)
                drift = torch.norm(believed_w[..., :3] - arm.anchor_w[..., :3], dim=-1).mean(1)
                ratio = (d.applied_torque.abs() / effort).nan_to_num(0.0)
                torque = torch.stack([ratio[:, groups[k]].amax(1) for k in ("hip", "knee", "ankle", "arm")], dim=1)
                step = {
                    "pos_err": pos_err, "rot_err": rot_err, "alive": alive, "root_pos": d.root_pos_w,
                    "root_quat": d.root_quat_w, "root_ang_vel_b": d.root_ang_vel_b, "ground": ground_height(scanner),
                    "com": com, "com_vel": com_vel, "feet_pos": d.body_pos_w[:, feet], "feet_quat": d.body_quat_w[:, feet],
                    "feet_vel": d.body_lin_vel_w[:, feet],
                    "feet_force": contact.data.net_forces_w[:, feet_sensor].norm(dim=-1),
                    "wrist_pos": d.body_pos_w[:, arm.body_ids], "torque_ratio": torque, "cmd_drift": drift,
                }
                for k, v in step.items():
                    log[k].append(v[:m].float().cpu().numpy() if v.dtype != torch.bool else v[:m].cpu().numpy())
                if t in (0, 1, 10, 50, goal_steps - 1):
                    debug(f"t={t}")
                keep = alive.clone()
                snap_q[keep] = d.joint_pos[:m][keep]
                snap_root[keep] = torch.cat([d.root_pos_w[:m], d.root_quat_w[:m]], dim=1)[keep]
                jp_trace.append(torch.cat([d.root_pos_w[:m], d.root_quat_w[:m], d.joint_pos[:m]], dim=1).cpu().numpy())
            log = {k: np.stack(v) for k, v in log.items()}
            series = {}
            metrics = trial_metrics(log, h_start, C.STANDING_PELVIS_HEIGHT, series)
            jp_trace = np.stack(jp_trace)                                              # (T, m, 7 + J)
            for i, g in enumerate(chunk):
                row = {"label": args.label, "sim": "isaac", "kind": "policy", "variant": args.variant,
                       "lambda_share": lam, "checkpoint": str(checkpoint), "seed": seed, "goal_id": g["goal_id"],
                       "base_id": g["base_id"], "depth": g["depth"], "ext": g["ext"],
                       "target_l_x": g["l_x"], "target_l_y": g["l_y"], "target_l_z": g["l_z"],
                       "target_r_x": g["r_x"], "target_r_y": g["r_y"], "target_r_z": g["r_z"],
                       "target_min_height": g["min_height"], "needs_crouch": bool(needs_crouch[i]), **metrics[i]}
                rows.append(row)
                snaps.append(np.concatenate([snap_root[i].cpu().numpy(), snap_q[i].cpu().numpy()]))
                if seed == seeds[0] and g["goal_id"] in trace_ids:
                    traces[g["goal_id"]] = {"state": jp_trace[:, i], "pos_err": log["pos_err"][:, i],
                                            "rot_err": log["rot_err"][:, i], "alive": log["alive"][:, i],
                                            **{k: v[:, i] for k, v in series.items()}}
            done = len(rows)
            ok = np.mean([r["success"] for r in rows[-m:]])
            print(f"[eval_isaac] seed {seed} goals {c0}-{c0 + m - 1}: success {100 * ok:.1f}%"
                  f" ({done}/{len(goals) * len(seeds)} trials, {time.time() - t_start:.0f} s)", flush=True)
            C.write_csv(out_dir / "trials.csv", rows, C.TRIAL_COLUMNS + ["variant"])

np.savez_compressed(
    out_dir / "traces.npz",
    joint_names=np.array(names), snap_goal_id=np.array([r["goal_id"] for r in rows]),
    snap_seed=np.array([r["seed"] for r in rows]), snap_state=np.stack(snaps),
    **{f"trace_{gid}_{k}": v for gid, tr in traces.items() for k, v in tr.items()},
)


def versions(*packages) -> dict:
    import importlib.metadata as md
    out = {}
    for p in packages:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


C.write_json(out_dir / "meta.json", {
    "label": args.label, "sim": "isaac", "variant": args.variant, "physics": args.physics, "odometry": args.odometry,
    "checkpoint": str(checkpoint), "checkpoint_sha256": C.sha256(checkpoint), "run_dir": str(run_dir),
    "lambda_share": lam, "estimator": estimator, "goals": str(Path(args.goals).resolve()),
    "goals_sha256": C.sha256(Path(args.goals)), "n_goals": len(goals), "seeds": seeds, "batch": n_envs,
    "actors": describe(actors), "total_mass_kg": float(total_mass[0]),
    "settle_s": C.SETTLE_S, "goal_s": C.GOAL_S, "task": args.task,
    "code": {"path": str(C.DEFAULT_CODE), **C.git_info(C.DEFAULT_CODE)},
    "versions": versions("isaacsim", "isaaclab", "torch", "skrl"),
    "wall_s": round(time.time() - t_start, 1),
})
print(f"[eval_isaac] wrote {out_dir} ({len(rows)} trials, {time.time() - t_start:.0f} s)", flush=True)
sys.stdout.flush()
os._exit(0)
