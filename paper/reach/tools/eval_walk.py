"""Velocity tracking of a checkpoint in Isaac Lab: the legs' task, to set against reach (what sharing costs).

    python paper/reach/tools/eval_walk.py --checkpoint <run>/checkpoints/agent_<N>.pt --label l1 \
        [--num_envs 256] [--seed 0] [--device cuda:0]

Navigation only (no arm goals; the wrists hold their rest pose), training physics at its midpoint, pushes off,
deterministic actors. Every env holds each command of COMMANDS for WALK_S after STAND_S of standing; the velocity is
the pelvis's body-frame (vx, vy, yaw rate) averaged over the last MEASURE_S. Per command:
    tracking error   |v_xy - v_cmd_xy| (m/s) and |yaw rate - cmd| (rad/s), means over upright envs
    ratio            achieved / commanded along the command's main axis
    falls            envs that tilted past 1 rad during the command (left out of the velocities)
Writes logs/reach/results/<label>/isaac_walk/{walk.csv, meta.json}.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachlib.common import DEFAULT_ASSETS as _ASSETS  # noqa: E402
_os.environ.setdefault("CL_ASSETS_DIR", str(_ASSETS))
from isaaclab.app import AppLauncher  # noqa: E402

REACH = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", default=None, help="default: from the checkpoint's agents")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402
from mujoco_evals.policy import WHOLE_BODY_TASK, checkpoint_agents, load_actors  # noqa: E402

COMMANDS = [("stand", (0.0, 0.0, 0.0)), ("forward", (0.5, 0.0, 0.0)), ("slow forward", (0.2, 0.0, 0.0)),
            ("backward", (-0.4, 0.0, 0.0)), ("left", (0.0, 0.25, 0.0)), ("right", (0.0, -0.25, 0.0)),
            ("turn left", (0.0, 0.0, 0.4)), ("turn right", (0.0, 0.0, -0.4)), ("forward + turn", (0.3, 0.0, 0.3))]
STAND_S, WALK_S, MEASURE_S = 2.0, 6.0, 4.5

t0 = time.time()
checkpoint = Path(args.checkpoint).resolve()
if args.task is None:
    args.task = WHOLE_BODY_TASK if checkpoint_agents(str(checkpoint)) == ["whole"] else "LocoManip-Marl-Direct-v0"
out_dir = C.RESULTS_DIR / args.label / "isaac_walk"
out_dir.mkdir(parents=True, exist_ok=True)
env_dir = out_dir / "envdir"
env_dir.mkdir(exist_ok=True)
table = C.GOALS_DIR / "arm_target_tables_v3.pt"
run_table = checkpoint.parent.parent / "arm_target_tables_v3.pt"
for src in (table, run_table):
    if src.is_file() and not (env_dir / src.name).is_file():
        shutil.copy(src, env_dir / src.name)
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.log_dir = str(env_dir)
cfg.estimator.train = False
cfg.estimator.checkpoint_path = estimator_checkpoint_for(str(checkpoint))
cfg.episode_length_s = 1.0e4
cfg.commands.arm_targets.arm_goal_prob = 0.0
cfg.commands.arm_targets.debug_vis = False
cfg.commands.base_velocity.debug_vis = False
cfg.events.push_robot = None
cfg.events.physics_material.params["static_friction_range"] = (0.9, 0.9)
cfg.events.physics_material.params["dynamic_friction_range"] = (0.65, 0.65)
cfg.events.add_torso_mass.params["mass_distribution_params"] = (0.0, 0.0)
env = gym.make(args.task, cfg=cfg)
base = env.unwrapped
actors = load_actors(str(checkpoint), device=base.device, clip_actions=base.cfg.clip_actions)
robot = base.scene["robot"]
vel, arm = base.command_manager.get_term("base_velocity"), base.command_manager.get_term("arm_targets")
dt = base.step_dt
rows = []


def hold(cmd):
    vel.vel_command_b[:] = torch.tensor(cmd, device=base.device)
    vel.time_left[:] = 1.0e6
    vel.is_standing_env[:] = not any(cmd)
    arm.time_left[:] = 1.0e6


with torch.inference_mode():
    obs, _ = base.reset(seed=args.seed)
    for name, cmd in COMMANDS:
        for _ in range(int(STAND_S / dt)):
            hold((0.0, 0.0, 0.0))
            obs, _, _, _, _ = base.step({a: actors[a](obs[a]) for a in actors})
        up = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        v = []
        steps = int(WALK_S / dt)
        for i in range(steps):
            hold(cmd)
            obs, _, term, trunc, _ = base.step({a: actors[a](obs[a]) for a in actors})
            a0 = base.cfg.possible_agents[0]
            up &= ~(term[a0] | trunc[a0])
            if i * dt >= WALK_S - MEASURE_S:
                d = robot.data
                v.append(torch.cat([d.root_lin_vel_b[:, :2], d.root_ang_vel_b[:, 2:3]], 1).cpu().numpy())
        v = np.stack(v).mean(0)                                    # (N, 3)
        ok = up.cpu().numpy()
        c = np.array(cmd)
        err_xy = np.linalg.norm(v[:, :2] - c[:2], axis=1)
        err_yaw = np.abs(v[:, 2] - c[2])
        axis = int(np.argmax(np.abs(c))) if any(cmd) else 0
        ratio = v[:, axis] / c[axis] if any(cmd) else np.full(len(v), np.nan)
        rows.append({"label": args.label, "command": name, "vx_cmd": c[0], "vy_cmd": c[1], "wz_cmd": c[2],
                     "n": int(ok.sum()), "falls": int((~ok).sum()), "vx": float(v[ok, 0].mean()),
                     "vy": float(v[ok, 1].mean()), "wz": float(v[ok, 2].mean()),
                     "err_xy": float(err_xy[ok].mean()), "err_yaw": float(err_yaw[ok].mean()),
                     "ratio": float(np.nanmean(ratio[ok])) if any(cmd) else float("nan"),
                     "lambda_share": C.read_reward_share(checkpoint.parent.parent)})
        print(f"[eval_walk] {name:15s} cmd {cmd} -> vx {rows[-1]['vx']:+.2f} vy {rows[-1]['vy']:+.2f}"
              f" wz {rows[-1]['wz']:+.2f}, falls {rows[-1]['falls']}", flush=True)
        if (~ok).any():
            obs, _ = base.reset(seed=args.seed + len(rows))
C.write_csv(out_dir / "walk.csv", rows)
C.write_json(out_dir / "meta.json", {"label": args.label, "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": C.sha256(checkpoint), "num_envs": args.num_envs,
                                     "seed": args.seed, "commands": COMMANDS, "wall_s": round(time.time() - t0, 1)})
print(f"[eval_walk] wrote {out_dir}", flush=True)
sys.stdout.flush()
os._exit(0)
