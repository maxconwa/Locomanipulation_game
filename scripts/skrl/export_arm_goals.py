"""Sample arm goals the way training does, for GOLEM's game commander.

Builds the task's arm-goal command with the run's target tables, holds the reach curriculum at each requested
spread:drop pair, and draws --goals goals per pair with the command's own sampler (ArmTargetsCommand.
_resample_command): half standing goals from the level's region, half low goals built in a squat. Each goal is
saved in the standing frame (origin standing_height above the ground under the pelvis, yaw of the pelvis), which
is the pelvis frame of a robot standing upright: the commander sends it as the pelvis-frame goal.

Writes <out>/arm_goals_<spread>_<drop>.npz with
  poses       (N, 2, 7)  left, right wrist_yaw_link pose (x, y, z, qw, qx, qy, qz), standing frame
  low         (N,)       built in a squat
  drop        (N,)       the squat's pelvis drop (m)
  needs_crouch(N,)       some target below what any standing arm pose reaches

    python scripts/skrl/export_arm_goals.py --checkpoint <agent.pt> --levels 4:10,7:10 --out <dir>
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Its run's target tables (and curriculum) are used.")
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--levels", default="4:10")
parser.add_argument("--goals", type=int, default=2048)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.mdp.commands import _relative  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

torch.set_grad_enabled(False)
RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
os.makedirs(args.out, exist_ok=True)
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
env_cfg.commands.arm_targets.arm_goal_prob = 1.0
base = gym.make(args.task, cfg=env_cfg).unwrapped
arm = base.command_manager.get_term("arm_targets")
base.reset()
ids = torch.arange(base.num_envs, device=base.device)
for pair in args.levels.split(","):
    spread, drop = (int(x) for x in pair.split(":"))
    arm.spread_level[:] = spread
    arm.drop_level[:] = drop
    poses, low, drops, crouch = [], [], [], []
    while sum(len(p) for p in poses) < args.goals:
        arm.settling[:] = False
        arm.arm_mode[:] = True
        arm._resample_command(ids)
        origin, quat = arm.standing_frame_w()
        poses.append(_relative(origin, quat, arm.anchor_w).cpu().numpy())
        low.append(arm.goal_low.cpu().numpy())
        drops.append(arm.height_drop.cpu().numpy())
        crouch.append(arm.needs_crouch.cpu().numpy())
    out = {k: np.concatenate(v)[: args.goals] for k, v in (("poses", poses), ("low", low), ("drop", drops), ("needs_crouch", crouch))}
    path = os.path.join(args.out, f"arm_goals_{spread}_{drop}.npz")
    np.savez(path, **out)
    print(f"{path}: {len(out['poses'])} goals, {out['low'].mean():.0%} low, {out['needs_crouch'].mean():.0%} need a crouch,"
          f" z {out['poses'][..., 2].min():.2f}..{out['poses'][..., 2].max():.2f} m")
os._exit(0)
