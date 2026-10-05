"""Does a trained LocoManip-Marl policy pair crouch for low arm targets?

Headless. Every env gets arm goals only (arm_goal_prob 1), with the arm
curriculum held at fixed (spread, drop) levels, deterministic (mean) actions,
and the run's estimator. For each pair it prints goals reached and missed and
falls (per env-minute), the wrist errors, and the commanded target drop against
the actual pelvis drop below standing height, with the regression slope of one
on the other. A slope near 1 means the pelvis follows the targets down; near 0
means it doesn't move.

    python scripts/skrl/eval_crouch.py --checkpoint <run>/checkpoints/agent_<N>.pt \\
        [--task LocoManip-Marl-Stand-Direct-v0] [--algorithm mappo] [--grid 4:0,4:2,4:4]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True, help="skrl agent checkpoint; its run's estimator and target table are used.")
parser.add_argument("--task", default="LocoManip-Marl-Stand-Direct-v0")
parser.add_argument("--algorithm", default="mappo", help="skrl config the checkpoint was trained with: ippo or mappo.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1500, help="Policy steps per (spread, drop) pair; 50 per second.")
parser.add_argument("--grid", default="4:0,4:2,4:4,4:5,8:0,8:4", help="spread:drop level pairs to hold.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import STANDING_PELVIS_HEIGHT  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.mdp.commands import ground_height  # noqa: E402
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, f"skrl_{args.algorithm}_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
# the run folder: its saved target table and estimator (with curriculum state) are reused
env_cfg.log_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
env_cfg.commands.arm_targets.arm_goal_prob = 1.0
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
arm = base.command_manager.get_term("arm_targets")
robot = base.scene["robot"]
scanner = base.scene.sensors["height_scanner"]
knee_ids, _ = robot.find_joints(["left_knee_joint", "right_knee_joint"])
# hold each test level exactly: no curriculum moves
arm._record_outcomes = lambda reached, missed: None
arm.update_levels = lambda env_ids, fell: None

runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)


def evaluate(spread: int, drop: int):
    arm.spread_level[:] = spread
    arm.drop_level[:] = drop
    obs, _ = base.reset()
    states = base.state()
    sums = dict(reached=0.0, missed=0.0, falls=0.0, pos=0.0, rot=0.0, knee=0.0, n=0)
    commanded, actual = [], []
    for step in range(args.steps):
        reached0 = arm.metrics["goals_reached"].clone()
        missed0 = arm.metrics["goals_missed"].clone()
        outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
        actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}
        obs, _, terminated, _, _ = base.step(actions)
        states = base.state()
        pos_err, rot_err = arm.errors()
        # per-episode counters; a reset zeroes them, so clamp that step's difference
        sums["reached"] += (arm.metrics["goals_reached"] - reached0).clamp(min=0).sum().item()
        sums["missed"] += (arm.metrics["goals_missed"] - missed0).clamp(min=0).sum().item()
        sums["falls"] += terminated["legs"].float().sum().item()
        sums["pos"] += pos_err.mean().item()
        sums["rot"] += rot_err.mean().item()
        sums["knee"] += robot.data.joint_pos[:, knee_ids].mean().item()
        sums["n"] += 1
        if step > 100 and step % 10 == 0:
            ground = ground_height(scanner)
            ok = torch.isfinite(ground)
            commanded.append(arm.height_drop[ok])
            actual.append((STANDING_PELVIS_HEIGHT - (robot.data.root_pos_w[:, 2] - ground))[ok])
    minutes = args.num_envs * args.steps * base.step_dt / 60.0
    cmd, act = torch.cat(commanded), torch.cat(actual)
    slope = float(((cmd - cmd.mean()) * (act - act.mean())).sum() / ((cmd - cmd.mean()).square().sum() + 1e-9))
    print(
        f"EVAL {spread:3d}:{drop:<3d} | {sums['reached'] / minutes:7.2f} {sums['missed'] / minutes:6.2f} |"
        f" {sums['falls'] / minutes:6.3f} | {sums['pos'] / sums['n']:.3f} m {sums['rot'] / sums['n']:.3f} rad |"
        f" {cmd.mean():.3f} -> {act.mean():.3f} m, slope {slope:5.2f} | {sums['knee'] / sums['n']:.2f}"
    )
    sys.stdout.flush()


print(f"EVAL checkpoint {args.checkpoint}  envs {args.num_envs}  steps {args.steps}")
print("EVAL spread:drop | reached missed /env-min | falls/env-min | goal err pos  rot | target drop -> pelvis drop | knee rad")
# one inference-mode block, resets included: the env's buffers are then all
# inference tensors, which can be updated in place
with torch.inference_mode():
    for pair in args.grid.split(","):
        spread, drop = (int(x) for x in pair.split(":"))
        evaluate(spread, drop)
# Kit can hang in app.close(); the results are printed
os._exit(0)
