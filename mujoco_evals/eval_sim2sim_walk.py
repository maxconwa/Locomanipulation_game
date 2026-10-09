"""The walking schedule of GOLEM's sim-to-sim test (tests/locomanipulation_game/sim2sim_walk.py), in Isaac Lab.

Every env walks the same fixed schedule of velocity commands (no arm goals, no pushes, no reset noise),
deterministic actions, and is recorded in sim2sim_walk.py's format, one directory per env: telemetry.jsonl at
the policy rate (pelvis pose, the 27 motor joints, the ankle_roll_link origins and each foot's vertical contact
force) and commander.jsonl (the schedule's events, falls). GOLEM's scorer then compares it with the MuJoCo runs:

    python mujoco_evals/eval_sim2sim_walk.py --checkpoint <run>/checkpoints/agent_<N>.pt --out <dir> --headless
    python3 ~/GOLEM/tests/locomanipulation_game/sim2sim_walk.py --score <dir>/env* --out <dir>/scored

--passive_damping and --armature hold every motor joint's added passive damping and armature at one value; the
defaults are RoboCasa's, which the task randomizes around (the Isaac asset alone: 0 and 0.01).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# the repository root in place of this directory, whose module names (common, policy, metrics) are generic
sys.path[0] = str(Path(__file__).resolve().parents[1])

from isaaclab.app import AppLauncher  # noqa: E402

DEFAULT_SCHEDULE = ("stand:2,walk:8:0.5:0:0,stand:2,walk:6:-0.4:0:0,stand:2,walk:6:0:0.25:0,stand:2,walk:6:0:-0.25:0,"
                    "stand:2,walk:5:0:0:0.4,stand:2,walk:8:0.3:0:0.3,stand:2,walk:6:0.7:0:0,stand:2,walk:6:0.5:0:0,stand:2")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--algorithm", default="mappo")
parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="stand:S | walk:S:vx:vy:wz items (reach items are skipped)")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--passive_damping", type=float, default=10.0)
parser.add_argument("--armature", type=float, default=0.1)
parser.add_argument("--out", required=True)
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
from locomanipulation_game.assets.h1_2 import ALL_JOINTS_NAMES  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
SETTLE_AFTER_RESET_S = 2.0   # a fallen env restarts its episode: its first seconds don't count as walking

events = []
for item in args.schedule.split(","):
    p = item.split(":")
    if p[0] == "stand":
        events.append({"cmd": [0.0, 0.0, 0.0], "duration": float(p[1])})
    elif p[0] == "walk":
        events.append({"cmd": [float(v) for v in p[2:5]], "duration": float(p[1])})
starts = np.cumsum([0.0] + [e["duration"] for e in events])

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, f"skrl_{args.algorithm}_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
env_cfg.episode_length_s = float(starts[-1]) + 30.0
env_cfg.commands.arm_targets.arm_goal_prob = 0.0
env_cfg.events.push_robot = None
env_cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)
env_cfg.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env_cfg.events.reset_base.params["velocity_range"] = {}
env_cfg.events.passive_damping.params["damping_distribution_params"] = (args.passive_damping, args.passive_damping)
env_cfg.events.joint_armature.params["armature_distribution_params"] = (args.armature, args.armature)
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
arm = base.command_manager.get_term("arm_targets")
vel = base.command_manager.get_term("base_velocity")
robot = base.scene["robot"]
contacts = base.scene.sensors["contact_forces"]
arm._record_outcomes = lambda reached, missed: None
arm.update_levels = lambda env_ids, fell: None
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)

joint_ids = robot.find_joints(ALL_JOINTS_NAMES, preserve_order=True)[0]
foot_ids = robot.find_bodies(FEET, preserve_order=True)[0]
sensor_foot_ids = contacts.find_bodies(FEET, preserve_order=True)[0]


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


os.makedirs(args.out, exist_ok=True)
dirs = [os.path.join(args.out, f"env{i}") for i in range(args.num_envs)]
for d in dirs:
    os.makedirs(d, exist_ok=True)
tele = [open(os.path.join(d, "telemetry.jsonl"), "w") for d in dirs]
logs = [open(os.path.join(d, "commander.jsonl"), "w") for d in dirs]
falls = [[] for _ in dirs]
restarted = np.full(args.num_envs, -1e9)
dt = base.step_dt
steps = int(round(starts[-1] / dt))
r3 = lambda v: [round(float(x), 5) for x in v]
with torch.inference_mode():
    obs, _ = base.reset()
    states = base.state()
    ev_i = -1
    for step in range(steps):
        t = step * dt
        i = int(np.searchsorted(starts, t + 1e-9, side="right") - 1)
        if i != ev_i:
            ev_i = i
            for k, log in enumerate(logs):
                log.write(json.dumps({"t": round(t, 3), "type": "event", "kind": "walk", "index": i,
                                      "cmd": events[i]["cmd"], "duration": events[i]["duration"]}) + "\n")
        cmd = torch.tensor(events[ev_i]["cmd"], device=base.device)
        vel.vel_command_b[:] = cmd
        vel.time_left[:] = 1.0e6
        vel.is_standing_env[:] = bool(cmd.abs().max() < 1e-6)
        obs, _, terminated, truncated, _ = base.step(act(obs, states))
        states = base.state()
        done = (terminated["legs"] | truncated["legs"]).cpu().numpy()
        pos = robot.data.root_pos_w.cpu().numpy()
        quat = robot.data.root_quat_w.cpu().numpy()
        q = robot.data.joint_pos[:, joint_ids].cpu().numpy()
        qd = robot.data.joint_vel[:, joint_ids].cpu().numpy()
        tau = robot.data.applied_torque[:, joint_ids].cpu().numpy()
        feet = robot.data.body_pos_w[:, foot_ids].cpu().numpy()
        fz = contacts.data.net_forces_w[:, sensor_foot_ids, 2].cpu().numpy()
        for k in range(args.num_envs):
            if done[k]:
                falls[k].append({"t": round(t + dt, 3), "event": ev_i, "kind": "walk", "cmd": events[ev_i]["cmd"]})
                logs[k].write(json.dumps({"t": round(t + dt, 3), "type": "fall", "event": ev_i}) + "\n")
                restarted[k] = t + dt
            settling = t + dt - restarted[k] < SETTLE_AFTER_RESET_S
            tele[k].write(json.dumps({
                "sim_time": round(t + dt, 4), "pelvis_pos": r3(pos[k]), "pelvis_quat": r3(quat[k]),
                "q": r3(q[k]), "qd": r3(qd[k]), "tau": r3(tau[k]), "band": False,
                "feet": [r3(f) for f in feet[k]], "foot_force": [round(float(f), 1) for f in fz[k]],
                "cmd": events[ev_i]["cmd"], "mode": "engaged" if settling else "nav", "event": ev_i,
            }) + "\n")
for k, d in enumerate(dirs):
    tele[k].close()
    logs[k].close()
    with open(os.path.join(d, "segments.json"), "w") as f:
        json.dump({"variant": f"isaac_env{k}", "falls": falls[k], "estops": [], "passive_damping": args.passive_damping,
                   "armature": args.armature, "checkpoint": args.checkpoint, "task": args.task}, f, indent=1)
print(f"[eval_sim2sim_walk] wrote {args.num_envs} recordings to {args.out}; falls {[len(f) for f in falls]}")
sys.stdout.flush()
os._exit(0)
