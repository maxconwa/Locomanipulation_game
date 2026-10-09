"""Does GOLEM's locomotion_game_deploy controller compute what the training env computes? Step by step, in Isaac.

Runs the training env (sensor noise off) with the checkpoint's deterministic actions. Every step it hands env 0's
state to the deploy controller (MarlController, numpy + pinocchio), synced to the env's command state (mode, arm
command, velocity command, gait clock, last actions, residual filter), and compares:
  observations   both agents', term by term
  networks       the exported legs.pt / arms.pt on the env's observation vs skrl's mean action (clamped)
  targets        the deploy joint targets for the env's action vs the targets the env applied
  odometry       the deploy leg odometry (IMU attitude + feet within stance_height of the lowest) vs the true
                 pelvis motion over each arm-goal step (m/s, rad/s)
and writes the worst differences per term to <out>/parity.json.

    python scripts/skrl/deploy_parity.py --checkpoint <agent.pt> --export <GOLEM pkg>/policies/marl_golem \
        --deploy_pkg <GOLEM>/core_ws/src/locomotion_game_deploy [--task ...] [--steps 600] [--out <dir>]
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--export", required=True, help="Folder with legs.pt, arms.pt, estimator.pt, export.yaml and the deploy yaml.")
parser.add_argument("--config", default=None, help="Deploy yaml; default: the export folder's only *.yaml besides export.yaml.")
parser.add_argument("--deploy_pkg", required=True)
parser.add_argument("--urdf", default="/home/max/GOLEM/CL_Assets/ros_assets/h1_2_magpie.urdf")
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--arm_goal_prob", type=float, default=None, help="Share of events that are arm goals; default: the task's.")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, axis_angle_from_quat  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

sys.path.insert(0, args.deploy_pkg)
from locomotion_game_deploy.controller import ARM_START, NUM_LEGS, MarlController, Mode, RobotState  # noqa: E402
from locomotion_game_deploy.kinematics import ArmKinematics  # noqa: E402
from locomotion_game_deploy.networks import TorchScriptNetwork  # noqa: E402
from locomotion_game_deploy.rotations import axis_angle_from_quat as aa  # noqa: E402

torch.set_grad_enabled(False)
cfg_path = args.config or [p for p in glob.glob(os.path.join(args.export, "*.yaml")) if not p.endswith("export.yaml")][0]
cfg = yaml.safe_load(open(cfg_path))
export_yaml = os.path.join(args.export, "export.yaml")
if os.path.isfile(export_yaml):
    cfg.update(yaml.safe_load(open(export_yaml)) or {})
names = cfg["joint_names"]
extra = list(cfg.get("legs_odometry", {}).get("foot_frames", []))
imu_frame = cfg.get("legs_odometry", {}).get("imu_frame", "pelvis")
if imu_frame != "pelvis":
    extra.append(imu_frame)
kin = ArmKinematics(args.urdf, names, cfg["arms"]["ee_frames"], [names[ARM_START:ARM_START + 7], names[ARM_START + 7:]],
                    extra_frames=extra)
legs_net = TorchScriptNetwork(os.path.join(args.export, cfg["legs"]["policy"]), cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])
arms_net = TorchScriptNetwork(os.path.join(args.export, cfg["arms"]["policy"]), cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
est_net = TorchScriptNetwork(os.path.join(args.export, cfg["estimator"]["path"]), cfg["estimator"]["num_inputs"], 6)
ctrl = MarlController(cfg, kin, legs_net, arms_net, est_net)

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
out_dir = args.out or os.path.join(args.export, "parity")
os.makedirs(out_dir, exist_ok=True)
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
if args.arm_goal_prob is not None:
    env_cfg.commands.arm_targets.arm_goal_prob = args.arm_goal_prob
for group in ("legs", "arms"):
    getattr(env_cfg.observations, group).enable_corruption = False
env_cfg.terminations.golem_estop = None  # parity is about the computation; keep episodes running
env_cfg.action_delay_substeps = (0, 0)  # the targets the env applies are the step's own
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
robot, imu = base.scene["robot"], base.scene["imu"]
arm = base.command_manager.get_term("arm_targets")
legs_term, arms_term = base.action_manager.get_term("joint_pos"), base.action_manager.get_term("arm_pos")
ids = robot.find_joints(names, preserve_order=True)[0]
runner = Runner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)

# observation terms, in order, with their sizes, as the env builds them
obs_manager = base.observation_manager
LEGS, ARMS = ([(n, d[0]) for n, d in zip(obs_manager.active_terms[g], obs_manager.group_obs_term_dim[g])] for g in ("legs", "arms"))
worst = {f"legs/{n}": 0.0 for n, _ in LEGS} | {f"arms/{n}": 0.0 for n, _ in ARMS}
worst |= {"net/legs": 0.0, "net/arms": 0.0, "target/legs": 0.0, "target/arms": 0.0}
odo_lin, odo_ang = [], []


def env_state(i: int = 0) -> RobotState:
    d = robot.data
    return RobotState(q=d.joint_pos[i, ids].double().cpu().numpy(), dq=d.joint_vel[i, ids].double().cpu().numpy(),
                      tau=d.applied_torque[i, ids].double().cpu().numpy(), quat=imu.data.quat_w[i].double().cpu().numpy(),
                      gyro=imu.data.ang_vel_b[i].double().cpu().numpy(), acc=imu.data.lin_acc_b[i].double().cpu().numpy())


def sync(i: int = 0):
    if bool(arm.arm_mode[i]):
        ctrl.mode = Mode.ARM
    elif bool(arm.settling[i]):
        ctrl.mode = Mode.SETTLE_TO_ARM if bool(arm.pending_arm[i]) else Mode.SETTLE_TO_NAV
    else:
        ctrl.mode = Mode.NAV
    ctrl.believed = arm.believed_b[i].double().cpu().numpy().copy()
    ctrl.cmd_vel = base.command_manager.get_command("base_velocity")[i, :3].double().cpu().numpy().copy()
    ctrl.steps = int(base.episode_length_buf[i])
    ctrl.legs_last = legs_term.applied_actions[i].double().cpu().numpy().copy()
    ctrl.arms_last = arms_term.applied_actions[i].double().cpu().numpy().copy()
    ctrl.residual = arms_term._residual[i].double().cpu().numpy().copy()


def compare(prefix, terms, deploy, trained):
    k = 0
    for name, size in terms:
        diff = float(np.abs(deploy[k:k + size] - trained[k:k + size]).max())
        worst[f"{prefix}/{name}"] = max(worst[f"{prefix}/{name}"], diff)
        k += size


def act(obs, states):
    outputs = runner.agent.act(obs, {a: states for a in base.possible_agents}, timestep=0, timesteps=0)
    return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in base.possible_agents}


obs, _ = base.reset()
states = base.state()
prev_root = None
for step in range(args.steps):
    state = env_state()
    sync()
    kin.update(state.q)
    common = ctrl._common_obs(state)
    legs_obs = np.concatenate([common, ctrl.legs_last])
    arms_obs = np.concatenate([common, ctrl.arms_last, ctrl._wrist_poses(), ctrl._wrist_errors()])
    compare("legs", LEGS, legs_obs, obs["legs"][0].double().cpu().numpy())
    compare("arms", ARMS, arms_obs, obs["arms"][0].double().cpu().numpy())

    actions = act(obs, states)
    a_legs = actions["legs"][0].double().clamp(-10, 10).cpu().numpy()
    a_arms = actions["arms"][0].double().clamp(-10, 10).cpu().numpy()
    worst["net/legs"] = max(worst["net/legs"], float(np.abs(legs_net(obs["legs"][0].cpu().numpy()) - a_legs).max()))
    worst["net/arms"] = max(worst["net/arms"], float(np.abs(arms_net(obs["arms"][0].cpu().numpy()) - a_arms).max()))

    # the deploy action mapping for the env's actions, from this state
    target = ctrl.default.copy()
    legs = slice(0, NUM_LEGS)
    target[legs] = ctrl._torque_bound(ctrl._bound(a_legs * ctrl.legs_scale + ctrl.default[legs], legs), state, legs)
    ik = state.q[ARM_START:].copy()
    for a in range(ctrl.num_arms):
        cols = kin.arm_cols[a] - ARM_START
        ik[cols] += kin.ik_step(a, ctrl.believed[a, :3], ctrl.believed[a, 3:], ctrl.ik_damping, ctrl.max_ik_step)
    residual = ctrl.residual + ctrl.filter_alpha * (a_arms - ctrl.residual)
    arms = slice(ARM_START, len(names))
    target[arms] = ctrl._torque_bound(ctrl._bound(ik + residual * ctrl.arms_scale, arms), state, arms)

    root = (robot.data.root_pos_w[0].clone(), robot.data.root_quat_w[0].clone())
    obs, _, terminated, truncated, _ = base.step(actions)
    states = base.state()
    applied = robot.data.joint_pos_target[0, ids].double().cpu().numpy()
    worst["target/legs"] = max(worst["target/legs"], float(np.abs(target[:NUM_LEGS] - applied[:NUM_LEGS]).max()))
    worst["target/arms"] = max(worst["target/arms"], float(np.abs(target[ARM_START:] - applied[ARM_START:]).max()))

    # deploy leg odometry over this step vs the truth (only while an arm goal holds the command)
    done = bool(terminated["legs"][0] | truncated["legs"][0])
    new_state = env_state()
    kin.update(new_state.q)
    motion = ctrl._leg_odometry(new_state)
    if motion is not None and not done and bool(arm.arm_mode[0]) and prev_root is not None:
        pos0, quat0 = root
        pos1, quat1 = robot.data.root_pos_w[0], robot.data.root_quat_w[0]
        true_dp = quat_apply_inverse(quat0[None], (pos1 - pos0)[None])[0].double().cpu().numpy()
        true_dq = axis_angle_from_quat(quat_mul(quat_inv(quat0[None]), quat1[None]))[0].double().cpu().numpy()
        odo_lin.append(float(np.linalg.norm(motion[0] - true_dp)) / base.step_dt)
        odo_ang.append(float(np.linalg.norm(aa(motion[1]) - true_dq)) / base.step_dt)
    if done:
        ctrl._prev_pelvis_quat = ctrl._prev_feet = None
    prev_root = root

result = {"checkpoint": args.checkpoint, "export": args.export, "steps": args.steps,
          "worst_abs_diff": {k: round(v, 6) for k, v in worst.items()},
          "odometry_error_arm_goals": {"lin_m_per_s_p50_p90": [round(float(v), 4) for v in np.percentile(odo_lin, [50, 90])] if odo_lin else None,
                                       "ang_rad_per_s_p50_p90": [round(float(v), 4) for v in np.percentile(odo_ang, [50, 90])] if odo_ang else None,
                                       "samples": len(odo_lin)}}
print(json.dumps(result, indent=1))
json.dump(result, open(os.path.join(out_dir, "parity.json"), "w"), indent=1)
os._exit(0)
