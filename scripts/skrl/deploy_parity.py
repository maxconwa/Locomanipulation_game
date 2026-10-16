"""Does GOLEM's lmg_deploy controller (locomotion_game_deploy before its rename) compute what the training env computes? Step by step, in Isaac.

Runs the training env (sensor noise off) with the checkpoint's deterministic actions. Every step it hands env 0's
state to the deploy controller (MarlController, numpy + pinocchio), synced to the env's command state (mode, arm
command, velocity command, gait clock, last actions, residual filter), and compares:
  observations   both agents', term by term
  networks       the exported legs.pt / arms.pt on the env's observation vs skrl's mean action (clamped)
  targets        the deploy joint targets for the env's action vs the targets the env applied
  odometry       the deploy leg odometry (IMU attitude + feet within stance_height of the lowest) vs the true
                 pelvis motion over each arm-goal step (m/s, rad/s); with learned odometry, the deploy estimator
                 input (window term by term + actions) and its motion after vertical_from_feet / tilt_from_gravity
                 vs the env's, and vs the true motion
and writes the worst differences per term to <out>/parity.json. The env runs the checkpoint's physics (sim dt,
decimation, explicit PD from its params/env.yaml) without the action delay, sensor noise or pushes; LSTM actors are
stepped alongside skrl's and both are reset when the episode ends.

    python scripts/skrl/deploy_parity.py --checkpoint <agent.pt> --export <GOLEM pkg>/policies/<policy> \
        --deploy_pkg <GOLEM>/core_ws/src/lmg_deploy [--task ...] [--steps 600] [--out <dir>]
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
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, axis_angle_from_quat  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402

sys.path.insert(0, args.deploy_pkg)
import importlib  # noqa: E402

# GOLEM's package: lmg_deploy (camera_ready from d932cb3), before that locomotion_game_deploy
PKG = "lmg_deploy" if os.path.isdir(os.path.join(args.deploy_pkg, "lmg_deploy")) else "locomotion_game_deploy"
_controller = importlib.import_module(f"{PKG}.controller")
ARM_START, NUM_LEGS, MarlController, Mode, RobotState = (getattr(_controller, n) for n in
                                                         ("ARM_START", "NUM_LEGS", "MarlController", "Mode", "RobotState"))
ArmKinematics = importlib.import_module(f"{PKG}.kinematics").ArmKinematics
_networks = importlib.import_module(f"{PKG}.networks")
TorchScriptNetwork, load_network = _networks.TorchScriptNetwork, _networks.load_network
_rot = importlib.import_module(f"{PKG}.rotations")
qconj, qmul, aa = _rot.quat_conjugate, _rot.quat_mul, _rot.axis_angle_from_quat

torch.set_grad_enabled(False)
cfg_path = args.config or [p for p in glob.glob(os.path.join(args.export, "*.yaml")) if not p.endswith("export.yaml")][0]
cfg = yaml.safe_load(open(cfg_path))
export_yaml = os.path.join(args.export, "export.yaml")
if os.path.isfile(export_yaml):
    cfg.update(yaml.safe_load(open(export_yaml)) or {})
names = cfg["joint_names"]
odo = cfg.get("odometry")
learned = odo == "learned" or (isinstance(odo, dict) and odo.get("mode", "learned") == "learned")
extra = list(cfg.get("legs_odometry", {}).get("foot_frames", []))
imu_frame = cfg.get("legs_odometry", {}).get("imu_frame", "pelvis")
if imu_frame != "pelvis":
    extra.append(imu_frame)
kin = ArmKinematics(args.urdf, names, cfg["arms"]["ee_frames"], [names[ARM_START:ARM_START + 7], names[ARM_START + 7:]],
                    extra_frames=extra)
legs_net = load_network(os.path.join(args.export, cfg["legs"]["policy"]), cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])
arms_net = load_network(os.path.join(args.export, cfg["arms"]["policy"]), cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
est_net = TorchScriptNetwork(os.path.join(args.export, cfg["estimator"]["path"]), cfg["estimator"]["num_inputs"], 6)
ctrl = MarlController(cfg, kin, legs_net, arms_net, est_net)

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
out_dir = args.out or os.path.join(args.export, "parity")
os.makedirs(out_dir, exist_ok=True)
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
run_env = open(os.path.join(RUN_DIR, "params", "env.yaml")).read()
import re  # noqa: E402
env_cfg.sim.dt = float(re.search(r"^  dt: (\S+)$", run_env, re.M).group(1))
env_cfg.decimation = int(re.search(r"^decimation: (\d+)$", run_env, re.M).group(1))
m = re.search(r"^explicit_pd: (true|false)$", run_env, re.M)
env_cfg.explicit_pd = m is None or m.group(1) == "true"  # runs before the field: explicit from almitrain on
env_cfg.events.push_robot = None
m = re.search(r"^\s*table_file: (\S+)$", run_env, re.M)
if m:
    env_cfg.commands.arm_targets.table_file = m.group(1)  # the tables the run trained with (absolute or in RUN_DIR)
agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.log_dir = RUN_DIR
env_cfg.estimator.train = False
env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args.checkpoint)
if args.arm_goal_prob is not None:
    env_cfg.commands.arm_targets.arm_goal_prob = args.arm_goal_prob
for group in ("legs", "arms", "odometry"):
    getattr(env_cfg.observations, group).enable_corruption = False
env_cfg.terminations.golem_estop = None  # parity is about the computation; keep episodes running
env_cfg.action_delay_range = None  # the targets the env applies are the step's own
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
robot, imu = base.scene["robot"], base.scene["imu"]
arm = base.command_manager.get_term("arm_targets")
legs_term, arms_term = base.action_manager.get_term("joint_pos"), base.action_manager.get_term("arm_pos")
ids = robot.find_joints(names, preserve_order=True)[0]
runner = MarlRunner(env, agent_cfg)
runner.agent.load(args.checkpoint)
runner.agent.enable_training_mode(False, apply_to_models=True)

# observation terms, in order, with their sizes, as the env builds them
obs_manager = base.observation_manager
LEGS, ARMS = ([(n, d[0]) for n, d in zip(obs_manager.active_terms[g], obs_manager.group_obs_term_dim[g])] for g in ("legs", "arms"))
worst = {f"legs/{n}": 0.0 for n, _ in LEGS} | {f"arms/{n}": 0.0 for n, _ in ARMS}
worst |= {"net/legs": 0.0, "net/arms": 0.0, "target/legs": 0.0, "target/arms": 0.0}
odo_lin, odo_ang = [], []
est_lin, est_ang, est_vs_env = [], [], []
cap = {}
if learned:
    worst |= {"estimator/input": 0.0, "estimator/motion_pos": 0.0, "estimator/motion_rot": 0.0}
    model = base.estimator.model

    class Capture(torch.nn.Module):
        def forward(self, x):
            y = model(x)
            cap["input"] = x[0].double().cpu().numpy()
            return y

    base.estimator.model = Capture()
    if base._vertical or base._tilt:
        corrected = base._from_feet_and_gravity

        def from_feet_and_gravity(*a, **k):
            out = corrected(*a, **k)
            cap["motion"] = (out[0][0].double().cpu().numpy(), out[1][0].double().cpu().numpy())
            return out

        base._from_feet_and_gravity = from_feet_and_gravity


def env_state(i: int = 0) -> RobotState:
    d = robot.data
    return RobotState(q=d.joint_pos[i, ids].double().cpu().numpy(), dq=d.joint_vel[i, ids].double().cpu().numpy(),
                      tau=d.applied_torque[i, ids].double().cpu().numpy(), quat=imu.data.quat_w[i].double().cpu().numpy(),
                      gyro=imu.data.ang_vel_b[i].double().cpu().numpy(), acc=imu.data.lin_acc_b[i].double().cpu().numpy())


def sync(i: int = 0):
    if bool(arm.arm_mode[i]):
        ctrl.mode = Mode.ARM
    elif hasattr(arm, "settling") and bool(arm.settling[i]):
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
worst["rest_pose"] = float(np.abs(ctrl.rest_pose - arm.rest_pose_b.double().cpu().numpy()).max())
prev_root = None
if learned:  # the env pushed its reset frame into the estimator window; so does the controller, as at engage
    first = env_state()
    kin.update(first.q)
    ctrl._learned_motion(first)
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

    # deploy odometry over this step vs the truth (only while an arm goal holds the command)
    done = bool(terminated["legs"][0] | truncated["legs"][0])
    new_state = env_state()
    kin.update(new_state.q)
    if learned:
        ctrl._prev_action = np.concatenate([a_legs, a_arms])
        motion = ctrl._learned_motion(new_state)
        if motion is not None and not done and "input" in cap:
            diff = np.abs(ctrl.last_estimator_input - cap["input"])
            worst["estimator/input"] = max(worst["estimator/input"], float(diff.max()))
            k = 0
            for term, size in (("lin_acc", 3), ("ang_vel", 3), ("gravity", 3), ("joint_pos", 27), ("joint_vel", 27),
                               ("leg_torques", 12)):
                n = size * ctrl.history
                key = f"estimator/input/{term}"
                worst[key] = max(worst.get(key, 0.0), float(diff[k:k + n].max()))
                k += n
            worst["estimator/input/actions"] = max(worst.get("estimator/input/actions", 0.0), float(diff[k:].max()))
            if "motion" in cap:
                worst["estimator/motion_pos"] = max(worst["estimator/motion_pos"], float(np.abs(motion[0] - cap["motion"][0]).max()))
                rot = np.linalg.norm(aa(qmul(qconj(cap["motion"][1]), motion[1])))
                worst["estimator/motion_rot"] = max(worst["estimator/motion_rot"], float(rot))
    else:
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
        ctrl._frames, ctrl._prev_action, ctrl._prev_up, ctrl._prev_height = [], None, None, None
        legs_net.reset(), arms_net.reset()
        if learned:  # the reset env's first frame (the env's observations after the step are the new episode's)
            ctrl._learned_motion(env_state())
    if hasattr(runner.agent, "reset_rnn"):
        runner.agent.reset_rnn(terminated["legs"] | truncated["legs"])  # every env's, as training zeroed them
    prev_root = root

result = {"checkpoint": args.checkpoint, "export": args.export, "steps": args.steps,
          "odometry": "learned" if learned else "legs",
          "physics": {"dt": env_cfg.sim.dt, "decimation": env_cfg.decimation, "explicit_pd": env_cfg.explicit_pd},
          "worst_abs_diff": {k: round(v, 6) for k, v in worst.items()},
          "odometry_error_arm_goals": {"lin_m_per_s_p50_p90": [round(float(v), 4) for v in np.percentile(odo_lin, [50, 90])] if odo_lin else None,
                                       "ang_rad_per_s_p50_p90": [round(float(v), 4) for v in np.percentile(odo_ang, [50, 90])] if odo_ang else None,
                                       "samples": len(odo_lin)}}
print(json.dumps(result, indent=1))
json.dump(result, open(os.path.join(out_dir, "parity.json"), "w"), indent=1)
os._exit(0)
