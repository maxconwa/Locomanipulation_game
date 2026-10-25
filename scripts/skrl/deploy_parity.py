"""Does GOLEM's lmg_deploy controller compute what the training env computes? Step by step, in Isaac.

Runs the run's env (its replayed overrides; sensor noise, action delay and pushes off) with the checkpoint's
deterministic actions. Every step it hands env 0's state to the deploy controller (MarlController, numpy +
pinocchio), synced to the env's command state (mode, arm command, velocity command and its low-pass, gait clock, last
actions, residual filter), and compares:
  observations   both agents', term by term
  networks       the exported legs.pt / arms.pt on the env's observation vs skrl's mean action (clamped)
  targets        the deploy joint targets for the env's action vs the targets the env applied
  gyro_lowpass   lmg_deploy's gyro low-pass on env 0's IMU samples vs the env's
and writes the worst differences per term to <out>/parity.json. The arm goals' odometry is FAST-LIO's on the robot and
the true motion here: not compared.

    python scripts/skrl/deploy_parity.py --checkpoint <run>/checkpoints/agent_<N>.pt --export <export dir> \
        --deploy_pkg ~/GOLEM/core_ws/src/lmg_deploy [--steps 600] [--out <dir>]
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--export", required=True, help="Folder with legs.pt, arms.pt, export.yaml and the deploy yaml.")
parser.add_argument("--config", default=None, help="Deploy yaml; default: the export folder's only *.yaml besides export.yaml.")
parser.add_argument("--deploy_pkg", default=os.path.expanduser("~/GOLEM/core_ws/src/lmg_deploy"))
parser.add_argument("--urdf", default=None, help="Default: the CL_Assets submodule's ros_assets/h1_2_magpie.urdf.")
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--arm_goal_prob", type=float, default=None, help="Share of events that are arm goals; default: the task's.")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.assets.h1_2 import CL_ASSETS_DIR  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    known_overrides,
    run_overrides,
    state_checkpoint_for,
)

sys.path.insert(0, args.deploy_pkg)
from lmg_deploy.controller import ARM_START, NUM_LEGS, MarlController, Mode, RobotState  # noqa: E402
from lmg_deploy.kinematics import ArmKinematics  # noqa: E402
from lmg_deploy.livox_imu import GyroLowpass as DeployGyroLowpass  # noqa: E402
from lmg_deploy.networks import load_network  # noqa: E402

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
replayed, dropped = known_overrides(run_overrides(RUN_DIR), args.task)
if dropped:
    print(f"[INFO] run overrides the current cfg no longer has (their values are the defaults now): {dropped}")
sys.argv = [sys.argv[0]] + replayed + hydra_args
torch.set_grad_enabled(False)


@hydra_task_config(args.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    cfg_path = args.config or [p for p in glob.glob(os.path.join(args.export, "*.yaml")) if os.path.basename(p) != "export.yaml"][0]
    cfg = yaml.safe_load(open(cfg_path))
    cfg.update(yaml.safe_load(open(os.path.join(args.export, "export.yaml"))) or {})
    names = cfg["joint_names"]
    urdf = args.urdf or str(CL_ASSETS_DIR / "ros_assets" / "h1_2_magpie.urdf")
    kin = ArmKinematics(urdf, names, cfg["arms"]["ee_frames"], [names[ARM_START:ARM_START + 7], names[ARM_START + 7:]],
                        extra_frames=[cfg["slam_frame"]])
    legs_net = load_network(os.path.join(args.export, cfg["legs"]["policy"]), cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])
    arms_net = load_network(os.path.join(args.export, cfg["arms"]["policy"]), cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
    ctrl = MarlController(cfg, kin, legs_net, arms_net)
    out_dir = args.out or os.path.join(args.export, "parity")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.log_dir = RUN_DIR
    env_cfg.training = False
    env_cfg.state_checkpoint_path = state_checkpoint_for(args.checkpoint)
    env_cfg.events.push_robot = None
    env_cfg.events.imu_mount = None  # the deploy reads the same, unperturbed sensor
    if args.arm_goal_prob is not None:
        env_cfg.commands.arm_targets.arm_goal_prob = args.arm_goal_prob
    for group in ("legs", "arms"):
        getattr(env_cfg.observations, group).enable_corruption = False
    env_cfg.action_delay_range = None  # the targets the env applies are the step's own
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    base = env.unwrapped
    robot, imu = base.scene["robot"], base.scene["imu"]
    arm, vel = base.command_manager.get_term("arm_targets"), base.command_manager.get_term("base_velocity")
    legs_term, arms_term = base.action_manager.get_term("joint_pos"), base.action_manager.get_term("arm_pos")
    ids = robot.find_joints(names, preserve_order=True)[0]
    runner = MarlRunner(env, agent_cfg)
    runner.agent.load(args.checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    # observation terms, in order, with their sizes, as the env builds them
    manager = base.observation_manager
    LEGS, ARMS = ([(n, d[0]) for n, d in zip(manager.active_terms[g], manager.group_obs_term_dim[g])] for g in ("legs", "arms"))
    worst = {f"legs/{n}": 0.0 for n, _ in LEGS} | {f"arms/{n}": 0.0 for n, _ in ARMS}
    worst |= {"net/legs": 0.0, "net/arms": 0.0, "target/legs": 0.0, "target/arms": 0.0, "imu/gyro_lowpass": 0.0}

    class DeployGyro:
        """The env's gyro low-pass wrapped: env 0's raw IMU samples also go through lmg_deploy's GyroLowpass (as
        marl_deploy_node filters /livox/imu), whose output the controller gets as RobotState.gyro."""

        def __init__(self, env_filter):
            self.inner = env_filter
            self.deploy = DeployGyroLowpass(env_cfg.gyro_lowpass_hz)
            self.value = None

        @property
        def output(self):
            return self.inner.output

        def reset(self, env_ids=None):
            self.inner.reset(env_ids)
            if env_ids is None or 0 in torch.as_tensor(env_ids).tolist():
                self.deploy._x = None  # restarts at (u, 0) on the next sample, as after a dropout

        def __call__(self, u):
            out = self.inner(u)
            self.value = self.deploy(u[0].double().cpu().numpy(), base.physics_dt)
            worst["imu/gyro_lowpass"] = max(worst["imu/gyro_lowpass"],
                                            float(np.abs(self.value - out[0].double().cpu().numpy()).max()))
            return out

    base.gyro_lowpass = deploy_gyro = DeployGyro(base.gyro_lowpass)

    def env_state(i: int = 0) -> RobotState:
        d = robot.data
        gyro = deploy_gyro.value if deploy_gyro.value is not None else imu.data.ang_vel_b[i].double().cpu().numpy()
        return RobotState(q=d.joint_pos[i, ids].double().cpu().numpy(), dq=d.joint_vel[i, ids].double().cpu().numpy(),
                          tau=d.applied_torque[i, ids].double().cpu().numpy(), quat=imu.data.quat_w[i].double().cpu().numpy(),
                          gyro=gyro, acc=imu.data.lin_acc_b[i].double().cpu().numpy())

    def sync(i: int = 0):
        ctrl.mode = Mode.ARM if bool(arm.arm_mode[i]) else Mode.NAV
        ctrl.believed = arm.believed_b[i].double().cpu().numpy().copy()
        ctrl.cmd_vel = vel.vel_command_b[i, :3].double().cpu().numpy().copy()
        ctrl.cmd_filtered = vel.command[i, :3].double().cpu().numpy().copy()
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
    for _ in range(args.steps):
        state = env_state()
        sync()
        kin.update(state.q)
        common = ctrl._common_obs(state)
        compare("legs", LEGS, np.concatenate([common, ctrl.legs_last]), obs["legs"][0].double().cpu().numpy())
        compare("arms", ARMS, np.concatenate([common, ctrl.arms_last, ctrl._wrist_poses(), ctrl._wrist_errors()]),
                obs["arms"][0].double().cpu().numpy())

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

        obs, _, terminated, truncated, _ = base.step(actions)
        states = base.state()
        applied = robot.data.joint_pos_target[0, ids].double().cpu().numpy()
        worst["target/legs"] = max(worst["target/legs"], float(np.abs(target[:NUM_LEGS] - applied[:NUM_LEGS]).max()))
        worst["target/arms"] = max(worst["target/arms"], float(np.abs(target[ARM_START:] - applied[ARM_START:]).max()))
        if bool(terminated["legs"][0] | truncated["legs"][0]):
            legs_net.reset(), arms_net.reset()
        runner.agent.reset_rnn(terminated["legs"] | truncated["legs"])  # every env's, as training zeroed them

    result = {"checkpoint": args.checkpoint, "export": args.export, "steps": args.steps,
              "physics": {"dt": env_cfg.sim.dt, "decimation": env_cfg.decimation},
              "worst_abs_diff": {k: round(v, 6) for k, v in worst.items()}}
    print(json.dumps(result, indent=1))
    json.dump(result, open(os.path.join(out_dir, "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
    os._exit(0)
