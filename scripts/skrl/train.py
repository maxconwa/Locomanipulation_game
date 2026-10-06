# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import faulthandler
import signal
import sys

from isaaclab.app import AppLauncher

# `kill -USR1 <pid>` prints every thread's Python stack to stderr (the run's log). Flat run A hung
# at step 22.5k (GPU and one worker thread at 100%, main thread in a futex wait, SIGTERM ignored)
# with no way to see where.
faulthandler.register(signal.SIGUSR1, all_threads=True)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--init_policies",
    type=str,
    default=None,
    help=(
        "Warm start: load only each agent's policy (and observation preprocessor) from this skrl checkpoint;"
        " critics, optimizers and the rest start fresh. For a new algorithm or critic input."
    ),
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help=(
        "Name of the RL algorithm to use (e.g. AMP, DDPG, IPPO, MAPPO, PPO, SAC, TD3, etc.) "
        "when several algorithms exist for the same task. For a more specific selection, use the argument --agent."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import locomanipulation_game.tasks  # noqa: F401
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def project_log_std_after_updates(agent, policy_cfg: dict, per_agent: dict | None = None):
    """Keep each multi-agent policy's log_std *parameter* inside [min_log_std, max_log_std].

    per_agent ({uid: [min, max]}, the agent config's log_std_bounds) narrows the
    bounds for one agent: skrl's model instantiator has one pair for all.

    skrl's GaussianMixin clamps log_std only in the forward pass. A parameter
    pushed past max_log_std (the entropy bonus does this) then gets zero
    gradient and stays there: the std is pinned at its maximum for good.
    LocoManip-Marl run 3 sat at std 1.0 for both agents for 1077 updates.
    Clamping the parameter after every update, and once now (a loaded
    checkpoint may be past the bound), gives the bounded std of the IBR
    runner's ClampedGaussianDistribution.
    """
    import torch

    if not hasattr(agent, "policies") or not policy_cfg.get("clip_log_std", True):
        return
    bounds = {uid: (policy_cfg["min_log_std"], policy_cfg["max_log_std"]) for uid in agent.policies}
    for uid, (low, high) in (per_agent or {}).items():
        bounds[uid] = (max(low, bounds[uid][0]), min(high, bounds[uid][1]))
    print(f"[INFO] log_std bounds per agent: {bounds}")

    def project(uid):
        parameter = getattr(agent.policies[uid], "log_std_parameter", None)
        if parameter is not None:
            with torch.no_grad():
                parameter.clamp_(*bounds[uid])

    for uid in agent.policies:
        project(uid)
    update = agent.update

    def update_and_project(*, timestep, timesteps, uid):
        update(timestep=timestep, timesteps=timesteps, uid=uid)
        project(uid)

    agent.update = update_and_project


def guard_updates(
    agent, log_dir: str, value_loss_alarm: float = 5.0, value_scale_alarm: float = 1.0e4, max_dumps: int = 3
):
    """Check every multi-agent update for divergence; on one, save the update's batch and say why.

    Flat run A: the arms' value loss went 0.57 -> 26 -> nan in two updates
    (twice, at the same point), then nan actions hung PhysX. The rewards were
    bounded throughout, so the cause is inside the update. Before each update
    this records the rollout memory's ranges and the preprocessors' smallest
    variances; after it, it checks the parameters. A non-finite parameter, a
    value scaler whose std passed value_scale_alarm, or a value-loss spike
    (above value_loss_alarm and 20x the median of the agent's last 50 updates;
    a fresh critic starts high) prints both and saves the memory and the
    pre-update weights to <log_dir>/divergence_<uid>_<n>.pt (the first
    max_dumps, ~150 MB each at 4096 envs). The first two then stop the run.
    """
    import torch

    if not hasattr(agent, "memories"):
        return
    from collections import deque

    update = agent.update
    alarms = {"count": 0}
    recent = {uid: deque(maxlen=50) for uid in agent.memories}

    def ranges(uid):
        memory = agent.memories[uid]
        out = {}
        for name in ("observations", "states", "actions", "rewards", "log_prob", "values"):
            t = memory.get_tensor_by_name(name)
            finite = torch.isfinite(t)
            tf = t[finite] if finite.any() else torch.zeros(1, device=t.device)
            out[name] = (tf.min().item(), tf.max().item(), int((~finite).sum()))
        for name in ("observation", "state", "value"):
            pre = getattr(agent, f"_{name}_preprocessor")[uid]
            if hasattr(pre, "running_variance"):
                var = pre.running_variance
                out[f"{name} scaler"] = (var.min().item(), var.max().item(), int((var < 1e-8).sum()))
        return out

    def update_and_check(*, timestep, timesteps, uid):
        before = ranges(uid)
        # the critic's predictions as stored in the rollout (un-normalised): the scale the advantages use
        agent.track_data(f"Value / rollout min ({uid})", before["values"][0])
        agent.track_data(f"Value / rollout max ({uid})", before["values"][1])
        weights = {
            "policy": {k: v.detach().clone() for k, v in agent.policies[uid].state_dict().items()},
            "value": {k: v.detach().clone() for k, v in agent.values[uid].state_dict().items()},
        }
        update(timestep=timestep, timesteps=timesteps, uid=uid)
        finite = all(
            torch.isfinite(p).all() for model in (agent.policies[uid], agent.values[uid]) for p in model.parameters()
        )
        losses = agent.tracking_data.get(f"Loss / Value loss ({uid})", [])
        value_loss = losses[-1] if losses else 0.0
        scaler = agent._value_preprocessor[uid]
        # a return scale this large is a feedback loop, not a task (flat run A: 6e9 by step 4.8k); without a
        # value preprocessor, the critic's raw predictions in the rollout say the same
        if hasattr(scaler, "running_variance"):
            runaway = scaler.running_variance.max().item() > value_scale_alarm**2
        else:
            runaway = max(abs(before["values"][0]), abs(before["values"][1])) > value_scale_alarm
        finite = finite and not runaway
        history = recent[uid]
        spike = (
            len(history) == history.maxlen
            and value_loss > value_loss_alarm
            and value_loss > 20.0 * sorted(history)[len(history) // 2]
        )
        history.append(value_loss)
        if finite and not spike:
            return
        alarms["count"] += 1
        if alarms["count"] > max_dumps and finite:
            return
        path = os.path.join(log_dir, f"divergence_{uid}_{alarms['count']}.pt")
        memory = agent.memories[uid]
        torch.save(
            {
                "timestep": timestep,
                "value_loss": value_loss,
                "ranges_before": before,
                "weights_before": weights,
                "memory": {n: memory.get_tensor_by_name(n).detach().cpu() for n in memory.get_tensor_names()},
            },
            path,
        )
        print(
            f"[DIVERGENCE] agent {uid} at timestep {timestep}: value loss {value_loss:.4g}, finite params {finite},"
            + (f" value scaler std {scaler.running_variance.max().sqrt().item():.4g}" if hasattr(scaler, "running_variance")
               else f" rollout values {before['values'][0]:.4g}..{before['values'][1]:.4g} (no value preprocessor)")
        )
        for name, (low, high, bad) in before.items():
            print(f"[DIVERGENCE]   {name:18s} min {low:.4g}  max {high:.4g}  non-finite / tiny-variance {bad}")
        print(f"[DIVERGENCE]   saved the update's memory and the weights before it to {path}")
        sys.stdout.flush()
        if not finite:
            raise RuntimeError(
                f"agent {uid}: non-finite parameters or a runaway value scale after the update at timestep {timestep};"
                f" see {path}"
            )

    agent.update = update_and_check


def drop_value_preprocessors(agent, uids) -> None:
    """Train these agents' critics on raw returns: no value preprocessor (skrl's yaml can't say it per agent).

    skrl's RunningStandardScaler for values is fitted on each update's returns
    *and on the critic's own un-normalised predictions*. Flat run B, with the
    arms' time-out bootstrap already off, still had the arms' value scale at
    mean -99, std 354 by step 4.8k, for returns that can only lie in about
    [-5, 13]: a poorly fitting critic widens the scale it is then measured in.
    rsl_rl (the IBR runs) never normalises values. Call before loading a
    checkpoint, so the dropped module is neither saved nor restored.
    """
    for uid in uids or []:
        agent._value_preprocessor[uid] = agent._empty_preprocessor
        agent.checkpoint_modules[uid].pop("value_preprocessor", None)
        print(f"[INFO] {uid}: no value preprocessor, the critic learns raw returns")


def load_policies_only(agent, path: str):
    """Load each agent's policy and observation preprocessor from a multi-agent skrl checkpoint.

    The policy's inputs (the agent's observations) are unchanged across IPPO
    and MAPPO; the critic's, its optimizer's and the state preprocessor's are
    not, so those start fresh.
    """
    import torch

    modules = torch.load(path, map_location=agent.device, weights_only=False)
    for uid in agent.possible_agents:
        saved = modules.get(uid, {})
        own = {name: agent.checkpoint_modules[uid].get(name) for name in ("policy", "observation_preprocessor")}
        # an agent whose observations or actions changed size starts fresh (e.g. the IK task's arms)
        fits = all(
            module is None or name not in saved or all(
                saved[name][k].shape == v.shape for k, v in module.state_dict().items() if k in saved[name]
            )
            for name, module in own.items()
        )
        if not fits:
            print(f"[INFO] Warm start: {uid} changed shape since {path}; it starts fresh")
            continue
        for name, module in own.items():
            if module is not None and name in saved:
                module.load_state_dict(saved[name])
                print(f"[INFO] Warm start: {uid}/{name} from {path}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    # LocoManip-Marl: the pelvis estimator (and curriculum state) is saved beside the skrl checkpoints, not in them
    warm_start = resume_path or (retrieve_file_path(args_cli.init_policies) if args_cli.init_policies else None)
    if warm_start and hasattr(env_cfg, "estimator"):
        env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(warm_start)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)
    drop_value_preprocessors(runner.agent, agent_cfg.get("no_value_preprocessor"))

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    elif args_cli.init_policies:
        load_policies_only(runner.agent, retrieve_file_path(args_cli.init_policies))
    project_log_std_after_updates(runner.agent, agent_cfg["models"]["policy"], agent_cfg.get("log_std_bounds"))
    guard_updates(runner.agent, log_dir)

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
