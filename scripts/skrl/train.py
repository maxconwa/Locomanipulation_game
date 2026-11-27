"""Train the two-agent task with skrl's MAPPO.

    python scripts/skrl/train.py --headless [--max_iterations 12500] [--name <run>] [--seed 42]
        [--checkpoint <run>/checkpoints/agent_<N>.pt] [env.<field>=... agent.<field>=...]

--checkpoint resumes the agents and the env's curriculum state saved beside them. Arguments after these are Hydra
overrides of the env and agent configs; they are saved to <run>/params/hydra_overrides.txt, one per line, and the
task's id to <run>/params/task.txt, for the evaluation and export scripts. An override of a field that the env cfg's
sync_derived writes would be lost without a word: the run is refused. When the run's last step is not a checkpoint step, a last checkpoint (and curriculum state) is
saved at the end.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="skrl checkpoint to resume from.")
parser.add_argument("--max_iterations", type=int, default=None, help="Policy updates (of agent.rollouts steps each).")
parser.add_argument("--name", default="mappo_torch", help="Run directory suffix: <time>_<name>.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    cfg_value,
    env_override_keys,
    state_checkpoint_for,
)


def bound_log_std(agent, bounds: dict):
    """Clamp each agent's log_std parameter into bounds[uid] ((low, high): MarlRunner.log_std_bounds), now and after
    every update. skrl clamps log_std only in the forward pass; a parameter pushed past the bound then gets no
    gradient and the std stays pinned."""
    print(f"[INFO] log_std bounds per agent: {bounds}")

    def project(uid):
        low, high = bounds[uid]
        with torch.no_grad():
            agent.policies[uid].log_std_parameter.clamp_(low, high)

    for uid in agent.policies:
        project(uid)
    update = agent.update

    def update_and_project(*, timestep, timesteps, uid):
        update(timestep=timestep, timesteps=timesteps, uid=uid)
        project(uid)

    agent.update = update_and_project


def drop_value_preprocessors(agent, uids):
    """Train these agents' critics on raw returns. Before loading a checkpoint, so the module is neither saved nor
    restored."""
    for uid in uids:
        agent._value_preprocessor[uid] = agent._empty_preprocessor
        agent.checkpoint_modules[uid].pop("value_preprocessor", None)
        print(f"[INFO] {uid}: no value preprocessor, the critic learns raw returns")


def scale_policy_outputs(agent, scales: dict):
    """Scale each listed agent's fresh actor's last layer (env_cfg.policy_output_init_scale): weights times the scale,
    bias zeroed. Before loading a checkpoint, which replaces them."""
    for uid, scale in scales.items():
        last = agent.policies[uid].net[-1]
        with torch.no_grad():
            last.weight.mul_(scale)
            last.bias.zero_()
        print(f"[INFO] {uid}: actor's last layer scaled by {scale} at initialisation, its bias zeroed")


def set_initial_std(agent, stds: dict):
    """Start each listed agent's fresh actor's action std at stds[uid] (env_cfg.policy_initial_std) instead of the
    yaml's initial_log_std. Before loading a checkpoint, which replaces it."""
    for uid, std in stds.items():
        with torch.no_grad():
            agent.policies[uid].log_std_parameter.fill_(math.log(std))
        print(f"[INFO] {uid}: action std {std} at initialisation")


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.name}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Logging to {log_dir}")
    overridden = {key: cfg_value(env_cfg, key) for key in env_override_keys(hydra_args)}
    # the curriculum state saved at the checkpoints' steps: an evaluated checkpoint starts from its own step's
    env_cfg.state_save_every = agent_cfg["agent"]["experiment"]["checkpoint_interval"] or env_cfg.state_save_every
    env_cfg.sync_derived()  # after the overrides, so env.yaml records what the run uses
    lost = [key for key, value in overridden.items() if cfg_value(env_cfg, key) != value]
    if lost:
        print(f"[ERROR] These overrides are of fields that follow from others (the env cfg's sync_derived;"
              f" state_save_every from the agent's checkpoint_interval), which would replace them: {lost}."
              f" Set the fields they follow from instead.", flush=True)
        os._exit(2)  # (a SystemExit here left the process's exit code at 0: the app's shutdown)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "hydra_overrides.txt"), "w") as f:
        f.writelines(f"{arg}\n" for arg in hydra_args)
    with open(os.path.join(log_dir, "params", "task.txt"), "w") as f:
        f.write(args_cli.task + "\n")
    env_cfg.log_dir = log_dir
    if args_cli.checkpoint:
        env_cfg.state_checkpoint_path = state_checkpoint_for(args_cli.checkpoint)

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = MarlRunner(env, agent_cfg)
    drop_value_preprocessors(runner.agent, agent_cfg["no_value_preprocessor"])
    scale_policy_outputs(runner.agent, env_cfg.policy_output_init_scale)
    set_initial_std(runner.agent, env_cfg.policy_initial_std)
    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
    bound_log_std(runner.agent, runner.log_std_bounds)

    start_time = time.time()
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    timesteps, interval = agent_cfg["trainer"]["timesteps"], agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    if interval > 0 and timesteps % interval:
        runner.agent.write_checkpoint(timestep=timesteps, timesteps=timesteps)
        env.unwrapped.save_state()
        print(f"[INFO] Saved the last checkpoint, agent_{timesteps}.pt")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
