"""Train the two-agent task with skrl's MAPPO.

    python scripts/skrl/train.py --headless [--max_iterations 12500] [--name <run>] [--seed 42]
        [--checkpoint <run>/checkpoints/agent_<N>.pt] [env.<field>=... agent.<field>=...]

--checkpoint resumes the agents and the env's curriculum state saved beside them. Arguments after these are Hydra
overrides of the env and agent configs; they are saved to <run>/params/hydra_overrides.txt, one per line, for the
evaluation scripts. When the run's last step is not a checkpoint step, a last checkpoint (and curriculum state) is
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
from locomanipulation_game.tasks.locomanip_marl.run_files import state_checkpoint_for  # noqa: E402


def bound_log_std(agent, policy_cfg: dict, per_agent: dict):
    """Clamp each agent's log_std parameter into [min_log_std, max_log_std], narrowed per agent to per_agent[uid]
    ([low, high]), now and after every update. skrl clamps log_std only in the forward pass; a parameter pushed past
    the bound then gets no gradient and the std stays pinned."""
    floor, ceiling = policy_cfg["min_log_std"], policy_cfg["max_log_std"]
    bounds = {uid: (floor, ceiling) for uid in agent.policies}
    for uid, (low, high) in per_agent.items():
        bounds[uid] = (max(low, floor), min(high, ceiling))
    print(f"[INFO] log_std bounds per agent: {bounds}")

    def project(uid):
        with torch.no_grad():
            agent.policies[uid].log_std_parameter.clamp_(*bounds[uid])

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
    env_cfg.sync_derived()  # after the overrides, so env.yaml records what the run uses
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "hydra_overrides.txt"), "w") as f:
        f.writelines(f"{arg}\n" for arg in hydra_args)
    env_cfg.log_dir = log_dir
    if args_cli.checkpoint:
        env_cfg.state_checkpoint_path = state_checkpoint_for(args_cli.checkpoint)

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = MarlRunner(env, agent_cfg)
    drop_value_preprocessors(runner.agent, agent_cfg["no_value_preprocessor"])
    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
    bound_log_std(runner.agent, agent_cfg["models"]["policy"], agent_cfg["log_std_bounds"])

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
