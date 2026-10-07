"""Train the two-agent task with skrl's MAPPO.

    python scripts/skrl/train.py --headless [--max_iterations 3800] [--checkpoint <run>/checkpoints/agent_<N>.pt]

--checkpoint resumes the agents and the env's pelvis estimator and curriculum state saved beside them. Arguments
after these are Hydra overrides of the env and agent configs (env.<field>=..., agent.<field>=...).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="skrl checkpoint to resume from.")
parser.add_argument("--max_iterations", type=int, default=None, help="Policy updates (rollouts of 24 steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.direct.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402


def bound_log_std(agent, policy_cfg: dict, per_agent: dict):
    """Clamp each agent's log_std parameter into [min_log_std, max_log_std], narrowed per agent by per_agent, now
    and after every update. skrl clamps log_std only in the forward pass; a parameter pushed past the bound then gets
    no gradient and the std stays pinned."""
    bounds = {uid: (policy_cfg["min_log_std"], policy_cfg["max_log_std"]) for uid in agent.policies}
    for uid, (low, high) in per_agent.items():
        bounds[uid] = (max(low, bounds[uid][0]), min(high, bounds[uid][1]))
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
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mappo_torch"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Logging to {log_dir}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    env_cfg.log_dir = log_dir
    if args_cli.checkpoint:
        env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args_cli.checkpoint)

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = Runner(env, agent_cfg)
    drop_value_preprocessors(runner.agent, agent_cfg["no_value_preprocessor"])
    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
    bound_log_std(runner.agent, agent_cfg["models"]["policy"], agent_cfg["log_std_bounds"])

    start_time = time.time()
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
