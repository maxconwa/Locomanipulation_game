"""Play a checkpoint of the two-agent task with deterministic actions.

    python scripts/skrl/play.py --checkpoint <run>/checkpoints/agent_<N>.pt [--num_envs 16] [env.<field>=...]

Builds the env the run trained in (its saved overrides, those the current cfg still has, then any given here) and
loads the curriculum state saved beside the checkpoint.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.run_files import (  # noqa: E402
    known_overrides,
    run_overrides,
    state_checkpoint_for,
)

checkpoint = os.path.abspath(args_cli.checkpoint)
run_dir = os.path.dirname(os.path.dirname(checkpoint))
replayed, dropped = known_overrides(run_overrides(run_dir), args_cli.task)
if dropped:
    print(f"[INFO] run overrides the current cfg no longer has (their values are the defaults now): {dropped}")
sys.argv = [sys.argv[0]] + replayed + hydra_args


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.log_dir = run_dir
    env_cfg.training = False
    env_cfg.state_checkpoint_path = state_checkpoint_for(checkpoint)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = MarlRunner(env, agent_cfg)
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)
    obs, _ = env.reset()
    states = env.state()
    while simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            obs, _, _, _, _ = env.step(actions)
            states = env.state()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
