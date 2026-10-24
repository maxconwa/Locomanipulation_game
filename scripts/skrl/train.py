"""Train the two-agent task with skrl's MAPPO.

    python scripts/skrl/train.py --headless [--max_iterations 3800] [--name <run>] [--checkpoint <run>/checkpoints/agent_<N>.pt]

--checkpoint resumes the agents and the env's pelvis estimator and curriculum state saved beside them. Arguments
after these are Hydra overrides of the env and agent configs (env.<field>=..., agent.<field>=...); they are saved to
<run>/params/hydra_overrides.txt, one per line, for the evaluation scripts. When the run's last step is not a
checkpoint step, a last checkpoint (and estimator) is saved at the end.
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
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.odometry import estimator_checkpoint_for  # noqa: E402


def bound_log_std(agent, policy_cfg: dict, per_agent: dict):
    """Clamp each agent's log_std parameter into [min_log_std, max_log_std], narrowed per agent by per_agent, now
    and after every update. per_agent[uid] is [low, high] for all of the agent's actions, or [[low, high, count], ...]
    for runs of its action dimensions in order. skrl clamps log_std only in the forward pass; a parameter pushed past
    the bound then gets no gradient and the std stays pinned."""
    floor, ceiling = policy_cfg["min_log_std"], policy_cfg["max_log_std"]
    bounds = {uid: (floor, ceiling) for uid in agent.policies}
    for uid, spec in per_agent.items():
        if isinstance(spec[0], (list, tuple)):
            param = agent.policies[uid].log_std_parameter
            low = [max(lo, floor) for lo, _, count in spec for _ in range(int(count))]
            high = [min(hi, ceiling) for _, hi, count in spec for _ in range(int(count))]
            if len(low) != param.numel():
                raise ValueError(f"{uid}: log_std bounds cover {len(low)} actions, the policy has {param.numel()}")
            bounds[uid] = (torch.tensor(low, device=param.device), torch.tensor(high, device=param.device))
            print(f"[INFO] {uid}: log_std bounds per run of actions {spec}")
        else:
            low, high = spec
            bounds[uid] = (max(low, floor), min(high, ceiling))
    print(f"[INFO] log_std bounds per agent: { {u: b for u, b in bounds.items() if not torch.is_tensor(b[0])} }")

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


def restrict_entropy(agent, per_agent: dict):
    """Pay each agent's entropy bonus over only some of its action dimensions: per_agent[uid] is [[include, count], ...]
    per run of action dimensions in order (include 1 or 0). skrl's GaussianMixin sums the entropy over all of them."""
    for uid, spec in per_agent.items():
        policy = agent.policies[uid]
        mask = torch.tensor([bool(include) for include, count in spec for _ in range(int(count))], device=policy.device)
        if len(mask) != policy.log_std_parameter.numel():
            raise ValueError(f"{uid}: the entropy mask covers {len(mask)} actions, the policy has {policy.log_std_parameter.numel()}")

        def get_entropy(*, role: str = "", policy=policy, mask=mask) -> torch.Tensor:
            if policy._g_distribution is None:
                return torch.tensor(0.0, device=policy.device)
            return policy._g_distribution.entropy()[..., mask].sum(dim=-1).unsqueeze(-1)

        policy.get_entropy = get_entropy
        print(f"[INFO] {uid}: entropy bonus over {int(mask.sum())} of {len(mask)} action dimensions {spec}")


def drop_value_preprocessors(agent, uids):
    """Train these agents' critics on raw returns. Before loading a checkpoint, so the module is neither saved nor
    restored."""
    for uid in uids:
        agent._value_preprocessor[uid] = agent._empty_preprocessor
        agent.checkpoint_modules[uid].pop("value_preprocessor", None)
        print(f"[INFO] {uid}: no value preprocessor, the critic learns raw returns")


def fix_observation_scales(agent, env):
    """With env cfg.observation_scales, freeze each agent's observation and state scalers (skrl's
    RunningStandardScaler) at mean 0 and variance 1/scale^2: the inputs are scaled by fixed constants, not normalized
    by running statistics. The checkpoint keeps the scaler's format, so the export and the MuJoCo tools read it as
    before. After loading a checkpoint, which would restore whatever statistics it saved."""
    if not env.cfg.observation_scales:
        return
    state = env.state_scales()
    for uid in agent.possible_agents:
        for kind, scaler, scale in (("observation", agent._observation_preprocessor[uid], env.observation_scales(uid)),
                                    ("state", agent._state_preprocessor[uid], state)):
            if not isinstance(scaler, RunningStandardScaler) or scaler.running_mean.shape != scale.shape:
                raise ValueError(f"{uid}: the {kind} preprocessor must be a RunningStandardScaler of {scale.shape}")
            scaler.running_mean.zero_()
            scaler.running_variance.copy_(scale.double().reciprocal().square())
            scaler._parallel_variance = lambda *args, **kwargs: None
        print(f"[INFO] {uid}: fixed observation scales, no running normalization")


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
    if hasattr(env_cfg, "sync_derived"):
        env_cfg.sync_derived()  # after the overrides, so env.yaml records the sensor periods, windows and gains the run uses
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "hydra_overrides.txt"), "w") as f:
        f.writelines(f"{arg}\n" for arg in hydra_args)
    env_cfg.log_dir = log_dir
    if args_cli.checkpoint:
        env_cfg.estimator.checkpoint_path = estimator_checkpoint_for(args_cli.checkpoint)

    env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    runner = MarlRunner(env, agent_cfg)
    drop_value_preprocessors(runner.agent, agent_cfg["no_value_preprocessor"])
    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
    fix_observation_scales(runner.agent, env.unwrapped)
    bound_log_std(runner.agent, agent_cfg["models"]["policy"], agent_cfg["log_std_bounds"])
    restrict_entropy(runner.agent, agent_cfg.get("entropy_action_dims", {}))

    start_time = time.time()
    runner.run()
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    timesteps, interval = agent_cfg["trainer"]["timesteps"], agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    if interval > 0 and timesteps % interval:
        runner.agent.write_checkpoint(timestep=timesteps, timesteps=timesteps)
        env.unwrapped._save_estimator()
        print(f"[INFO] Saved the last checkpoint, agent_{timesteps}.pt")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
