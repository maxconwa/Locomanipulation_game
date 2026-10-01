"""Train one RSL-RL run inside the current process.

Import this only after an AppLauncher has started Kit: the imports below need
the app running. train.py (one run) and game.py (one run per schedule step)
both launch the app first, then call train().
"""

import importlib.metadata as metadata
import os
import time
from datetime import datetime

import gymnasium as gym
import omni.usd
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path

import locomanipulation_game.tasks  # noqa: F401

INSTALLED_VERSION = metadata.version("rsl-rl-lib")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train(
    task: str,
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
    opponent: str | None = None,
    log_dir: str | None = None,
    export: bool = False,
) -> tuple[str, str | None]:
    """Train `task` with the given cfgs. Returns (log_dir, export_path).

    opponent: exported policy.pt for the task's frozen `actions.opponent` term.
    log_dir:  exact run folder; None means logs/rsl_rl/<experiment>/<time-stamp>_<run_name>.
              agent_cfg.load_run is resolved in log_dir's parent folder.
    export:   write <log_dir>/exported/policy.pt (+ .onnx); export_path is None otherwise.
    """
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, INSTALLED_VERSION)
    # certain randomizations happen at env creation, so seed before gym.make
    env_cfg.seed = agent_cfg.seed
    if opponent is not None:
        env_cfg.actions.opponent.policy_path = opponent

    if log_dir is None:
        # logs/rsl_rl/<experiment>/<time-stamp>_<run_name>
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)
    else:
        # the caller picked the run folder (game.py: <game>/sNN-<task>); load_run resolves beside it
        log_dir = os.path.abspath(log_dir)
        log_root_path = os.path.dirname(log_dir)
    env_cfg.log_dir = log_dir

    # resolve before the runner creates log_dir, which a ".*" load_run would match
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # a fresh stage per env, as IsaacLab's own sequential-env test does
    # (isaaclab_tasks/test/env_test_utils.py)
    omni.usd.get_context().new_stage()
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # same export play.py did, from the in-memory policy (= the final checkpoint)
    export_path = None
    if export:
        export_dir = os.path.join(log_dir, "exported")
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        export_path = os.path.join(export_dir, "policy.pt")

    # learn() leaves the TensorBoard writer open
    if runner.logger.writer is not None:
        runner.logger.writer.close()
    env.close()
    # free this run's GPU tensors before the next env allocates its own
    del runner, env
    torch.cuda.empty_cache()

    return log_dir, export_path
