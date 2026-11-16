"""A training run's files, for the scripts that resume or evaluate it: its saved curriculum state, its task
(<run>/params/task.txt) and the command line overrides it trained with (<run>/params/hydra_overrides.txt), both
written by scripts/skrl/train.py.

No Isaac Lab imports at module level: known_overrides imports the task's cfg, so call it after the app is up.
"""

from __future__ import annotations

import glob
import os
import re


def state_checkpoint_for(agent_checkpoint: str) -> str | None:
    """The curriculum state saved with an skrl checkpoint: <run>/curriculum/state_<step>.pt, else the newest one;
    runs from before 2026-10-25 saved it in <run>/estimator/estimator_<step>.pt."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(agent_checkpoint)))
    step = re.search(r"_(\d+)\.pt$", os.path.basename(agent_checkpoint))
    for folder, prefix in (("curriculum", "state"), ("estimator", "estimator")):
        if step:
            same_step = os.path.join(run_dir, folder, f"{prefix}_{step.group(1)}.pt")
            if os.path.isfile(same_step):
                return same_step
        latest = os.path.join(run_dir, folder, f"{prefix}_latest.pt")
        saved = sorted(glob.glob(os.path.join(run_dir, folder, f"{prefix}_*.pt")), key=os.path.getmtime)
        other = latest if os.path.isfile(latest) else saved[-1] if saved else None
        if other:
            if step:
                print(f"[WARNING] no curriculum state was saved at step {step.group(1)}: using {other}, which may be"
                      f" from another step of the run")
            return other
    return None


def run_overrides(run_dir: str) -> list[str]:
    """The run's command line overrides, one per line of <run>/params/hydra_overrides.txt ([] if it has none)."""
    path = os.path.join(run_dir, "params", "hydra_overrides.txt")
    if not os.path.isfile(path):
        return []
    return [line.strip() for line in open(path) if line.strip()]


def known_overrides(overrides: list[str], task: str) -> tuple[list[str], list[str]]:
    """(kept, dropped): the overrides whose key the task's current env cfg or agent cfg still has, and the rest.
    A run trained before the 2026-10-25 cleanup set switches (pd_gains, default_leg_pose, estimator.*, ...) whose
    values are now the defaults and whose keys are gone; replaying them would stop Hydra with a KeyError."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    roots = {"env": load_cfg_from_registry(task, "env_cfg_entry_point"),
             "agent": load_cfg_from_registry(task, "skrl_mappo_cfg_entry_point")}
    kept, dropped = [], []
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        head, *path = key.split(".")
        node, found = roots.get(head), head in roots
        for name in path:
            if not found:
                break
            if isinstance(node, dict):
                found, node = name in node, node.get(name)
            else:
                found, node = hasattr(node, name), getattr(node, name, None)
        (kept if found else dropped).append(override)
    return kept, dropped


def run_task(run_dir: str, asked: str | None = None, fallback: str = "LocoManip-Marl-Direct-v0") -> str:
    """The task a script should build for a run: `asked` (its --task) if given, else the one the run was trained on
    (<run>/params/task.txt), else `fallback`, said aloud: runs from before 2026-11-16 did not record theirs, and a
    legs-only or arms-only run rebuilt as the walking task moves joints its policy never trained."""
    path = os.path.join(run_dir, "params", "task.txt")
    recorded = open(path).read().strip() if os.path.isfile(path) else None
    if asked:
        if recorded and recorded != asked:
            print(f"[WARNING] --task {asked}, but {run_dir} was trained on {recorded}")
        return asked
    if recorded:
        return recorded
    print(f"[WARNING] {run_dir} does not record its task; assuming {fallback} (pass --task if it was another)")
    return fallback


def replay_run(checkpoint: str, task: str | None, extra_overrides: list[str], fallback: str = "LocoManip-Marl-Direct-v0"
               ) -> tuple[str, str, str, list[str]]:
    """What every script that rebuilds a run for a checkpoint does first: (the checkpoint's absolute path, its run's
    folder, the task (run_task), the run's overrides the current cfgs still have). Sets sys.argv to those overrides
    followed by extra_overrides, for hydra_task_config. Call it after the app is up."""
    import sys

    checkpoint = os.path.abspath(checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint))
    task = run_task(run_dir, task, fallback)
    replayed, dropped = known_overrides(run_overrides(run_dir), task)
    if dropped:
        print(f"[INFO] run overrides the current cfg no longer has (their values are the defaults now): {dropped}")
    sys.argv = [sys.argv[0]] + replayed + extra_overrides
    return checkpoint, run_dir, task, replayed


def evaluation_cfgs(env_cfg, agent_cfg: dict, checkpoint: str, run_dir: str):
    """The cfgs of a run rebuilt to evaluate a checkpoint: no warm start, the curriculum state saved with the
    checkpoint, nothing logged or saved."""
    env_cfg.log_dir = run_dir
    env_cfg.training = False
    env_cfg.state_checkpoint_path = state_checkpoint_for(checkpoint)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0


def env_override_keys(overrides: list[str]) -> list[str]:
    """The env cfg fields a command line's Hydra overrides set: their keys, without the leading "env."."""
    keys = [override.split("=", 1)[0].lstrip("+~") for override in overrides]
    return [key[len("env."):] for key in keys if key.startswith("env.")]


def cfg_value(cfg, key: str):
    """The value at a dotted key of a cfg of configclasses and dicts."""
    node = cfg
    for name in key.split("."):
        node = node[name] if isinstance(node, dict) else getattr(node, name)
    return node
