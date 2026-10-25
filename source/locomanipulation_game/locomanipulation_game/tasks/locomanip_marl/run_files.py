"""A training run's files, for the scripts that resume or evaluate it: its saved curriculum state and the command
line overrides it trained with (<run>/params/hydra_overrides.txt, which scripts/skrl/train.py writes).

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
        if os.path.isfile(latest):
            return latest
        saved = sorted(glob.glob(os.path.join(run_dir, folder, f"{prefix}_*.pt")), key=os.path.getmtime)
        if saved:
            return saved[-1]
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
