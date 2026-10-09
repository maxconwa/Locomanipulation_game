"""The reach evaluation's paths and condition registry, over the shared constants and helpers in mujoco_evals.common.

Layout:
    paper/reach/conditions.yaml     which controllers exist: label, kind, run directory, display name
    logs/reach/goals/               the fixed goal grid every controller is scored on (mujoco_evals.common)
    logs/reach/results/             <label>/<sim>/ trials.csv, traces.npz, meta.json; summary/; curves/
    logs/reach/figures/, tex/       outputs of figures.py / analyze.py
    logs/reach/<page>.html          the result page, from build_page.py

Importing this puts the repository root on sys.path, so the tools can import mujoco_evals afterwards.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH.parents[1]))

from mujoco_evals.common import *  # noqa: E402, F401, F403
from mujoco_evals.common import REACH_OUT, _yaml_load_any  # noqa: E402, F401

FIG_DIR = REACH_OUT / "figures"
TEX_DIR = REACH_OUT / "tex"
PAGE = REACH_OUT / "20261008-reach_cooperation.html"
CONDITIONS_FILE = REACH / "conditions.yaml"


def newest_checkpoint(run_dir: Path) -> Path | None:
    """<run>/checkpoints/agent_<N>.pt with the largest N (skrl's best_agent.pt only when nothing else exists)."""
    ckpts = list((Path(run_dir) / "checkpoints").glob("agent_*.pt"))
    if ckpts:
        return max(ckpts, key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1)))
    best = Path(run_dir) / "checkpoints" / "best_agent.pt"
    return best if best.is_file() else None


def load_conditions() -> list[dict]:
    """conditions.yaml entries with run paths resolved relative to paper/reach."""
    import yaml
    if not CONDITIONS_FILE.is_file():
        return []
    entries = yaml.safe_load(CONDITIONS_FILE.read_text()).get("conditions", []) or []
    for e in entries:
        if e.get("run"):
            e["run"] = str((REACH / e["run"]).resolve()) if not os.path.isabs(e["run"]) else e["run"]
    return entries
