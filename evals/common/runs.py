"""Training runs: where they are, what they trained, their checkpoints, and the runs a plot compares against.

A run is logs/skrl/locomanip_marl/<YYYY-MM-DD_HH-MM-SS>_<name>/ (PPO, scripts/skrl/train.py, and SuperMimic,
scripts/supermimic/train.py: same layout) with params/task.txt, params/env.yaml and checkpoints/agent_<env steps>.pt.
"""

from __future__ import annotations

import re
from pathlib import Path

from .outputs import REPO

RUNS_ROOT = REPO / "logs" / "skrl" / "locomanip_marl"
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_")
_SEED = re.compile(r"_s\d+$")


def run_dir_of(checkpoint: str | Path) -> Path:
    return Path(checkpoint).resolve().parents[1]


def resolve(run: str | Path) -> Path:
    """A run dir from a path, a run folder name, or a name (the newest run of that name)."""
    p = Path(run)
    if p.is_dir():
        return p.resolve()
    if (RUNS_ROOT / str(run)).is_dir():
        return RUNS_ROOT / str(run)
    hits = sorted(RUNS_ROOT.glob(f"*_{run}"))
    if not hits:
        raise FileNotFoundError(f"no run {run!r} (a path, or a name under {RUNS_ROOT})")
    return hits[-1]


def name(run_dir: Path) -> str:
    """The run's folder name, timestamp included: the key of logs/evals/<run>."""
    return Path(run_dir).name


def short(run_dir: Path | str) -> str:
    """The name without the timestamp: LegsDeploy200k_s42."""
    return _STAMP.sub("", Path(run_dir).name)


def seed_group(run_dir: Path | str) -> str:
    """Runs differing only in their _sNN seed share a group (the plots draw a group as one line with a band)."""
    return _SEED.sub("", short(run_dir))


def task(run_dir: Path) -> str | None:
    f = Path(run_dir) / "params" / "task.txt"
    return f.read_text().strip() if f.is_file() else None


def trainer(run_dir: Path) -> str:
    return "supermimic" if (Path(run_dir) / "params" / "supermimic.yaml").is_file() else "skrl"


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """[(env steps, path)] of checkpoints/agent_<N>.pt, ascending."""
    out = []
    for p in (Path(run_dir) / "checkpoints").glob("agent_*.pt"):
        m = re.fullmatch(r"agent_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def step_of(checkpoint: str | Path) -> int | None:
    m = re.search(r"_(\d+)\.pt$", str(checkpoint))
    return int(m.group(1)) if m else None


def prior_runs(run_dir: Path, n: int, has_evals=None) -> list[Path]:
    """The newest n other runs of the same task (by start time), optionally only those has_evals(run) accepts
    (e.g. runs with cached scorecards for the plot's backend and body)."""
    run_dir = Path(run_dir).resolve()
    t = task(run_dir)
    out = []
    for p in sorted(RUNS_ROOT.iterdir(), reverse=True):
        if not p.is_dir() or p.resolve() == run_dir or task(p) != t:
            continue
        if has_evals is not None and not has_evals(p):
            continue
        out.append(p)
        if len(out) >= n:
            break
    return out
