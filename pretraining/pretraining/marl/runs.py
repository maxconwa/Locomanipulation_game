# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/runs.py.
# Changes: REPO_ROOT is this repository's root (parents[3] of this file); the runs root is $PRETRAINING_RUNS_DIR,
# default data/pretraining/runs under it (h12_train: $H12_RUNS_DIR, default data/runs).
"""Run-directory bookkeeping: paths, atomic writes, the progress CSV, recorded configs.

A run directory holds ``run.json`` (every setting, rebuilt from by ``config_from_dict``),
``progress.csv``, the deploy-facing ``world_model.pt``, ``actors.pt`` and ``critics.pt``,
the training state ``checkpoint.pt`` with ``checkpoint.json`` (its iteration and wall
time) and ``checkpoints/it_<iteration>.pt``, the last few periodic ones.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import typing
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def default_runs_dir() -> str:
    """Return ``$PRETRAINING_RUNS_DIR``, else ``data/pretraining/runs`` under the repository root."""
    return os.environ.get("PRETRAINING_RUNS_DIR", str(REPO_ROOT / "data" / "pretraining" / "runs"))


def to_repo_path(path: str | os.PathLike) -> str:
    """Return ``path`` relative to the repository root when it lies inside it, else absolute."""
    p = Path(path).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def from_repo_path(path: str | os.PathLike) -> Path:
    """Return a path stored in ``run.json`` resolved against the repository root; an absolute one is kept."""
    return REPO_ROOT / path


def atomic_save(obj, path: Path) -> None:
    """``torch.save`` ``obj`` to ``path`` through a temporary file and a rename, so a crash leaves the old file."""
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and a rename."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def config_from_dict(cls, data: dict, strict: bool = False, _where: str = ""):
    """Return a ``cls`` dataclass from its ``dataclasses.asdict`` form, nested dataclasses and tuples included.

    Missing keys keep their defaults. Keys ``cls`` does not have are ignored (an older run's
    ``run.json``), or with ``strict`` refused with a ``ValueError`` naming them (a config file).
    """
    hints = typing.get_type_hints(cls)
    if strict:
        unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"unknown {cls.__name__} key(s): {', '.join(_where + k for k in unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value, hint = data[f.name], hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = config_from_dict(hint, value, strict, f"{_where}{f.name}.")
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


class ProgressCsv:
    """``progress.csv`` with a header that always names every column any row has.

    Rows are kept in memory. A row with a new key rewrites the file (atomically) with the
    wider header; other rows are appended. Opening it for a resumed run keeps the rows up
    to the checkpoint's iteration and drops later ones, so the curves continue.
    """

    def __init__(self, path: Path, keep_through_iteration: int | None = None):
        """Open ``path``: a new file, or with ``keep_through_iteration`` an existing one to continue."""
        self.path = path
        self.rows: list[dict] = []
        self.fields: list[str] = []
        if keep_through_iteration is not None and path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                self.fields = list(reader.fieldnames or [])
                self.rows = [r for r in reader if int(r["iteration"]) <= keep_through_iteration]
        self._rewrite()

    def _rewrite(self) -> None:
        """Write the header and every row."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.fields)
        writer.writeheader()
        writer.writerows(self.rows)
        atomic_write_text(self.path, buf.getvalue())

    def append(self, row: dict) -> None:
        """Add one row, widening the header if it has new keys."""
        self.rows.append(row)
        new = [k for k in row if k not in self.fields]
        if new:
            self.fields += new
            self._rewrite()
            return
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def read_checkpoint_info(run_dir: Path) -> dict:
    """Return ``checkpoint.json`` of a run: its ``iteration`` and ``wall_s``."""
    return json.loads((Path(run_dir) / "checkpoint.json").read_text())


def keep_last(directory: Path, pattern: str, n: int) -> None:
    """Delete all but the ``n`` newest (by name) files matching ``pattern`` in ``directory``."""
    for old in sorted(directory.glob(pattern))[:-n] if n > 0 else []:
        old.unlink()
