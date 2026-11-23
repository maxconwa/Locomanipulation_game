"""Where evals write: logs/evals/<run>/<backend>/<body>/<kind>_<step>[<suffix>].{json,md,...}.

backend: isaac | mujoco | deploy; body: legs | arms. A file whose json exists is not scored again (get_scorecard's
cache); --force rescores.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALS_ROOT = Path(os.environ.get("LMG_EVALS_DIR", str(REPO / "logs" / "evals")))


def eval_dir(run_name: str, backend: str, body: str) -> Path:
    return EVALS_ROOT / run_name / backend / body


def base(run_name: str, backend: str, body: str, kind: str, step: int | None, suffix: str = "") -> Path:
    """The path without extension, e.g. logs/evals/<run>/isaac/legs/scorecard_200160."""
    return eval_dir(run_name, backend, body) / f"{kind}{'' if step is None else f'_{step}'}{suffix}"


def cached(path_base: Path) -> bool:
    return Path(str(path_base) + ".json").is_file()
