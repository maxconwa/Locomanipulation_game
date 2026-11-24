"""Dump the resolved env cfgs of task ids to JSON, or compare two dumps.

The guard for changes to shared files: the main tasks and the pretraining's track task must keep bit-identical
resolved cfgs, so a change ends with a dump compared against the baseline (exit 1 and every differing key path if
anything moved).

    python scripts/dump_task_cfgs.py --out logs/cfg_dumps/baseline.json [--tasks <id> ...]
    python scripts/dump_task_cfgs.py --compare logs/cfg_dumps/baseline.json logs/cfg_dumps/now.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_TASKS = [
    "LocoManip-LegsOnly-v0",
    "LocoManip-ArmsOnly-v0",
    "LocoManip-Marl-Direct-v0",
    "LocoManip-Pretrain-Track-v0",
]


def _json_safe(value):
    """``value`` with every leaf JSON can hold kept and anything else (callables, classes) as its repr."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def dump(task_ids: list[str]) -> dict[str, dict]:
    """Each id's env cfg as a JSON-safe dict (needs the Isaac app running)."""
    from isaaclab.utils import class_to_dict
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401  registers the main ids
    import pretraining.tasks  # noqa: F401  registers the track ids

    return {t: _json_safe(class_to_dict(load_cfg_from_registry(t, "env_cfg_entry_point"))) for t in task_ids}


def compare(a, b, path: str = "") -> list[str]:
    """The key paths (dot-joined, list items by index) at which ``a`` and ``b`` differ, sorted."""
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for k in sorted(set(a) | set(b)):
            sub = f"{path}.{k}" if path else str(k)
            diffs += [sub] if (k in a) != (k in b) else compare(a[k], b[k], sub)
        return diffs
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare(x, y, f"{path}[{i}]")]
    # NaN != NaN, so compare the JSON spellings, not the values
    return [] if json.dumps(a) == json.dumps(b) else [path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="write the dump of --tasks here")
    group.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="compare two dumps")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = parser.parse_args()

    if args.compare:
        a, b = (json.loads(p.read_text()) for p in args.compare)
        diffs = compare(a, b)
        for d in diffs:
            print(d)
        print(f"{len(diffs)} differing key path(s)")
        return 1 if diffs else 0

    from isaaclab.app import AppLauncher

    sys.argv = sys.argv[:1]  # Kit reads sys.argv
    AppLauncher(headless=True)
    dumps = dump(args.tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dumps, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {', '.join(args.tasks)}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # app.close() has hung in this environment


if __name__ == "__main__":
    sys.exit(main())
