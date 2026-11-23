"""get_scorecard --run: score a run's checkpoints, one process per checkpoint (an Isaac app scores one), skipping
the ones whose json exists."""

from __future__ import annotations

import subprocess
import sys

from . import outputs, runs


def strip(argv: list[str], names: set[str]) -> list[str]:
    """argv without the named options (and their values)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        if a.split("=", 1)[0] in names:
            continue
        out.append(a)
    return out


def select(run_dir, steps: str):
    ckpts = runs.checkpoints(run_dir)
    if steps == "last":
        return ckpts[-1:]
    if steps != "all":
        wanted = {int(s) for s in steps.split(",")}
        return [c for c in ckpts if c[0] in wanted]
    return ckpts


def run_steps(module: str, run: str, steps: str, out_base, force: bool, passthrough: list[str]) -> int:
    """For each selected checkpoint without a cached json (out_base(checkpoint) -> path without extension), run
    `python -m <module> --checkpoint <ckpt> <passthrough>`. Returns 1 if any failed."""
    run_dir = runs.resolve(run)
    ckpts = select(run_dir, steps)
    if not ckpts:
        print(f"[evals] {runs.short(run_dir)}: no checkpoints selected ({steps})")
        return 1
    failed = 0
    for base, path in [(out_base(str(p)), p) for _, p in ckpts]:
        if outputs.cached(base) and not force:
            print(f"[evals] cached: {base}.json")
            continue
        cmd = [sys.executable, "-m", module, "--checkpoint", str(path)] + passthrough
        print(f"[evals] {' '.join(cmd)}", flush=True)
        failed += subprocess.call(cmd) != 0
    return 1 if failed else 0
