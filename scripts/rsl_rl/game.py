"""Run the round-0 curriculum end to end: legs, then upper, then whole body.

One subprocess per round. train.py launches Isaac at import and binds its task
name into the hydra decorator at module scope, so one process can only ever
host one task. Fail fast: a non-zero exit stops the queue.

Nothing is handed between rounds. The three have different observation and
action dimensions (567/402/651, 12/14/26), so no checkpoint loads into the
next -- this is a queue, not a curriculum handoff.

Run from the repo root: train.py writes to a relative logs/rsl_rl/ path.
"""

import argparse
import subprocess
import sys
from pathlib import Path

TRAIN_PY = Path(__file__).parent / "train.py"

# (task, max_iterations)
ROUNDS = [
    ("Legs-R0-v0", 15000),
    ("Upper-R0-v0", 15000),
    # ("WB-R0-v0", 15000),
]

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Unrecognised arguments pass through to train.py, and come after the\n"
           "per-round flags, so --max_iterations 5 overrides the table.\n"
           "Smoke test: python scripts/rsl_rl/game.py --num_envs 4 --max_iterations 5",
)
parser.add_argument("--gui", action="store_true", help="Run with a window. Default is headless.")
parser.add_argument("--dry-run", action="store_true", help="Print each command without running it.")
args, passthrough = parser.parse_known_args()

for i, (task, iterations) in enumerate(ROUNDS, start=1):
    cmd = [sys.executable, str(TRAIN_PY), f"--task={task}", f"--max_iterations={iterations}"]
    if not args.gui:
        cmd.append("--headless")
    cmd += passthrough

    print(f"\n=== round {i}/{len(ROUNDS)}: {task} ===", flush=True)
    print("  " + " ".join(cmd), flush=True)
    if args.dry_run:
        continue

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n{task} exited {result.returncode}. Stopping.")

print(f"\n=== {len(ROUNDS)} rounds complete ===")
