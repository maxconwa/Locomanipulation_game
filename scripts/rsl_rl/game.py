"""Run an ordered schedule of training rounds in one process, handing policies between them.

Everything one game produces lives in one folder, one subfolder per step:

    logs/<game>/               --game NAME, or one past the highest game_run<N>
        game.log               console output (Python and Kit)
        s00-Legs-R0/           checkpoints, params/, TensorBoard events, exported/
        s01-Upper-Adv-Ri/
        ...

Each step trains one task against the frozen opponent it consumes, then exports
the policy it produces:

    Legs-R0-v0       consumes nothing   produces legs
    Upper-Adv-Ri-v0  consumes legs      produces adv
    Legs-Ri-v0       consumes adv       produces legs

A task repeated later in the schedule resumes from its own previous step.
Continue a game after a crash with --game <name> --start <step>. A step whose
folder already exists is never overwritten: move it aside first.
Fail fast: an exception stops everything.
"""

import argparse
import atexit
import os
import sys
import threading
from pathlib import Path

from isaaclab.app import AppLauncher

LOGS = Path("logs")

TASKS = {
    "Legs-R0-v0": {
        "experiment": "locoManipulation_legs",
        "consumes": None,
        "produces": "legs",
    },
    "Upper-Adv-Ri-v0": {
        "experiment": "locoManipulation_upper",
        "consumes": "legs",
        "produces": "adv",
    },
    "Legs-Ri-v0": {
        "experiment": "locoManipulation_legs",
        "consumes": "adv",
        "produces": "legs",
    },
}

SCHEDULE = [
    #Round 0
    ("Legs-R0-v0",      7500),
    #Round 1
    ("Upper-Adv-Ri-v0", 7500),
    ("Legs-Ri-v0",      5000),
    #Round 2
    ("Upper-Adv-Ri-v0", 5000),
    ("Legs-Ri-v0",      5000),
    #Round 3
    ("Upper-Adv-Ri-v0", 5000),
    ("Legs-Ri-v0",      5000),
]

POLICIES: dict[str, str] = {}   # "legs" / "adv" -> exported policy.pt
LAST_RUN: dict[str, str] = {}   # experiment -> its latest step folder in this game


def step_name(index, task):
    return f"s{index:02d}-{task.removesuffix('-v0')}"


def next_game_dir():
    """logs/game_run<N>, one past the highest N already there."""
    taken = [int(p.name.removeprefix("game_run")) for p in LOGS.glob("game_run*")
             if p.name.removeprefix("game_run").isdigit()]
    return LOGS / f"game_run{max(taken, default=0) + 1}"


def tee_output(path):
    """Copy fds 1 and 2 (Python's and Kit's C++ output) into path while still
    printing to the terminal. Returns a function that flushes and stops the copy."""
    log = open(path, "ab", buffering=0)
    terminal = os.dup(1)
    read_end, write_end = os.pipe()
    os.dup2(write_end, 1)
    os.dup2(write_end, 2)
    os.close(write_end)
    # fd 1 is a pipe now, which Python would otherwise block-buffer
    sys.stdout.reconfigure(line_buffering=True)

    def pump():
        while chunk := os.read(read_end, 65536):
            os.write(terminal, chunk)
            log.write(chunk)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()

    def stop():
        if log.closed:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        # point 1 and 2 back at the terminal; with the pipe's last writers gone, pump() reads EOF
        os.dup2(terminal, 1)
        os.dup2(terminal, 2)
        thread.join(timeout=5)
        log.close()

    return stop


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--game", default=None, help="Game folder under logs/; default: the next game_run<N>.")
parser.add_argument("--dry-run", action="store_true", help="Print the plan only; doesn't start Kit.")
parser.add_argument("--start", type=int, default=0, help="Schedule index to start from.")
parser.add_argument("--smoke", type=int, default=None, help="Override every entry's iterations.")
parser.add_argument("--num_envs", type=int, default=None, help="Override every task's num_envs.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

game_dir = LOGS / args.game if args.game else next_game_dir()

# Refuse before Kit starts: an existing step folder holds an earlier attempt.
for index, (task, _) in enumerate(SCHEDULE):
    if index >= args.start and (game_dir / step_name(index, task)).exists():
        sys.exit(f"{game_dir / step_name(index, task)} already exists; move it aside to rerun step {index}.")

if not args.dry_run:
    game_dir.mkdir(parents=True, exist_ok=True)
    stop_tee = tee_output(game_dir / "game.log")
    # atexit runs after an uncaught exception's traceback is printed, so crashes land in game.log
    atexit.register(stop_tee)
    simulation_app = AppLauncher(args).app
    # after AppLauncher: both import isaaclab
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from train_lib import train

print(f"game: {game_dir.resolve()}", flush=True)

for index, (task, iterations) in enumerate(SCHEDULE):
    spec = TASKS[task]
    iterations = args.smoke or iterations
    step = step_name(index, task)
    export = game_dir / step / "exported" / "policy.pt"

    if index < args.start:
        # only finished steps count: train() exports after the last iteration
        if export.is_file():
            LAST_RUN[spec["experiment"]] = step
            POLICIES[spec["produces"]] = str(export.resolve())
        continue

    if spec["consumes"] and spec["consumes"] not in POLICIES:
        sys.exit(f"\nStep {index} ({task}) needs a {spec['consumes']} policy, which nothing in "
                 f"{game_dir} has produced. To continue a game, pass its --game.")

    resume_run = LAST_RUN.get(spec["experiment"])
    opponent_path = POLICIES.get(spec["consumes"])
    print(f"\n===== step {index}: {task}  ({iterations} iterations) =====", flush=True)
    print(f"  resume={resume_run}  opponent={opponent_path}", flush=True)

    if args.dry_run:
        # step folders are deterministic, so later steps print the paths they would really use
        LAST_RUN[spec["experiment"]] = step
        POLICIES[spec["produces"]] = str(export)
        continue

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = iterations
    agent_cfg.run_name = step
    if resume_run:
        agent_cfg.resume = True
        # a regex matched inside game_dir; "$" so s02-Legs-Ri can't also match a moved-aside s02-Legs-Ri.old
        agent_cfg.load_run = f"{resume_run}$"
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    _, export_path = train(task, env_cfg, agent_cfg, opponent=opponent_path,
                           log_dir=str(game_dir / step), export=True)

    LAST_RUN[spec["experiment"]] = step
    POLICIES[spec["produces"]] = export_path
    print(f"  {spec['produces']} -> {export_path}", flush=True)

print(f"\n===== schedule complete ({len(SCHEDULE) - args.start} steps) =====", flush=True)
if not args.dry_run:
    stop_tee()
    simulation_app.close()
