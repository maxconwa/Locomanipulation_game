"""Run an ordered schedule of training rounds, handing policies between them.

Each entry trains one task, then exports its policy. Tasks declare which
environment variable they read their frozen opponent from and which one they
write into, so the schedule is a flat list and the plumbing is a dict:

    Legs-R0-v0       consumes nothing            produces LEGS_POLICY_PATH
    Upper-Adv-R1-v0  consumes LEGS_POLICY_PATH   produces ADV_POLICY_PATH
    Legs-R1-v0       consumes ADV_POLICY_PATH    produces LEGS_POLICY_PATH

A task repeated later in the schedule resumes from its own previous run.
Fail fast: a non-zero exit stops everything.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
TRAIN, PLAY = HERE / "train.py", HERE / "play.py"
LOGS = Path("logs/rsl_rl")

TASKS = {
    "Legs-R0-v0": {
        "experiment": "locoManipulation_legs_r0",
        "consumes": None,
        "produces": "LEGS_POLICY_PATH",
    },
    "Upper-Adv-R1-v0": {
        "experiment": "locoManipulation_upper_adv_r1",
        "consumes": "LEGS_POLICY_PATH",
        "produces": "ADV_POLICY_PATH",
    },
    "Legs-R1-v0": {
        "experiment": "locoManipulation_legs_r0",
        "consumes": "ADV_POLICY_PATH",
        "produces": "LEGS_POLICY_PATH",
    },
}

SCHEDULE = [
    ("Legs-R0-v0",      7500),
    ("Upper-Adv-R1-v0", 7500),
    ("Legs-R1-v0",      5000),
    ("Upper-Adv-R1-v0", 5000),
    ("Legs-R1-v0",      5000),
    ("Upper-Adv-R1-v0", 5000),
    ("Legs-R1-v0",      5000),
]

# Seed the pool to skip a round: e.g. start at the adversary by pre-supplying
# the legs policy and deleting the Legs-R0-v0 entry above.
POLICIES: dict[str, str] = {}


def run(cmd, label, experiment=None, stall_s=2400, timeout_s=None):
    """Child inherits stdout; PYTHONUNBUFFERED keeps it live under tee.

    experiment: kill if its log root gains no new checkpoint for stall_s.
    timeout_s:  hard wall-clock cap, for the export (which saves nothing).
    """
    print("  " + " ".join(f"{k}={v}" for k, v in POLICIES.items()), flush=True)
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1", **POLICIES})
    started = moved = time.time()
    seen = None
    while proc.poll() is None:
        time.sleep(30)
        now = time.time()
        if timeout_s and now - started > timeout_s:
            print(f"  {label}: {timeout_s}s wall clock exceeded, killing.", flush=True)
            proc.kill()
            break
        if experiment:
            d = newest_run(experiment)
            it = last_iteration(d) if d else None
            if it != seen:
                seen, moved = it, now
            elif now - moved > stall_s:
                print(f"  {label}: no new checkpoint in {stall_s}s, killing.", flush=True)
                proc.kill()
                break
    return proc.wait()


def newest_run(experiment):
    runs = sorted((LOGS / experiment).glob("2*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def last_iteration(run_dir):
    """Highest model_<n>.pt in a run directory, or None if it saved nothing."""
    saved = [int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pt")]
    return max(saved) if saved else None


def train_command(task, index, iterations, resume_run=None):
    cmd = [sys.executable, str(TRAIN), f"--task={task}", "--headless",
           f"--max_iterations={iterations}",
           f"--run_name=s{index:02d}-{task.removesuffix('-v0')}"]
    if args.num_envs:
        cmd.append(f"--num_envs={args.num_envs}")
    if resume_run:
        cmd += ["--resume", f"--load_run={resume_run}"]
    return cmd


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
parser.add_argument("--start", type=int, default=0, help="Schedule index to start from.")
parser.add_argument("--smoke", type=int, default=None, help="Override every entry's iterations.")
parser.add_argument("--num_envs", type=int, default=None, help="Forwarded to train and play.")
parser.add_argument("--resume-from", action="append", default=[], metavar="EXPERIMENT=RUN",
                    help="Pin an experiment's prior run; repeatable. Use with --start.")
args = parser.parse_args()

LAST_RUN: dict[str, str] = {}
for pair in args.resume_from:
    experiment, _, run_dir = pair.partition("=")
    LAST_RUN[experiment] = run_dir

for index, (task, iterations) in enumerate(SCHEDULE):
    spec = TASKS[task]
    iterations = args.smoke or iterations

    if index < args.start:
        pinned = LAST_RUN.get(spec["experiment"])
        done = (LOGS / spec["experiment"] / pinned) if pinned else newest_run(spec["experiment"])
        if done is not None:
            LAST_RUN[spec["experiment"]] = done.name
            POLICIES[spec["produces"]] = str(done.resolve() / "exported" / "policy.pt")
        continue

    if spec["consumes"] and spec["consumes"] not in POLICIES:
        sys.exit(f"\nStep {index} ({task}) needs {spec['consumes']}, which nothing has produced.")

    print(f"\n===== step {index}: {task}  ({iterations} iterations) =====", flush=True)

    resume_run = LAST_RUN.get(spec["experiment"])
    start_it = last_iteration(LOGS / spec["experiment"] / resume_run) if resume_run else None
    # rsl-rl runs range(start, start + N), so the last checkpoint is N-1.
    target = (start_it or 0) + iterations - 1
    train_cmd = train_command(task, index, iterations, resume_run)


    # play.py writes exported/policy.pt BEFORE its `while simulation_app
    # .is_running()` loop, and --video is the only thing that makes that loop
    # exit. --video_length=1 turns it into a one-step exporter.
    export_cmd = [sys.executable, str(PLAY), f"--task={task}", "--headless", "--video", "--video_length=1"]
    if args.num_envs:
        export_cmd.append(f"--num_envs={args.num_envs}")

    if args.dry_run:
        print("  " + " ".join(train_cmd) + "\n  " + " ".join(export_cmd), flush=True)
        # Stand-ins, so later steps print the --load_run they would really get.
        LAST_RUN[spec["experiment"]] = f"<step{index}-run>"
        POLICIES[spec["produces"]] = f"<step {index} export>"
        continue

    if run(train_cmd, f"train {task}", experiment=spec["experiment"]) != 0:
        crashed = newest_run(spec["experiment"])
        got = last_iteration(crashed) if crashed else None
        if got is None:                       # died before the first save
            retry_run, remaining = resume_run, iterations
        else:
            retry_run, remaining = crashed.name, target - got
        if remaining > 0:
            print(f"  crashed at {target - remaining}/{target}; retrying {remaining} from "
                  f"{retry_run}", flush=True)
            if run(train_command(task, index, remaining, retry_run),
                   f"retry {task}", experiment=spec["experiment"]) != 0:
                sys.exit(f"\ntrain {task} failed twice. Stopping.")
        else:
            print(f"  non-zero exit but reached {target}; continuing.", flush=True)

    if run(export_cmd, f"export {task}", timeout_s=1800) != 0:
        sys.exit(f"\nexport {task} failed. Stopping.")

    produced = newest_run(spec["experiment"])
    LAST_RUN[spec["experiment"]] = produced.name
    POLICIES[spec["produces"]] = str(produced.resolve() / "exported" / "policy.pt")

    print(f"  {spec['produces']} -> {POLICIES[spec['produces']]}", flush=True)

print(f"\n===== schedule complete ({len(SCHEDULE) - args.start} steps) =====")
