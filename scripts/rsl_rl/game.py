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


def run(cmd, label):
    print("  " + " ".join(f"{k}={v}" for k, v in POLICIES.items()), flush=True)
    print("  " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, env={**os.environ, **POLICIES}).returncode != 0:
        sys.exit(f"\n{label} exited non-zero. Stopping.")


def newest_run(experiment):
    runs = sorted((LOGS / experiment).glob("2*"), key=lambda p: p.name)
    return runs[-1] if runs else None


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
parser.add_argument("--start", type=int, default=0, help="Schedule index to start from.")
parser.add_argument("--smoke", type=int, default=None, help="Override every entry's iterations.")
parser.add_argument("--num_envs", type=int, default=None, help="Forwarded to train and play.")
args = parser.parse_args()

seen = set()

for index, (task, iterations) in enumerate(SCHEDULE):
    spec = TASKS[task]
    iterations = args.smoke or iterations

    if index < args.start:
        seen.add(spec["experiment"])
        done = newest_run(spec["experiment"])
        if done is not None:
            POLICIES[spec["produces"]] = str(done / "exported" / "policy.pt")
        continue

    if spec["consumes"] and spec["consumes"] not in POLICIES:
        sys.exit(f"\nStep {index} ({task}) needs {spec['consumes']}, which nothing has produced.")

    print(f"\n===== step {index}: {task}  ({iterations} iterations) =====", flush=True)

    train_cmd = [sys.executable, str(TRAIN), f"--task={task}", "--headless",
                 f"--max_iterations={iterations}",
                 f"--run_name=s{index:02d}-{task.removesuffix('-v0')}"]
    if args.num_envs:
        train_cmd.append(f"--num_envs={args.num_envs}")
    # Keyed on the experiment, not the task: Legs-R1 shares r0's log root, so
    # this is what warm-starts it from the r0 run.
    if spec["experiment"] in seen:
        train_cmd.append("--resume")
    seen.add(spec["experiment"])

    # play.py writes exported/policy.pt BEFORE its `while simulation_app
    # .is_running()` loop, and --video is the only thing that makes that loop
    # exit. --video_length=1 turns it into a one-step exporter.
    export_cmd = [sys.executable, str(PLAY), f"--task={task}", "--headless", "--video", "--video_length=1"]
    if args.num_envs:
        export_cmd.append(f"--num_envs={args.num_envs}")

    if args.dry_run:
        print("  " + " ".join(train_cmd) + "\n  " + " ".join(export_cmd), flush=True)
        POLICIES[spec["produces"]] = f"<step {index} export>"
        continue

    run(train_cmd, f"train {task}")
    run(export_cmd, f"export {task}")
    POLICIES[spec["produces"]] = str(newest_run(spec["experiment"]) / "exported" / "policy.pt")
    print(f"  {spec['produces']} -> {POLICIES[spec['produces']]}", flush=True)

print(f"\n===== schedule complete ({len(SCHEDULE) - args.start} steps) =====")
