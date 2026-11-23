"""The pinned eval conditions, stamped into every scorecard so plots only join points taken the same way.

DEFAULTS holds, per backend and body, what the scorecards run with unless a flag overrides it. The legs' are the ones
the 200k runs' history was scored with (logs/candidates/200k_v2/progress.sh: 84 envs, no wrist phase, nominal motors,
seed 7); that history, imported with the arms held by their IK (not at 0 rad), still differs in its stamp and is drawn
apart (hollow) in the plots. stamp() records the conditions, a hash of them (the plots join points
with equal hashes) and the code's commit (not hashed: the env's code changes between runs, the eval doesn't).
"""

from __future__ import annotations

import hashlib
import json
import subprocess

from .outputs import REPO

# the actuator overrides for nominal motors (Isaac hydra syntax)
NOMINAL_MOTORS = ["env.events.actuator_scales.params.gain_range=[1.0,1.0]",
                  "env.events.actuator_scales.params.strength_range=[1.0,1.0]",
                  "env.events.actuator_scales.params.joint_gain_ranges=null"]

DEFAULTS = {
    ("isaac", "legs"): dict(num_envs=84, segments_per_env=6, seed=7, wrist_goals=0, motors="nominal",
                            actions="mean"),
    ("isaac", "arms"): dict(num_envs=64, goals=8, seed=7, motors="run", actions="mean", walk=False,
                            offtable=False),
    ("mujoco", "legs"): dict(num_envs=84, segments_per_env=6, seed=7, plant="isaac", actions="mean"),
    ("mujoco", "arms"): dict(num_envs=32, goals=8, seed=7, plant="isaac", actions="mean"),
}


def commit() -> str:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def conditions_hash(conditions: dict) -> str:
    return hashlib.sha1(json.dumps(conditions, sort_keys=True, default=str).encode()).hexdigest()[:10]


def stamp(backend: str, body: str, **conditions) -> dict:
    """{"conditions", "conditions_hash", "code_commit"} for a result dict."""
    c = dict(backend=backend, body=body, **conditions)
    return {"conditions": c, "conditions_hash": conditions_hash(c), "code_commit": commit()}
