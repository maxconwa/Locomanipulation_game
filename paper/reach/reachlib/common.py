"""Paths, task constants, the condition registry and the results layout shared by every tool in paper/reach.

Layout (all relative to paper/reach):
    conditions.yaml                 which controllers exist: label, kind, run directory, display name
    goals/eval_goals_<v>.csv        the fixed goal grid every controller is scored on
    results/<label>/<sim>/          trials.csv (one row per trial), traces.npz, meta.json
    figures/, tex/                  outputs of analyze.py / figures.py / build_tex.py

Constants are the task's (LocoManip-Marl-Direct-v0 at marl-direct 72171c7) and are checked against the env at
evaluation time where the env exposes them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REACH = Path(__file__).resolve().parents[1]
PAPER = REACH.parent
GOALS_DIR = REACH / "goals"
RESULTS_DIR = REACH / "results"
FIG_DIR = REACH / "figures"
TEX_DIR = REACH / "tex"
CONDITIONS_FILE = REACH / "conditions.yaml"

# The task's code is the checkout this pipeline sits in (override with LOCOMANIP_CODE). Assets: CL_ASSETS_DIR, else
# the checkout's third_party/CL_Assets submodule when it holds the robot USD, else the workstation's pinned clone.
DEFAULT_CODE = Path(os.environ.get("LOCOMANIP_CODE", str(REACH.parents[1])))


def _assets() -> Path:
    if os.environ.get("CL_ASSETS_DIR"):
        return Path(os.environ["CL_ASSETS_DIR"])
    sub = DEFAULT_CODE / "third_party" / "CL_Assets"
    if (sub / "isaac_assets" / "robots" / "h1_2_magpie" / "h1_2_magpie.usd").is_file():
        return sub
    return Path.home() / "Programs" / "CL_Assets-cf87bfe"


DEFAULT_ASSETS = _assets()

# -- task constants (locomanip_marl_env_cfg.py, commands.py, assets/h1_2.py)
STANDING_PELVIS_HEIGHT = 1.0024   # m, pelvis above the ground when standing: the standing frame's origin
POLICY_DT = 0.02                  # s, 50 Hz policy, physics 200 Hz
POS_TOL = 0.05                    # m, per wrist, the reach test's position tolerance
ROT_TOL = 0.35                    # rad, per wrist, the reach test's orientation tolerance
HOLD_S = 1.0                      # s, both wrists held inside the tolerances
GOAL_S = 4.0                      # s, one arm goal
SETTLE_S = 2.0                    # s, standing (navigation, zero velocity) before the goal
# Shoulder pitch joint origins in the pelvis frame at torso yaw 0 (URDF: torso_joint at the pelvis origin,
# left_shoulder_pitch_joint xyz 0 0.14806 0.42333): the centre of the horizontal reach extension.
SHOULDER_B = {"left": (0.0, 0.14806, 0.42333), "right": (0.0, -0.14806, 0.42333)}
# Sole rectangle of each foot in its ankle_roll_link frame (bounding box of the collision mesh's lowest 5 mm).
SOLE_X = (-0.086, 0.174)
SOLE_Y = (-0.043, 0.043)
SOLE_Z = -0.045
CONTACT_FORCE_N = 20.0            # a foot counts as in contact above this normal force
STANCE_FORCE_N = 50.0             # ... and as planted (for slip) above this one (the task's leg-odometry threshold)
ARMS = ("left", "right")

# Columns every trials.csv carries; evaluators may add more. NaN where a quantity does not apply (kinematic
# baselines have no dynamics).
TRIAL_COLUMNS = [
    "label", "sim", "kind", "lambda_share", "checkpoint", "seed", "goal_id", "base_id", "depth", "ext",
    "target_l_x", "target_l_y", "target_l_z", "target_r_x", "target_r_y", "target_r_z",
    "target_min_height",            # m above the ground, the lower of the two wrist targets
    "needs_crouch",                 # a target below everything the standing table reaches
    "success",                      # both wrists inside 5 cm / 0.35 rad for 1 s
    "success_l", "success_r",       # each wrist on its own for 1 s
    "t_success",                    # s from goal start to the end of the first qualifying 1 s hold
    "closest",                      # m, least mean of both wrists' position errors over the goal
    "hold_err_pos", "hold_err_rot", # m / rad: the best 1 s window's worst wrist error (success at tol tau: < tau)
    "success_2cm", "success_3cm", "success_8cm", "success_10cm",   # strict success at 2/3/8/10 cm, 0.2-0.6 rad
    "err_final_l", "err_final_r", "rot_final_l", "rot_final_r",
    "err_last1s", "rot_last1s",     # m / rad, mean over both wrists and the goal's last second
    "pelvis_h_start", "pelvis_h_min", "pelvis_h_last1s",
    "pelvis_drop_last1s",           # m, pelvis_h_start - pelvis_h_last1s
    "fell", "t_fall",
    "tilt_max_deg", "angvel_rms",   # base tilt from vertical; RMS of roll and pitch rate (rad/s)
    "com_margin_min", "com_margin_last1s",   # m, CoM ground projection inside the support polygon (+ inside)
    "dcm_margin_min", "dcm_margin_last1s",   # m, the same for the divergent component of motion
    "foot_slip_max", "steps",       # m/s, planted-foot horizontal speed; count of foot lift-offs
    "ee_jitter_l", "ee_jitter_r",   # mm, RMS deviation of the wrist position about its mean, last second
    "ee_speed_rms",                 # m/s, RMS wrist speed over the last second, both wrists
    "torque_knee_max", "torque_ankle_max", "torque_hip_max", "torque_arm_max",  # peak |tau| / effort limit
    "cmd_drift_end",                # m, believed (odometry) vs true target at the goal's end
    "pitch_last1s_deg",             # pelvis pitch, last-second mean (+ leaning forward)
]


def git_info(path: Path) -> dict:
    """Commit, branch and dirty flag of the checkout at path (or Nones)."""
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "describe": run("describe", "--always", "--dirty"),
            "status": run("status", "--short")}


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _yaml_load_any(text: str):
    """yaml.safe_load that turns the python/* tags Isaac Lab's dump_yaml writes into plain values."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def any_tag(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", any_tag)
    Loader.add_multi_constructor("!", any_tag)
    return yaml.load(text, Loader=Loader)


def read_reward_share(run_dir: Path) -> float | None:
    """lambda of a run from its params/env.yaml: the legs' weight on arms_left_ee_pos over the arms' own weight.

    The task applies one scalar REWARD_SHARE in both directions (locomanip_marl_env_cfg.MarlRewardsCfg); the two
    directions are read separately and must agree.
    """
    path = Path(run_dir) / "params" / "env.yaml"
    if not path.is_file():
        return None
    cfg = _yaml_load_any(path.read_text())
    rewards = cfg.get("rewards", {})
    try:
        to_legs = rewards["legs"]["arms_left_ee_pos"]["weight"] / rewards["arms"]["left_ee_pos"]["weight"]
        to_arms = rewards["arms"]["legs_track_lin_vel_xy"]["weight"] / rewards["legs"]["track_lin_vel_xy"]["weight"]
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    if abs(to_legs - to_arms) > 1e-6:
        print(f"[reach] {run_dir}: asymmetric sharing, legs <- arms {to_legs}, arms <- legs {to_arms}; using legs <- arms")
    return round(float(to_legs), 6)


def newest_checkpoint(run_dir: Path) -> Path | None:
    """<run>/checkpoints/agent_<N>.pt with the largest N (skrl's best_agent.pt only when nothing else exists)."""
    ckpts = list((Path(run_dir) / "checkpoints").glob("agent_*.pt"))
    if ckpts:
        return max(ckpts, key=lambda p: int(re.search(r"_(\d+)\.pt$", p.name).group(1)))
    best = Path(run_dir) / "checkpoints" / "best_agent.pt"
    return best if best.is_file() else None


def load_conditions() -> list[dict]:
    """conditions.yaml entries with run paths resolved relative to paper/reach."""
    import yaml
    if not CONDITIONS_FILE.is_file():
        return []
    entries = yaml.safe_load(CONDITIONS_FILE.read_text()).get("conditions", []) or []
    for e in entries:
        if e.get("run"):
            e["run"] = str((REACH / e["run"]).resolve()) if not os.path.isabs(e["run"]) else e["run"]
    return entries


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.6g}"
    if isinstance(v, bool):
        return int(v)
    return v


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_goals(path: Path) -> list[dict]:
    """The goal grid as dicts with float fields; poses are (x, y, z, qw, qx, qy, qz) in the standing frame."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        g = {k: (float(v) if k not in ("goal_id",) else v) for k, v in r.items()}
        g["base_id"] = int(g["base_id"])
        out.append(g)
    return out


def goal_pose(goal: dict, arm: str) -> list[float]:
    p = "l" if arm == "left" else "r"
    return [goal[f"{p}_{k}"] for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
