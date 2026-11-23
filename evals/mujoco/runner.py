"""The MuJoCo scorecards' episodes: one robot per job (jobs in parallel processes), recorded at the policy rate in the
arrays evals/common/legs_scorecard.py and arms_scorecard.py score.

Arm goals: rows of the arms export's own goal table (arm_seed_table: training's table for that task, in the pelvis
frame), else of the walking task's tables (data/arm_target_tables_v3.pt, the standing frame, taken as the pelvis
frame: upright at the standing height they differ by millimetres), one row per wrist drawn at random.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import common as C
from .policy import actor
from .sim import MarlMujoco, qconj, qmul

TABLE_FILE = C.REPO / "source" / "locomanipulation_game" / "locomanipulation_game" / "data" / "arm_target_tables_v3.pt"
FALL_TILT, FALL_HEIGHT = 1.0, 0.55   # rad, m: the episode ends (as the env's fall termination)
CONTACT_N = 5.0                      # N: a foot counts as down (as the Isaac scorecard's contact sensor threshold)


def goal_table(settings: dict) -> list[np.ndarray]:
    """Per arm, (rows, 7) pelvis-frame goal poses (xyz + wxyz)."""
    if settings.get("arm_seed_table"):
        t = np.load(settings["arm_seed_table"])
        return [t[f"poses_{i}"].astype(float) for i in range(2)]
    return [t.numpy().astype(float) for t in torch.load(str(TABLE_FILE), map_location="cpu", weights_only=False)["tables"]]


def draw_goal(tables, rng) -> np.ndarray:
    return np.stack([t[rng.integers(len(t))] for t in tables])


def make_sim(settings, legs_export, arms_export, plant, pinned=False, hold_zero=False) -> MarlMujoco:
    actors = {}
    if legs_export is not None:
        actors["legs"] = actor(legs_export, "legs")
    if arms_export is not None:
        actors["arms"] = actor(arms_export, "arms")
    model = None
    if pinned:
        from .sim import pinned_model
        model = pinned_model()
    return MarlMujoco(actors, settings, physics=plant, model=model, pinned=pinned, arms_hold_zero=hold_zero)


def reset_actors(sim):
    for a in sim.actors.values():
        a.reset_state()


def yaw_frame_velocity(sim) -> np.ndarray:
    """(vx, vy) of the pelvis in its yaw frame and the world yaw rate."""
    d = sim.d
    q = d.qpos[3:7]
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    v = d.qvel[0:3]
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, q)
    wz = (R.reshape(3, 3) @ d.qvel[3:6])[2]
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], wz])


def torso_rp(sim, torso_id: int) -> np.ndarray:
    w, x, y, z = sim.d.xquat[torso_id]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return np.array([roll, pitch])


def fell(sim) -> bool:
    return sim.fell() or sim.root()[0][2] < FALL_HEIGHT


def wrist_errors(sim) -> tuple[np.ndarray, np.ndarray]:
    """Each wrist's position (m) and rotation (rad) error against the believed command (pelvis frame; with the
    pelvis's own motion tracked, the true target)."""
    cur = sim.wrist_poses_b()
    pos = np.linalg.norm(sim.believed[:, :3] - cur[:, :3], axis=1)
    rot = np.array([2 * math.acos(min(1.0, abs(qmul(b[3:], qconj(c[3:]))[0]))) for b, c in zip(sim.believed, cur)])
    return pos, rot
