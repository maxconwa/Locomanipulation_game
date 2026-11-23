"""The rollout and scoring that ``crosscheck.py`` and ``play.py`` share: stage-1 actors tracking clips in the Isaac env.

One env per clip, all clips of a batch rolled at once. Each env starts on its clip's first frame (the reference pose
and velocities, no perturbation) and is scored until its clip ends (``min(frames - 1, seconds / dt)`` steps, as
h12_train's ``render_stage1.py``), it falls, or one of the env's own terminations (``anchor_pos``, ``anchor_ori``)
resets it; the env is then recorded as ended there, with the reason, and never scored again.

The actors are fed as ``ImitationRunner`` collection feeds them, without the proprio noise: the history is the env's
``actor`` frame (87), the legs goal ``legs_goal(motion, time_steps, full_state[:, :7])`` and the arms goal
``arms_goal(tables, time_steps)``; the action is the actors' mean (they are deterministic).

The errors are ``render_stage1.py``'s, against the reference at the frame after each step: the tracked bodies' mean
world position error (m), the pelvis's world position error (m; the anchor is the pelvis) and the mean absolute joint
error over the 27 policy joints (rad). ``Safety/*`` is the share of scored steps on which each estop flag of the
training runner's (``runner.safety_flags``) was set.

The env runs nominal, as ``render_stage1.py``'s MuJoCo: foot friction h12_train's 0.6, no torso COM offset, no
encoder bias. Everything here needs the running app.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict

from pretraining.marl.config import AGENTS
from pretraining.marl.rewards import JointLimits
from pretraining.marl.runner import safety_flags
from pretraining.marl.wm_env import ClipTables, arms_goal, legs_goal
from pretraining.constants import JOINT_NAMES
from pretraining.supermimic.representations import GenCoordRepresentation

PLAY_TASK = "LocoManip-Pretrain-Track-Play-v0"
# render_stage1.py's fall rule (stage 1's sim terminations' thresholds)
FALL_HEIGHT = 0.6  # m, the pelvis's height above the ground
FALL_TILT = 1.0  # rad, the pelvis's tilt from upright
NOMINAL_FOOT_FRICTION = 0.6  # h12_train's foot geoms' (h12_constants)
MAX_SECONDS = 10.0
ENV_TERMINATIONS = ("anchor_pos", "anchor_ori")
ERROR_COLUMNS = ("body_pos_err_m", "pelvis_pos_err_m", "joint_err_rad")


def choose_clips(
    motion_dir: Path, n: int, seed: int, include: list[str], clips_file: Path | None = None
) -> list[Path]:
    """Return the clips to roll: those named in ``clips_file`` (one name per line), else the ``include`` names found
    in ``motion_dir`` followed by clips drawn with ``seed`` from the rest, ``n`` in all."""
    motion_dir = Path(motion_dir)
    if clips_file is not None:
        names = Path(clips_file).read_text().split()
        paths = [motion_dir / f"{c}.npz" for c in names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{clips_file}: {len(missing)} clips not in {motion_dir}, e.g. {missing[:3]}")
        return paths
    available = sorted(p.stem for p in motion_dir.glob("*.npz"))
    known = set(available)
    chosen = []
    for c in include:
        if c in known and c not in chosen:
            chosen.append(c)
        elif c not in known:
            print(f"[rollout] --include {c}: not in {motion_dir}, skipped")
    if len(chosen) > n:
        raise ValueError(f"{len(chosen)} included clips but only {n} to roll")
    rest = [c for c in available if c not in set(chosen)]
    chosen += random.Random(seed).sample(rest, n - len(chosen))
    return [motion_dir / f"{c}.npz" for c in chosen]


def make_env(files: list[Path], num_envs: int, device: str, seed: int, task: str = PLAY_TASK, render_mode=None):
    """Create the play env on ``files`` only (never a whole dataset: its body tensors fill the GPU), nominal."""
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    motion = cfg.commands.motion
    motion.motion_file = [str(f) for f in files]
    motion.sampling_mode = "start"
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    cfg.events.foot_friction.params["friction_range"] = (NOMINAL_FOOT_FRICTION, NOMINAL_FOOT_FRICTION)
    cfg.events.base_com = None
    cfg.events.encoder_bias = None
    return gym.make(task, cfg=cfg, render_mode=render_mode)


def _read_state(env) -> dict[str, torch.Tensor]:
    """The scored quantities of every env, as the sim holds them now (copies)."""
    from pretraining.tasks.events import encoder_bias

    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")
    ids = cmd.joint_ids
    return {
        "pelvis_pos": cmd.robot_anchor_pos_w - env.scene.env_origins,
        "pelvis_quat": cmd.robot_anchor_quat_w.clone(),
        "body_pos": cmd.robot_body_pos_w - env.scene.env_origins[:, None],
        "q": robot.data.joint_pos[:, ids].clone(),
        "dq": robot.data.joint_vel[:, ids].clone(),
        # the actuator's target: the clipped processed action less the encoder bias
        "target": env.action_manager.get_term("joint_pos")._processed_actions - encoder_bias(env),
        "torque": robot.data.applied_torque[:, ids].clone(),
    }


def fell(pelvis_pos: torch.Tensor, pelvis_quat: torch.Tensor) -> torch.Tensor:
    """The fall rule: the pelvis below ``FALL_HEIGHT`` or tilted more than ``FALL_TILT`` (quaternions wxyz)."""
    x, y = pelvis_quat[:, 1], pelvis_quat[:, 2]
    up = 1.0 - 2.0 * (x * x + y * y)  # the pelvis z axis's world z component
    return (pelvis_pos[:, 2] < FALL_HEIGHT) | (up < math.cos(FALL_TILT))


def rollout(env, actors: dict, history_length: int, seconds: float = MAX_SECONDS) -> list[dict]:
    """Roll every clip of ``env``'s motion (clip ``k`` on env ``k mod num_envs``, in batches of ``num_envs``) and
    return one row per clip. ``env`` is the gym env (it may be wrapped, e.g. by ``RecordVideo``)."""
    u = env.unwrapped
    cmd = u.command_manager.get_term("motion")
    motion = cmd.motion
    tables = ClipTables(motion)
    limits = JointLimits.for_joints(JOINT_NAMES, u.device)
    dt = u.step_dt
    cap = int(round(seconds / dt))
    num_clips = len(motion.files)

    # The state an env had when a termination reset it, read before the reset overwrote it.
    snapshot: dict[str, dict] = {}
    reset_idx = u._reset_idx

    def _reset_idx(env_ids):
        snapshot["state"] = _read_state(u)
        reset_idx(env_ids)

    u._reset_idx = _reset_idx
    env.reset()
    rows = []
    try:
        for first in range(0, num_clips, u.num_envs):
            clips = list(range(first, min(first + u.num_envs, num_clips)))
            rows += _roll_batch(env, u, cmd, motion, tables, limits, actors, history_length, clips, cap, dt, snapshot)
    finally:
        u._reset_idx = reset_idx
    return rows


def _roll_batch(env, u, cmd, motion, tables, limits, actors, history_length, clips, cap, dt, snapshot) -> list[dict]:
    device = u.device
    n = len(clips)
    ids = torch.arange(n, device=device)
    starts = motion.motion_starts[clips].to(device)
    lengths = (motion.motion_ends[clips] - motion.motion_starts[clips]).to(device)
    budget = torch.clamp(lengths - 1, max=cap)  # scored steps per clip
    for k in range(n):
        cmd.reset_to_frame(ids[k : k + 1], int(starts[k]))
    cmd.refresh_robot_state()
    cmd.update_relative_body_poses()
    u.action_manager.reset(ids)
    u.episode_length_buf[ids] = 0
    obs = u.observation_manager.compute()
    history = obs["actor"].unsqueeze(1).repeat(1, history_length, 1)

    active = torch.ones(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, dtype=torch.long, device=device)
    sums = {c: torch.zeros(n, device=device) for c in ERROR_COLUMNS}
    safety: dict[str, torch.Tensor] = {}
    ended_by = [""] * n
    fell_at: list[float | None] = [None] * n
    names = [Path(motion.files[c]).stem for c in clips]
    for t in range(int(budget.max())):
        with torch.inference_mode():
            full = obs["full_state"]
            times = cmd.time_steps.clone()
            goals = {
                # the pelvis pose [pos - env origin, quat]: GenCoordRepresentation.extract_aux_state, as collection
                "legs": legs_goal(motion, times, GenCoordRepresentation.extract_aux_state(None, full)),
                "arms": arms_goal(tables, times),
            }
            m = u.num_envs
            actions = torch.cat(
                [
                    actors[a](TensorDict({"history": history, "goal_motion": goals[a]}, batch_size=[m], device=device))
                    for a in AGENTS
                ],
                dim=-1,
            )
            nj = len(JOINT_NAMES)
            q, dq = full[:, 7 : 7 + nj].clone(), full[:, 13 + nj :].clone()
            snapshot.clear()
            obs, _, terminated, truncated, _ = env.step(actions)
            state = _read_state(u)
            done = (terminated | truncated)[:n]
            if done.any():
                for key, value in snapshot["state"].items():
                    state[key][done.nonzero()[:, 0]] = value[done.nonzero()[:, 0]]
            state = {k: v[:n] for k, v in state.items()}

            # the reference after the step (clamped: a clip that has ended is no longer scored)
            frames = torch.minimum(starts + t + 1, starts + lengths - 1)
            ref_body = motion.body_pos_w[frames]
            err = {
                "body_pos_err_m": torch.norm(state["body_pos"] - ref_body, dim=-1).mean(-1),
                "pelvis_pos_err_m": torch.norm(state["pelvis_pos"] - motion.body_pos_w[frames, cmd.motion_anchor_body_index], dim=-1),
                "joint_err_rad": (state["q"] - motion.joint_pos[frames]).abs().mean(-1),
            }
            flags = safety_flags(
                limits, q[:n], dq[:n], state["q"], state["dq"], state["target"], state["torque"]
            )
            a = active.float()
            for c in ERROR_COLUMNS:
                sums[c] += err[c] * a
            for key, v in flags.items():
                safety[key] = safety.get(key, torch.zeros(n, device=device)) + v.float() * a
            steps += active.long()
            down = fell(state["pelvis_pos"], state["pelvis_quat"])
            term = {name: u.termination_manager.get_term(name)[:n] for name in ENV_TERMINATIONS}
            for k in active.nonzero()[:, 0].tolist():
                if down[k]:
                    ended_by[k], fell_at[k] = "fall", (t + 1) * dt
                elif term["anchor_pos"][k]:
                    ended_by[k] = "anchor_pos"
                elif term["anchor_ori"][k]:
                    ended_by[k] = "anchor_ori"
                elif done[k]:
                    raise RuntimeError(f"env {k} ({names[k]}) reset by another termination at step {t}")
                elif steps[k] == budget[k]:
                    ended_by[k] = "clip_end" if budget[k] == lengths[k] - 1 else "time_cap"
                if ended_by[k]:
                    active[k] = False
            if history_length > 1:
                history = torch.cat([history[:, 1:], obs["actor"].unsqueeze(1)], dim=1)
            else:
                history = obs["actor"].unsqueeze(1)
        if not active.any():
            break

    rows = []
    for k in range(n):
        s = int(steps[k])
        row = {
            "clip": names[k],
            "steps": s,
            "clip_s": round(float(budget[k]) * dt, 4),
            "ended_by": ended_by[k],
            "fell": int(fell_at[k] is not None),
            "fell_at_s": "" if fell_at[k] is None else round(fell_at[k], 2),
            "time_upright_s": round(s * dt, 4),
        }
        row.update({c: float(sums[c][k]) / s for c in ERROR_COLUMNS})
        row.update({f"Safety/{key}": float(v[k]) / s for key, v in safety.items()})
        rows.append(row)
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    """Write the rows as a csv (columns of the first row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(label: str, rows: list[dict]) -> str:
    """One line: clips, falls, terminations, mean errors."""
    n = len(rows)
    falls = sum(r["fell"] for r in rows)
    terms = sum(r["ended_by"] in ENV_TERMINATIONS for r in rows)
    mean = {c: sum(r[c] for r in rows) / n for c in ERROR_COLUMNS}
    return (
        f"{label}: {n} clips, {falls} fell ({100.0 * falls / n:.0f}%), {terms} ended by an env termination; "
        f"mean body pos err {mean['body_pos_err_m']:.3f} m, pelvis pos err {mean['pelvis_pos_err_m']:.3f} m, "
        f"joint err {mean['joint_err_rad']:.3f} rad"
    )
