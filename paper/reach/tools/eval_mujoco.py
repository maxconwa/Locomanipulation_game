"""Sim-to-sim: a checkpoint on the fixed goal grid in MuJoCo, the RoboCasa model GOLEM deploys into, without ROS.

    python paper/reach/tools/eval_mujoco.py --checkpoint <run>/checkpoints/agent_<N>.pt --label lambda1 \
        [--variant full|legs_blind|arms_ik] [--physics robocasa|isaac] [--odometry learned|true] \
        [--seeds 0] [--workers 8] [--limit N]

Same trial as eval_isaac.py: the task's reset randomization, SETTLE_S standing in navigation mode with a zero velocity
command, then one goal of the grid placed in the standing frame and held for GOAL_S; deterministic actors; the
learned pelvis estimator moves the command. The policy loop is reachlib/mujoco_sim.py (a port of the Isaac env's
observations, actions, PD and odometry). Physics: robocasa = the CL_Assets MJCF as GOLEM's RoboCasa sim loads it
(joint damping 10, armature 0.1, friction loss 0.2, 2 ms steps); isaac = those set to the training plant's values.

Writes results/<label>/mujoco[_<variant>][_<physics>][_trueodom]/{trials.csv, traces.npz, meta.json}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

_W = {}


def _init(args):
    import torch
    torch.set_num_threads(1)
    from reachlib.mujoco_sim import MarlMujoco, load_estimator
    from reachlib.policy import load_actors
    meta = json.loads((C.GOALS_DIR / "tables_meta.json").read_text())
    rest = np.array(meta["rest_pose_b"], dtype=float)
    est_path = None
    if args["odometry"] == "learned":
        sys.path.insert(0, str(C.DEFAULT_CODE / "source" / "locomanipulation_game"))
        est_path = _estimator_for(args["checkpoint"])
    sim = MarlMujoco(load_actors(args["checkpoint"]), load_estimator(est_path), physics=args["physics"],
                     rest_pose_b=rest)
    _W.update(sim=sim, args=args, est_path=est_path)


def _estimator_for(checkpoint: str) -> str | None:
    """The same rule as odometry.estimator_checkpoint_for, without importing Isaac Lab."""
    import glob
    import re
    run_dir = Path(checkpoint).resolve().parent.parent
    m = re.search(r"_(\d+)\.pt$", Path(checkpoint).name)
    if m and (run_dir / "estimator" / f"estimator_{m.group(1)}.pt").is_file():
        return str(run_dir / "estimator" / f"estimator_{m.group(1)}.pt")
    if (run_dir / "estimator" / "estimator_latest.pt").is_file():
        return str(run_dir / "estimator" / "estimator_latest.pt")
    saved = sorted(glob.glob(str(run_dir / "estimator" / "estimator_*.pt")), key=os.path.getmtime)
    return saved[-1] if saved else None


def _trial(job):
    from reachlib.metrics import trial_metrics
    sim, args = _W["sim"], _W["args"]
    seed, index, goal = job
    rng = np.random.default_rng(seed * 1_000_003 + index)
    sim.reset(rng)
    obs = sim.observations()
    blind, ik = args["variant"] == "legs_blind", args["variant"] == "arms_ik"
    settle, steps = int(round(C.SETTLE_S / C.POLICY_DT)), int(round(C.GOAL_S / C.POLICY_DT))
    fell_settle = False
    for _ in range(settle):
        obs = sim.step(obs, blind, ik)
        if sim.fell():
            fell_settle = True
            break
    h_start = sim.d.qpos[2] - 0.0
    poses = np.array([C.goal_pose(goal, "left"), C.goal_pose(goal, "right")], dtype=float)
    sim.set_goal(poses, use_estimate=args["odometry"] == "learned")
    obs = sim.observations()
    keys = ("pos_err", "rot_err", "root_pos", "root_quat", "root_ang_vel_b", "ground", "com", "com_vel", "feet_pos",
            "feet_quat", "feet_vel", "feet_force", "wrist_pos", "torque_ratio", "cmd_drift")
    log = {k: [] for k in keys}
    alive, state = [], []
    up = not fell_settle
    last = None
    for _ in range(steps):
        if up:
            obs = sim.step(obs, blind, ik)
            up = not sim.fell()
            last = sim.log_step()
            state.append(np.r_[sim.d.qpos[0:7], sim.d.qpos[sim.qadr]])
        alive.append(up)
        for k in keys:
            log[k].append(last[k] if last is not None else _blank(k))
    arr = {k: np.asarray(v, dtype=float)[:, None] for k, v in log.items()}
    arr["alive"] = np.asarray(alive)[:, None]
    row = trial_metrics(arr, np.array([h_start]), C.STANDING_PELVIS_HEIGHT)[0]
    trace = np.stack(state) if state else np.zeros((0, 34))
    return seed, index, row, trace


def _blank(k):
    shapes = {"pos_err": 2, "rot_err": 2, "root_pos": 3, "root_quat": 4, "root_ang_vel_b": 3, "ground": None,
              "com": 3, "com_vel": 3, "feet_pos": (2, 3), "feet_quat": (2, 4), "feet_vel": (2, 3), "feet_force": 2,
              "wrist_pos": (2, 3), "torque_ratio": 4, "cmd_drift": None}
    s = shapes[k]
    return 0.0 if s is None else np.zeros(s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--variant", choices=["full", "legs_blind", "arms_ik"], default="full")
    p.add_argument("--physics", choices=["robocasa", "isaac"], default="robocasa")
    p.add_argument("--odometry", choices=["learned", "true"], default="learned")
    p.add_argument("--goals", default=str(C.GOALS_DIR / "eval_goals_v1.csv"))
    p.add_argument("--seeds", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--trace_bases", type=int, default=12)
    a = p.parse_args()
    t0 = time.time()
    goals = C.read_goals(Path(a.goals))
    if a.limit:
        goals = goals[: a.limit]
    seeds = [int(s) for s in a.seeds.split(",")]
    suffix = "".join(s for s in ("" if a.variant == "full" else f"_{a.variant}",
                                 "" if a.physics == "robocasa" else f"_{a.physics}phys",
                                 "" if a.odometry == "learned" else "_trueodom") if s)
    out = C.RESULTS_DIR / a.label / f"mujoco{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(a.checkpoint).resolve())
    lam = C.read_reward_share(Path(checkpoint).parent.parent)
    args = {"checkpoint": checkpoint, "variant": a.variant, "physics": a.physics, "odometry": a.odometry}
    jobs = [(s, i, g) for s in seeds for i, g in enumerate(goals)]
    rows, traces = [], {}
    trace_ids = {g["goal_id"] for g in goals if g["base_id"] < a.trace_bases and g["ext"] == 0.0}
    print(f"[eval_mujoco] {checkpoint} lambda={lam} variant={a.variant} physics={a.physics}: {len(jobs)} trials,"
          f" {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(args,)) as pool:
        for n, (seed, index, row, trace) in enumerate(pool.imap_unordered(_trial, jobs, chunksize=2)):
            g = goals[index]
            rows.append({"label": a.label, "sim": "mujoco", "kind": "policy", "variant": a.variant,
                         "lambda_share": lam, "checkpoint": checkpoint, "seed": seed, "goal_id": g["goal_id"],
                         "base_id": g["base_id"], "depth": g["depth"], "ext": g["ext"],
                         "target_l_x": g["l_x"], "target_l_y": g["l_y"], "target_l_z": g["l_z"],
                         "target_r_x": g["r_x"], "target_r_y": g["r_y"], "target_r_z": g["r_z"],
                         "target_min_height": g["min_height"],
                         "needs_crouch": min(g["l_z"], g["r_z"]) < -0.109, **row})
            if seed == seeds[0] and g["goal_id"] in trace_ids:
                traces[g["goal_id"]] = trace
            if (n + 1) % 100 == 0 or n + 1 == len(jobs):
                rows.sort(key=lambda r: (r["seed"], r["goal_id"]))
                C.write_csv(out / "trials.csv", rows, C.TRIAL_COLUMNS + ["variant"])
                ok = np.mean([r["success"] for r in rows])
                print(f"[eval_mujoco] {n + 1}/{len(jobs)} trials, success {100 * ok:.1f}%, {time.time() - t0:.0f} s",
                      flush=True)
    np.savez_compressed(out / "traces.npz", **{f"trace_{k}": v for k, v in traces.items()})
    import mujoco
    import torch
    C.write_json(out / "meta.json", {
        "label": a.label, "sim": "mujoco", "variant": a.variant, "physics": a.physics, "odometry": a.odometry,
        "checkpoint": checkpoint, "checkpoint_sha256": C.sha256(Path(checkpoint)), "lambda_share": lam,
        "estimator": _estimator_for(checkpoint) if a.odometry == "learned" else None,
        "goals": str(Path(a.goals).resolve()), "goals_sha256": C.sha256(Path(a.goals)), "seeds": seeds,
        "n": len(rows), "mjcf": str(C.DEFAULT_ASSETS / "mujoco_assets" / "h1_2_magpie.xml"),
        "versions": {"mujoco": mujoco.__version__, "torch": torch.__version__},
        "wall_s": round(time.time() - t0, 1),
    })
    print(f"[eval_mujoco] wrote {out} ({len(rows)} trials, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
