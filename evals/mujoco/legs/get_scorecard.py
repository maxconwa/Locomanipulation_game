"""The legs scorecard in MuJoCo: the Isaac legs scorecard's segment program and scoring (evals/common/
legs_scorecard.py) on the exported policy in MuJoCo (evals/mujoco/sim.py, the deploy's settings), one robot per job.

    python -m evals.mujoco.legs.get_scorecard --checkpoint <run>/checkpoints/agent_<N>.pt [--arms_checkpoint <ckpt>|same]
    python -m evals.mujoco.legs.get_scorecard --run <run> [--steps all|last|<N>,...]
    python -m evals.mujoco.legs.get_scorecard --export <export dir> --out <base>      (an export without its run)
        [--num_envs 84] [--plant isaac|robocasa] [--workers 12] [--force]

A checkpoint is exported once (scripts/skrl/export_marl.py) into logs/evals/<run>/mujoco/exports/. Without an arms
policy every arm joint is held at 0 rad; with one, its arms run their own goals (a new one every GOAL_EVERY_S, riding
on the pelvis as in LocoManip-Marl-Direct-v0) and the scorecard adds the "Coupled" section; the two exports must agree
on the settings both bodies live by (evals/mujoco/policy.py). Plant: isaac (the training plant's joint dynamics, the
default) or robocasa (GOLEM's sim's). No GOLEM e-stop model. CPU only: any python with mujoco, torch, numpy, scipy,
pyyaml.

Writes logs/evals/<run>/mujoco/legs/scorecard_<N>[_robocasa][_with_<arms run>_<M>].{json,md}.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import conditions as C  # noqa: E402
from evals.common import legs_scorecard as L  # noqa: E402
from evals.common import outputs, runs, scorecard  # noqa: E402

D = C.DEFAULTS[("mujoco", "legs")]
GOAL_EVERY_S = 5.0


def job(spec):
    """One robot through its command sequence. Returns its recording and the step it fell at (T: never)."""
    import mujoco
    import torch

    from evals.mujoco import runner as R
    torch.set_num_threads(1)
    e, cmds, settings, legs_export, arms_export, plant, seed = spec
    sim = R.make_sim(settings, legs_export, arms_export, plant, hold_zero=arms_export is None)
    rng = np.random.default_rng(seed + e)
    sim.reset(rng)
    R.reset_actors(sim)
    torso = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    coupled = arms_export is not None
    if coupled:
        sim.pelvis_goals, tables, goal_rng = True, R.goal_table(settings), np.random.default_rng(seed + 10_000 + e)
        every = int(round(GOAL_EVERY_S / sim.policy_dt))
    T = len(cmds)
    rec = {"vel": np.zeros((T, 3), np.float32), "q": np.zeros((T, 12), np.float32), "rp": np.zeros((T, 2), np.float32),
           "foot_z": np.zeros((T, 2), np.float32), "contact": np.zeros((T, 2), bool),
           "arm_pos": np.zeros((T, 2), np.float32), "arm_rot": np.zeros((T, 2), np.float32), "arm_mode": np.zeros(T, bool)}
    obs = sim.observations()
    fell_at = T
    for k in range(T):
        sim.vel_cmd = cmds[k]
        if coupled and k % every == 0:
            sim.arm_mode, sim.believed = True, R.draw_goal(tables, goal_rng)
        obs = sim.step(obs)
        rec["vel"][k] = R.yaw_frame_velocity(sim)
        rec["q"][k] = sim.d.qpos[sim.qadr[:12]]
        rec["rp"][k] = R.torso_rp(sim, torso)
        rec["foot_z"][k] = [sim.d.xpos[f][2] for f in sim.feet]
        rec["contact"][k] = sim.foot_forces() > R.CONTACT_N
        if coupled:
            rec["arm_pos"][k], rec["arm_rot"][k] = R.wrist_errors(sim)
            rec["arm_mode"][k] = sim.arm_mode
        if R.fell(sim):
            fell_at = k
            break
    return e, rec, fell_at, sim.policy_dt, float(settings["gait_period"])


def coupled_rows(pos, rot, mode, fell_at):
    valid = np.arange(pos.shape[1])[None] < fell_at[:, None]
    m = mode & valid
    p, r = (pos[m].mean(-1), rot[m].mean(-1)) if m.any() else (np.array([]), np.array([]))
    pct = (lambda x, q, s: float(np.percentile(x, q) * s) if len(x) else float("nan"))
    sec = "Coupled: the arms policy's own goals while the legs walk"
    return [dict(section=sec, key="coupled.goal_share", label="Steps with an arm goal active, share",
                 value=float(m.sum() / max(valid.sum(), 1))),
            dict(section=sec, key="coupled.pos_err_median_mm", label="Wrist position error, median (mm)",
                 value=pct(p, 50, 1000), digits=1),
            dict(section=sec, key="coupled.pos_err_p90_mm", label="Wrist position error, p90 (mm)",
                 value=pct(p, 90, 1000), digits=1),
            dict(section=sec, key="coupled.rot_err_median_deg", label="Wrist rotation error, median (deg)",
                 value=pct(r, 50, 180 / np.pi), digits=1)]


def out_base(checkpoint, a) -> str:
    run_dir = runs.run_dir_of(checkpoint)
    suffix = "" if a.plant == "isaac" else f"_{a.plant}"
    if a.arms_checkpoint:
        other = run_dir if a.arms_checkpoint == "same" else runs.run_dir_of(a.arms_checkpoint)
        suffix += f"_with_{runs.short(other)}_{runs.step_of(checkpoint if a.arms_checkpoint == 'same' else a.arms_checkpoint)}"
    return str(outputs.base(runs.name(run_dir), "mujoco", "legs", "scorecard", runs.step_of(checkpoint), suffix))


def score_one(a, checkpoint: str | None):
    from evals.mujoco.policy import combined_settings, export_for
    if checkpoint:
        run_dir = runs.run_dir_of(checkpoint)
        legs_export = export_for(checkpoint, outputs.eval_dir(runs.name(run_dir), "mujoco", "exports"))
        base = a.out or out_base(checkpoint, a)
    else:
        run_dir, legs_export, base = None, Path(a.export), a.out
    if outputs.cached(base) and not a.force:
        print(f"[evals] cached: {base}.json")
        return 0
    arms_export, arms_ckpt = None, None
    if a.arms_checkpoint:
        arms_ckpt = checkpoint if a.arms_checkpoint == "same" else os.path.abspath(a.arms_checkpoint)
        arms_run = runs.run_dir_of(arms_ckpt)
        arms_export = export_for(arms_ckpt, outputs.eval_dir(runs.name(arms_run), "mujoco", "exports"))
    elif a.arms_export:
        arms_export = Path(a.arms_export)
    settings = combined_settings(legs_export, arms_export)
    dt = float(settings["control_dt"])
    r = settings.get("command_ranges") or {}
    full = tuple(float(max(abs(v[0]), abs(v[1]))) for v in (r.get("lin_vel_x", [-0.55, 0.55]),
                                                            r.get("lin_vel_y", [-0.54, 0.54]),
                                                            r.get("ang_vel_z", [-0.97, 0.97])))
    rng = np.random.default_rng(a.seed)
    cmds, kinds = L.program(a.num_envs, a.segments_per_env, dt, rng, full)
    specs = [(e, cmds[e], settings, str(legs_export), str(arms_export) if arms_export else None, a.plant, a.seed)
             for e in range(a.num_envs)]
    print(f"[evals] mujoco legs scorecard: {a.num_envs} robots x {cmds.shape[1] * dt:.0f} s, plant {a.plant},"
          f" {a.workers} workers; arms {'policy ' + str(arms_export) if arms_export else 'held at 0 rad'}", flush=True)
    with Pool(a.workers) as pool:
        results = sorted(pool.map(job, specs), key=lambda x: x[0])
    N, T = cmds.shape[:2]
    rec = {k: np.stack([res[1][k] for res in results]) for k in results[0][1]}
    fell_at = np.array([res[2] for res in results])
    out = L.compute(cmds, kinds, {k: rec[k] for k in ("vel", "q", "rp", "foot_z", "contact")}, fell_at, dt,
                    results[0][4], None)
    if arms_export is not None:
        out["coupled_rows"] = coupled_rows(rec["arm_pos"], rec["arm_rot"], rec["arm_mode"], fell_at)
    out.update(backend="mujoco", body="legs", run=runs.name(run_dir) if run_dir else str(legs_export),
               step=runs.step_of(checkpoint) if checkpoint else None, checkpoint=checkpoint, export=str(legs_export),
               plant=a.plant, full=list(full), overrides=[], other_policy=(f"arms policy {arms_export}" if arms_export else None),
               **C.stamp("mujoco", "legs", num_envs=N, segments_per_env=a.segments_per_env, seed=a.seed, plant=a.plant,
                         actions="mean", arms="policy" if arms_export else "zero"))
    scorecard.write(base, out, L.rows(out), L.header(out), L.footer(out))
    print(open(base + ".md").read())
    print(f"[evals] wrote {base}.json / .md")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--run")
    p.add_argument("--steps", default="all")
    p.add_argument("--export", help="Score an export folder (with --out).")
    p.add_argument("--arms_checkpoint", help="Run this arms policy meanwhile (`same`: this checkpoint's).")
    p.add_argument("--arms_export", help="... or this export's arms.")
    p.add_argument("--num_envs", type=int, default=D["num_envs"])
    p.add_argument("--segments_per_env", type=int, default=D["segments_per_env"])
    p.add_argument("--seed", type=int, default=D["seed"])
    p.add_argument("--plant", default=D["plant"], choices=["isaac", "robocasa"])
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--force", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    if a.run:
        from evals.common import dispatch
        code = 0
        for _, ckpt in dispatch.select(runs.resolve(a.run), a.steps):
            code |= score_one(a, str(ckpt))
        return code
    if a.checkpoint:
        return score_one(a, os.path.abspath(a.checkpoint))
    if a.export and a.out:
        return score_one(a, None)
    p.error("--checkpoint, --run, or --export with --out")


if __name__ == "__main__":
    sys.exit(main())
