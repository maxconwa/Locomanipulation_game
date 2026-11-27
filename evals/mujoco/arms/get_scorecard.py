"""The arms scorecard in MuJoCo: arm goals held in a row on the exported arms policy, scored as the Isaac arms
scorecard (evals/common/arms_scorecard.py), against the IK alone on the same goals.

    python -m evals.mujoco.arms.get_scorecard --checkpoint <run>/checkpoints/agent_<N>.pt [--legs_checkpoint <ckpt>|same]
    python -m evals.mujoco.arms.get_scorecard --run <run> [--steps all|last|<N>,...]   (the IK-only reference first)
        [--num_envs 32] [--goals 8] [--plant isaac|robocasa] [--ik_only] [--workers 12] [--force]

Without a legs policy the pelvis is pinned (a weld; the legs at their default pose); with one, the pelvis is free,
the legs policy balances, the goals ride on the pelvis (LocoManip-Marl-Direct-v0), and the goals are scored twice:
standing, then while the legs walk (a velocity command from the export's ranges x 0.6, new each goal). Goals: the
export's own goal table (evals/mujoco/runner.py). --ik_only: no arms actor (residual 0, the IK alone): the reference,
written as scorecard_ik_only. CPU only.

Writes logs/evals/<run>/mujoco/arms/scorecard_<N>[_robocasa][_with_<legs run>_<M>].{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import arms_scorecard as A  # noqa: E402
from evals.common import conditions as C  # noqa: E402
from evals.common import outputs, runs, scorecard  # noqa: E402

D = C.DEFAULTS[("mujoco", "arms")]


def job(spec):
    """One robot: `goals` goals standing (and, with legs, `goals` more walking). Returns its recordings."""
    import torch

    from evals.mujoco import runner as R
    torch.set_num_threads(1)
    e, settings, arms_export, legs_export, plant, goals, seed, walk_cmds = spec
    sim = R.make_sim(settings, legs_export, arms_export, plant, pinned=legs_export is None)
    rng = np.random.default_rng(seed + e)
    sim.reset(rng)
    R.reset_actors(sim)
    sim.arm_goal_keeps_walking = legs_export is not None
    tables, goal_rng = R.goal_table(settings), np.random.default_rng(seed + 10_000 + e)
    steps = int(round(A.GOAL_S / sim.policy_dt))
    obs = sim.observations()
    phases = [None] + ([walk_cmds] if legs_export is not None else [])
    out = []
    for cmds in phases:
        err, rot = np.zeros((goals, steps, 2), np.float32), np.zeros((goals, steps, 2), np.float32)
        pos, qd = np.zeros((goals, steps, 2, 3), np.float32), np.zeros((goals, steps, 14), np.float32)
        fell = np.zeros(goals, bool)
        for g in range(goals):
            sim.arm_mode, sim.believed = True, R.draw_goal(tables, goal_rng)
            sim.vel_cmd = np.zeros(3) if cmds is None else cmds[g]
            for k in range(steps):
                if fell[g]:
                    break
                obs = sim.step(obs)
                err[g, k], rot[g, k] = R.wrist_errors(sim)
                pos[g, k] = [sim.d.xpos[w] for w in sim.wrists]
                qd[g, k] = sim.d.qvel[sim.vadr[13:27]]
                if not sim.pinned and R.fell(sim):
                    fell[g] = True
                    sim.reset(rng)
                    R.reset_actors(sim)
                    obs = sim.observations()
        out.append((err, rot, pos, qd, fell))
    return e, out, sim.policy_dt


def out_base(checkpoint, a) -> str:
    run_dir = runs.run_dir_of(checkpoint)
    plant = "" if a.plant == "isaac" else f"_{a.plant}"
    if a.ik_only:
        return str(outputs.base(runs.name(run_dir), "mujoco", "arms", f"scorecard{plant}_ik_only", None))
    suffix = plant
    if a.legs_checkpoint:
        other = run_dir if a.legs_checkpoint == "same" else runs.run_dir_of(a.legs_checkpoint)
        suffix += f"_with_{runs.short(other)}_{runs.step_of(checkpoint if a.legs_checkpoint == 'same' else a.legs_checkpoint)}"
    return str(outputs.base(runs.name(run_dir), "mujoco", "arms", "scorecard", runs.step_of(checkpoint), suffix))


def score_one(a, checkpoint: str):
    from evals.mujoco.policy import combined_settings, export_for
    run_dir = runs.run_dir_of(checkpoint)
    base = a.out or out_base(checkpoint, a)
    if outputs.cached(base) and not a.force:
        print(f"[evals] cached: {base}.json")
        return 0
    arms_export = export_for(checkpoint, outputs.eval_dir(runs.name(run_dir), "mujoco", "exports"))
    legs_export = None
    if a.legs_checkpoint:
        lc = checkpoint if a.legs_checkpoint == "same" else os.path.abspath(a.legs_checkpoint)
        legs_export = export_for(lc, outputs.eval_dir(runs.name(runs.run_dir_of(lc)), "mujoco", "exports"))
    settings = combined_settings(legs_export, arms_export)
    r = settings.get("command_ranges") or {}
    rng = np.random.default_rng(a.seed)
    walk = np.stack([rng.uniform(*r.get(k, [-0.5, 0.5]), (a.num_envs, a.goals))
                     for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")], -1) * 0.6
    specs = [(e, settings, None if a.ik_only else str(arms_export), str(legs_export) if legs_export else None, a.plant,
              a.goals, a.seed, walk[e]) for e in range(a.num_envs)]
    print(f"[evals] mujoco arms scorecard: {a.num_envs} robots x {a.goals} goals, plant {a.plant},"
          f" {'IK alone' if a.ik_only else 'arms policy'}, legs {'policy ' + str(legs_export) if legs_export else 'pinned'}",
          flush=True)
    with Pool(a.workers) as pool:
        results = sorted(pool.map(job, specs), key=lambda x: x[0])
    dt = results[0][2]

    def phase(i):
        err, rot, pos, qd, fell = (np.stack([res[1][i][j] for res in results]) for j in range(5))
        return A.compute(err, rot, pos, qd, fell, dt)
    res = phase(0)
    if legs_export is not None:
        res["walk"] = phase(1)
    ref_base = out_base(checkpoint, argparse.Namespace(**{**vars(a), "ik_only": True}))
    ref = json.load(open(ref_base + ".json")) if (not a.ik_only and outputs.cached(ref_base)) else None
    res.update(backend="mujoco", body="arms", run=runs.name(run_dir), step=runs.step_of(checkpoint),
               checkpoint=checkpoint, export=str(arms_export), plant=a.plant, ik_only=a.ik_only, stochastic=False,
               offtable=False, other_policy=(f"legs policy {legs_export}" if legs_export else None), overrides=[],
               reference=ref_base + ".json" if ref else None,
               **C.stamp("mujoco", "arms", num_envs=a.num_envs, goals=a.goals, seed=a.seed, plant=a.plant,
                         actions="zero" if a.ik_only else "mean", legs="policy" if legs_export else "pinned"))
    scorecard.write(base, res, A.rows(res, ref), A.header(res, res["reference"]), A.footer(res))
    print(open(base + ".md").read())
    print(f"[evals] wrote {base}.json / .md")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--run")
    p.add_argument("--steps", default="all")
    p.add_argument("--legs_checkpoint", help="Run this legs policy, the pelvis free (`same`: this checkpoint's).")
    p.add_argument("--num_envs", type=int, default=D["num_envs"])
    p.add_argument("--goals", type=int, default=D["goals"])
    p.add_argument("--seed", type=int, default=D["seed"])
    p.add_argument("--plant", default=D["plant"], choices=["isaac", "robocasa"])
    p.add_argument("--ik_only", action="store_true")
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--force", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    if a.run:
        from evals.common import dispatch
        ckpts = dispatch.select(runs.resolve(a.run), a.steps)
        code = 0
        if not a.ik_only and ckpts:  # the run's IK-only reference first
            code |= score_one(argparse.Namespace(**{**vars(a), "ik_only": True, "legs_checkpoint": None, "out": None}),
                              str(ckpts[-1][1]))
        for _, ckpt in ckpts:
            code |= score_one(a, str(ckpt))
        return code
    if a.checkpoint:
        return score_one(a, os.path.abspath(a.checkpoint))
    p.error("--checkpoint or --run")


if __name__ == "__main__":
    sys.exit(main())
