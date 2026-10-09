"""Training curves of every run in conditions.yaml, from its TensorBoard file: curriculum progress and the
training-time reach and walking metrics against environment transitions.

    python paper/reach/tools/training_curves.py          # all listed runs with an events file

Writes logs/reach/results/curves/<label>_train.csv, one row per logged update (every 24 policy steps):
    step, transitions        policy steps, and policy steps x the run's num_envs
    depth_level              mean curriculum level k_d over the environments (0-10; z_max = 0.1 k_d m)
    reach_rate               goals reached / goals ended, over the episodes that ended in that update (the task's
                             own 5 cm / 0.35 rad / 1 s test); crouch_reach_rate the same over goals below the table
    goal_position_error_cm, goal_orientation_error_rad, best_error_cm   training-goal errors (mean, closest approach)
    pelvis_drop_cm, target_drop_cm   mean pelvis drop and goal lowering during arm goals
    walk_error_m_s           velocity tracking error during navigation; falls  share of episodes ending in a fall
These are the curriculum's own goals, not the fixed grid: their difficulty changes with k_d (see eval_curve.py for
the fixed-grid curve of saved checkpoints).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

TAGS = {
    "depth_level": "Curriculum/arm_target_levels/depth_level",
    "reached": "Metrics/arm_targets/goals_reached",
    "missed": "Metrics/arm_targets/goals_missed",
    "crouch_reached": "Metrics/arm_targets/crouch_goals_reached",
    "crouch_missed": "Metrics/arm_targets/crouch_goals_missed",
    "goal_position_error": "Metrics/arm_targets/goal_position_error",
    "goal_orientation_error": "Metrics/arm_targets/goal_orientation_error",
    "best_error": "Metrics/arm_targets/goal_best_error",
    "pelvis_drop": "Metrics/arm_targets/pelvis_drop",
    "target_drop": "Metrics/arm_targets/target_drop",
    "walk_error": "Metrics/base_velocity/nav_error_vel_xy",
    "falls": "Episode_Termination/fell",
}


def curves(run: Path) -> list[dict]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(run.glob("events.out.tfevents.*")) + sorted(run.glob("*/events.out.tfevents.*"))
    if not files:
        return []
    series: dict[str, dict[int, float]] = {k: {} for k in TAGS}
    for f in files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags()["scalars"])
        for key, tag in TAGS.items():
            if tag in have:
                for ev in ea.Scalars(tag):
                    series[key][ev.step] = ev.value
    cfg = C._yaml_load_any((run / "params" / "env.yaml").read_text()) if (run / "params" / "env.yaml").is_file() else {}
    num_envs = int(cfg.get("scene", {}).get("num_envs", 4096))
    steps = sorted(set().union(*[set(v) for v in series.values()]))
    nan = float("nan")
    rows = []
    for st in steps:
        g = {k: series[k].get(st, nan) for k in TAGS}
        ended = g["reached"] + g["missed"]
        cended = g["crouch_reached"] + g["crouch_missed"]
        rows.append({"step": st, "transitions": st * num_envs, "depth_level": g["depth_level"],
                     "reach_rate": g["reached"] / ended if ended > 0 else nan,
                     "crouch_reach_rate": g["crouch_reached"] / cended if cended > 0 else nan,
                     "goal_position_error_cm": 100 * g["goal_position_error"],
                     "goal_orientation_error_rad": g["goal_orientation_error"], "best_error_cm": 100 * g["best_error"],
                     "pelvis_drop_cm": 100 * g["pelvis_drop"], "target_drop_cm": 100 * g["target_drop"],
                     "walk_error_m_s": g["walk_error"], "falls": g["falls"]})
    return rows


def main():
    for cond in C.load_conditions():
        run = cond.get("run")
        if not run or not Path(run).is_dir():
            continue
        rows = curves(Path(run))
        if rows:
            out = C.RESULTS_DIR / "curves" / f"{cond['label']}_train.csv"
            C.write_csv(out, rows)
            last = rows[-1]
            print(f"[training_curves] {cond['label']}: {len(rows)} updates, final depth level {last['depth_level']:.2f},"
                  f" reach rate {last['reach_rate']:.3f} -> {out}")


if __name__ == "__main__":
    main()
