"""Tables for docs/experiments/sportmode_ablations.md from the queue's runs (scripts/sportmode_ablations.sh): each
run's scorecards (final and 50.4k checkpoints), the base / base_s2 gap as the noise floor, every ablation's change
against base, and the training logs' arm-goal reach rate, wrist error and fall rate.

    python scripts/sportmode_summary.py [--status logs/sportmode_ablations/status.tsv] [--out <file.md>] [--json <file>]

Training-log values are means over the run's last 2400 env steps (or the 2400 up to 50.4k): reach rate =
goals reached / (reached + missed) per ended episode, wrist error = mean wrist position error during arm goals
(Metrics/arm_targets/goal_position_error), closest approach (goal_best_error), fall share = the share of ended
episodes that fell (Episode_Termination/fell).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "skrl"))
from scorecard_metrics import TARGETS, verdict  # noqa: E402

TAG_ORDER = ["base", "base_s2", "gate", "cmd", "cmd_tw", "range", "yaw", "gait070", "gains", "hz100", "combo", "combo_tw",
             "combo_all"]
CHANGES = {"base": "none", "base_s2": "none (seed 43)", "gate": "1", "cmd": "2", "cmd_tw": "2, kernel pace in s",
           "range": "3", "yaw": "4", "gait070": "5", "gains": "6", "hz100": "7", "combo": "1–5",
           "combo_tw": "1–5, kernel pace in s", "combo_all": "1–7"}
SCALARS = {
    "reached": "Metrics/arm_targets/goals_reached",
    "missed": "Metrics/arm_targets/goals_missed",
    "wrist_err": "Metrics/arm_targets/goal_position_error",
    "best_err": "Metrics/arm_targets/goal_best_error",
    "fell": "Episode_Termination/fell",
    "nav_cos": "Metrics/base_velocity/nav_cos_vel_xy",
    "goals_on": "Curriculum/arm_goals_enabled",
}
WINDOW = 2400


def read_status(path: str) -> dict:
    runs = {}
    for line in open(path):
        cols = line.rstrip("\n").split("\t")
        if len(cols) >= 3:
            runs[cols[0]] = {"state": cols[1], "run": cols[2], "hours": cols[3] if len(cols) > 3 else "",
                             "kernels": cols[5] if len(cols) > 5 else ""}
    return runs


def training_metrics(run: str, end: int) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = glob.glob(os.path.join(run, "events.out.tfevents.*"))
    if not files:
        return {}
    ea = EventAccumulator(files[0], size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out = {}
    for key, tag in SCALARS.items():
        if tag not in tags:
            continue
        ev = ea.Scalars(tag)
        s, v = np.array([e.step for e in ev]), np.array([e.value for e in ev])
        m = (s > end - WINDOW) & (s <= end)
        out[key] = float(np.nanmean(v[m])) if m.any() else float("nan")
    r, mi = out.get("reached", np.nan), out.get("missed", np.nan)
    out["reach_rate"] = r / (r + mi) if r + mi > 0 else float("nan")
    return out


def load(status_path: str) -> dict:
    data = {}
    for tag, st in read_status(status_path).items():
        row = {"tag": tag, **st, "scorecard": {}, "training": {}}
        if st["state"] == "done":
            for step in (100008, 50400):
                path = os.path.join(st["run"], "checkpoints", f"scorecard_{step}.json")
                if os.path.isfile(path):
                    row["scorecard"][step] = json.load(open(path))
                row["training"][step] = training_metrics(st["run"], step)
        data[tag] = row
    return data


def f(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def lag(sc: dict, key: str) -> float:
    return sc.get("tracking_vs_lagged", {}).get(key, float("nan")) if sc else float("nan")


def tables(data: dict) -> str:
    tags = [t for t in TAG_ORDER if t in data] + [t for t in data if t not in TAG_ORDER]
    out = []
    for step, title in ((100008, "final checkpoint (100k steps)"), (50400, "50.4k checkpoint")):
        out += [f"### Scorecard, {title}", "",
                "| Run | Changes | Turn sense | Speed ratio | Turn ratio | cos | Ang err | Lin err | Met | Stride (IQR) s |"
                " Roll / pitch sd ° | Falls | Commanded, not stepping |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                "| Target | | ≥ 0.99 | ≥ 0.90 | ≥ 0.90 | ≥ 0.95 | ≤ 0.05 | ≤ 0.15 | 6 | | | | |",
                "| Sport mode (bags) | | 0.994 | 0.93 | 0.91 | 0.96* | 0.050 | 0.147 | | 0.700 (0.01) | 0.83 / 0.75 | | |",
                "| Explicit500 (robot) | | 0.896 | 0.96 | 0.61 | 0.54 | 0.137 | 0.151 | | 0.790 (0.08) | 2.03 / 2.22 | | 28% |"]
        for tag in tags:
            row = data[tag]
            sc = row["scorecard"].get(step)
            if row["state"] != "done" or not sc:
                out.append(f"| {tag} | {CHANGES.get(tag, '')} | {row['state']}: no scorecard |" + " |" * 10)
                continue
            vals = []
            for key, _, _, target, sense in TARGETS:
                v = lag(sc, key)
                vals.append(f"**{f(v)}**" if verdict(v, target, sense) else f(v))
            g, t = sc["gait"], sc["tilt"]
            out.append(f"| {tag} | {CHANGES.get(tag, '')} | " + " | ".join(vals) + f" | {sc['targets_met']} |"
                       f" {f(g['stride_T_median'])} ({f(g['stride_T_iqr'], 2)}) | {f(t['roll_sd_deg'], 2)} /"
                       f" {f(t['pitch_sd_deg'], 2)} | {sc['falls']}/{sc['num_envs']} |"
                       f" {100 * sc['commanded_not_stepping_share']:.0f}% (small {100 * sc['small_not_stepping_share']:.0f}%) |")
        out += ["", "Bold: meets the target. *Sport mode's cos with its forward creep removed (0.65 raw); ours are raw.", ""]

    # noise floor and changes against base, final checkpoint
    base, s2 = data.get("base", {}).get("scorecard", {}).get(100008), data.get("base_s2", {}).get("scorecard", {}).get(100008)
    if base and s2:
        out += ["### Change against base, final checkpoint", "",
                "The noise floor is |base − base_s2|. A change counts as beyond noise (✓) when it moves a metric the"
                " right way by more than that.", "",
                "| Run | " + " | ".join(lbl.split(",")[0] for _, lbl, _, _, _ in TARGETS) + " | Beyond noise |",
                "| --- |" + " --- |" * (len(TARGETS) + 1)]
        noise = {k: abs(lag(base, k) - lag(s2, k)) for k, *_ in TARGETS}
        out.append("| noise floor | " + " | ".join(f(noise[k]) for k, *_ in TARGETS) + " | |")
        for tag in tags:
            sc = data[tag]["scorecard"].get(100008)
            if tag in ("base", "base_s2") or not sc:
                continue
            cells, better = [], 0
            for key, _, _, _, sense in TARGETS:
                d = lag(sc, key) - lag(base, key)
                good = (d > noise[key]) if sense == "ge" else (d < -noise[key])
                better += good
                cells.append(f"{d:+.3f}{' ✓' if good else ''}" if np.isfinite(d) else "—")
            out.append(f"| {tag} | " + " | ".join(cells) + f" | {better} of {len(TARGETS)} |")
        out.append("")

    out += ["### Arms and falls in training", "",
            "Means over the last 2400 env steps before each checkpoint, from the training logs.", "",
            "| Run | Hours | Kernel check | Arm goals on | Reach rate 100k | Wrist error 100k (m) | Closest approach 100k (m) |"
            " Fall share 100k | Reach rate 50k | Fall share 50k | Training cos 100k |",
            "| --- |" + " --- |" * 10]
    for tag in tags:
        row = data[tag]
        tr, tr50 = row["training"].get(100008, {}), row["training"].get(50400, {})
        out.append(f"| {tag} | {row['hours']} | {row['kernels']} | {f(tr.get('goals_on'), 0)} | {f(tr.get('reach_rate'))} |"
                   f" {f(tr.get('wrist_err'))} | {f(tr.get('best_err'))} | {f(tr.get('fell'))} | {f(tr50.get('reach_rate'))} |"
                   f" {f(tr50.get('fell'))} | {f(tr.get('nav_cos'))} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", default="logs/sportmode_ablations/status.tsv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    data = load(args.status)
    text = tables(data)
    if args.out:
        open(args.out, "w").write(text)
    if args.json:
        json.dump(data, open(args.json, "w"), indent=1, default=float)
    print(text)
