"""Aggregate every evaluated condition into the tables, numbers and macros the figures, the page and the paper read.

    python paper/reach/tools/analyze.py

Reads results/<label>/<sim>/trials.csv for every label not starting with "_" (smoke tests). A condition is one
(label, sim) pair, e.g. (lambda1, isaac), (lambda1, isaac_legs_blind), (golem_ik, kinematic). Display names,
colours and order come from conditions.yaml when it names the label, otherwise from the label and its lambda.

Writes
    results/summary/by_depth.csv        per condition and depth: N, strict success with a Wilson 95% interval,
                                        per-wrist success, errors, pelvis height and drop, falls, balance metrics
    results/summary/by_height.csv       the same per 0.1 m bin of the lower wrist target's height above the ground
    results/summary/workspace.csv       per-wrist success per (horizontal distance from the shoulder, height) cell
    results/summary/paired.csv          paired differences on identical (goal, seed) between conditions that share a
                                        policy (full vs legs_blind vs arms_ik) or a goal set (any vs golem_ik)
    results/summary/summary.json        everything above plus per-condition totals and run metadata
    tex/results_macros.tex              \\newcommand numbers the paper text uses (regenerated, never edited by hand)
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402
from reachlib.metrics import TOL_PAIRS  # noqa: E402

SUMMARY = C.RESULTS_DIR / "summary"
HEIGHT_BIN = 0.1
WS_R = np.round(np.arange(0.0, 1.21, 0.1), 2)     # horizontal distance of a wrist target from its shoulder, m
WS_Z = np.round(np.arange(0.0, 2.01, 0.1), 2)     # wrist target height above the ground, m
FLOAT_FIELDS = None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_trials() -> dict[tuple[str, str], list[dict]]:
    out = {}
    for path in sorted(C.RESULTS_DIR.glob("*/*/trials.csv")):
        label, sim = path.parent.parent.name, path.parent.name
        if label.startswith("_") or label in ("summary", "curves") or "@" in label:   # @: learning-curve checkpoints
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k, v in list(r.items()):
                if k in ("label", "sim", "kind", "goal_id", "checkpoint", "variant", "seed_name", "error"):
                    continue
                try:
                    r[k] = float(v) if v not in ("", None) else math.nan
                except ValueError:
                    pass
        if rows and rows[0].get("kind") == "golem_ik" and "hold_err_pos" not in rows[0]:
            # a kinematic solve holds its solution: the hold error is the final error, worst wrist
            for r in rows:
                r["hold_err_pos"] = max(r["err_final_l"], r["err_final_r"])
                r["hold_err_rot"] = max(r["rot_final_l"], r["rot_final_r"])
                for t, (tp, tr) in TOL_PAIRS.items():
                    r[t] = float(r["hold_err_pos"] < tp and r["hold_err_rot"] < tr)
        if rows:
            out[(label, sim)] = rows
    return out


def condition_info(key, rows, registry: dict) -> dict:
    label, sim = key
    reg = registry.get(label, {})
    lam = rows[0].get("lambda_share", math.nan)
    lam = None if lam is None or (isinstance(lam, float) and math.isnan(lam)) else float(lam)
    variant = sim.split("_", 1)[1] if "_" in sim else ""
    base_sim = sim.split("_", 1)[0]
    kind = rows[0].get("kind", "")
    if kind == "golem_ik":
        name = "GOLEM arm IK, standing"
    elif reg.get("name"):
        name = reg["name"]
    elif lam is not None:
        name = f"λ = {lam:g}"
    else:
        name = label
    for tag, text in (("legs_blind", ", legs blind"), ("arms_ik", ", arms IK only"), ("train", ", training physics"),
                      ("trueodom", ", true odometry")):
        if tag in variant:
            name += text
    if base_sim == "mujoco":
        name += " (MuJoCo, training joints)" if "isaacphys" in variant else " (MuJoCo, RoboCasa joints)"
    return {"key": f"{label}/{sim}", "label": label, "sim": sim, "base_sim": base_sim, "variant": variant, "kind": kind,
            "lambda": lam, "name": name, "n": len(rows),
            "checkpoint": rows[0].get("checkpoint", ""), "order": reg.get("order", 0)}


def _f(rows, k):
    return np.array([r.get(k, math.nan) for r in rows], dtype=float)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    s = _f(rows, "success")
    k = int(np.nansum(s))
    lo, hi = wilson(k, n)
    fell = _f(rows, "fell")
    ok = fell < 0.5
    out = {"n": n, "success": k / n if n else math.nan, "success_lo": lo, "success_hi": hi,
           **{f"{t}": float(np.nanmean(_f(rows, t))) for t in ("success_2cm", "success_3cm", "success_8cm",
                                                               "success_10cm") if t in rows[0]},
           "success_l": float(np.nanmean(_f(rows, "success_l"))), "success_r": float(np.nanmean(_f(rows, "success_r"))),
           "fall_rate": float(np.nanmean(fell)) if n else math.nan}
    for t in TOL_PAIRS:
        if t in rows[0]:
            out[f"{t}_lo"], out[f"{t}_hi"] = wilson(int(np.nansum(_f(rows, t))), n)
    if "hold_err_pos" in rows[0]:
        # which half of the strict test fails: position alone (5 cm) and orientation alone (0.35 rad), each on its own
        # best 1 s window
        out["pos_ok"] = float(np.mean(np.nan_to_num(_f(rows, "hold_err_pos"), nan=np.inf) < C.POS_TOL))
        out["rot_ok"] = float(np.mean(np.nan_to_num(_f(rows, "hold_err_rot"), nan=np.inf) < C.ROT_TOL))
    for key in ("err_last1s", "rot_last1s", "closest", "hold_err_pos", "hold_err_rot", "pelvis_h_last1s",
                "pelvis_drop_last1s", "pelvis_h_start",
                "tilt_max_deg", "angvel_rms", "com_margin_min", "com_margin_last1s", "dcm_margin_min",
                "dcm_margin_last1s", "foot_slip_max", "steps", "ee_jitter_l", "ee_jitter_r", "ee_speed_rms",
                "torque_knee_max", "torque_ankle_max", "pitch_last1s_deg", "t_success", "cmd_drift_end"):
        v = _f(rows, key)
        v = v[ok & np.isfinite(v)] if key not in ("t_success",) else v[np.isfinite(v)]
        out[f"{key}_med"] = float(np.median(v)) if len(v) else math.nan
        if key in ("steps", "foot_slip_max", "tilt_max_deg"):
            out[f"{key}_mean"] = float(np.mean(v)) if len(v) else math.nan
        out[f"{key}_q25"] = float(np.percentile(v, 25)) if len(v) else math.nan
        out[f"{key}_q75"] = float(np.percentile(v, 75)) if len(v) else math.nan
    # wrist error of the successful trials only: the accuracy at which a goal counts as reached
    v = _f(rows, "err_last1s")[(s > 0.5) & ok]
    out["err_last1s_success_med"] = float(np.median(v)) if len(v) else math.nan
    return out


def workspace(rows: list[dict]) -> list[dict]:
    """Per-wrist success by (horizontal distance from that arm's shoulder, height above ground)."""
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        for arm, pre in (("left", "l"), ("right", "r")):
            x, y, z = r[f"target_{pre}_x"], r[f"target_{pre}_y"], r[f"target_{pre}_z"]
            sx, sy, _ = C.SHOULDER_B[arm]
            rad = math.hypot(x - sx, y - sy)
            h = z + C.STANDING_PELVIS_HEIGHT
            i, j = int(math.floor(rad / 0.1)), int(math.floor(h / 0.1))
            c = cells[(i, j)]
            c[0] += 1
            c[1] += int(r[f"success_{pre}"] > 0.5)
    return [{"r_lo": round(0.1 * i, 2), "z_lo": round(0.1 * j, 2), "n": n, "success": k / n}
            for (i, j), (n, k) in sorted(cells.items())]


def operational_floor(by_height: list[dict], thresh: float = 0.8, min_n: int = 20) -> float:
    """The lowest height bin from which every bin above has strict success >= thresh (m above ground)."""
    rows = sorted([b for b in by_height if b["n"] >= min_n], key=lambda b: b["bin_lo"], reverse=True)
    floor = math.nan
    for b in rows:
        if b["success"] >= thresh:
            floor = b["bin_lo"]
        else:
            break
    return floor


def lowest_bin(by_height: list[dict], key: str = "success_10cm", thresh: float = 0.5, min_n: int = 10) -> float:
    """The lowest height bin whose success on `key` is >= thresh (m above ground): how low the wrists get at all,
    where operational_floor asks how low the whole range above stays reliable."""
    ok = [b["bin_lo"] for b in by_height if b["n"] >= min_n and b.get(key, math.nan) >= thresh]
    return min(ok) if ok else math.nan


def paired(trials: dict, infos: dict) -> list[dict]:
    """Paired success differences on identical (goal, seed) cells."""
    out = []
    keys = list(trials)
    index = {k: {(r["goal_id"], int(r["seed"]) if not math.isnan(r["seed"]) else 0): r for r in v} for k, v in trials.items()}
    pairs = []
    for a in keys:
        for b in keys:
            if a >= b:
                continue
            ia, ib = infos[a], infos[b]
            same_policy = ia["label"] == ib["label"] and ia["base_sim"] == ib["base_sim"]
            vs_ik = "golem_ik" in (ia["kind"], ib["kind"])
            sim2sim = ia["label"] == ib["label"] and ia["variant"] == ib["variant"] and ia["base_sim"] != ib["base_sim"]
            if same_policy or vs_ik or sim2sim:
                pairs.append((a, b, "sim2sim" if sim2sim else "variant" if same_policy else "vs_golem_ik"))
    for a, b, why in pairs:
        ka, kb = index[a], index[b]
        if "golem_ik" in (infos[a]["kind"], infos[b]["kind"]):
            # the IK has one deterministic row per goal: pair it with every seed of the other
            ik, other = (ka, kb) if infos[a]["kind"] == "golem_ik" else (kb, ka)
            ik_by_goal = {g: r for (g, _), r in ik.items()}
            common = [(ik_by_goal[g], r) for (g, s), r in other.items() if g in ik_by_goal]
            if infos[a]["kind"] != "golem_ik":
                common = [(o, i) for i, o in common]
        else:
            common = [(ka[c], kb[c]) for c in ka.keys() & kb.keys()]
        for depth in sorted({ra["depth"] for ra, _ in common}) + [None]:
            sel = [(ra, rb) for ra, rb in common if depth is None or ra["depth"] == depth]
            if not sel:
                continue
            sa = np.array([ra["success"] for ra, _ in sel]) > 0.5
            sb = np.array([rb["success"] for _, rb in sel]) > 0.5
            out.append({"a": a_key(a), "b": a_key(b), "relation": why, "depth": "all" if depth is None else depth,
                        "n": len(sel), "success_a": sa.mean(), "success_b": sb.mean(), "diff": sa.mean() - sb.mean(),
                        "only_a": int((sa & ~sb).sum()), "only_b": int((sb & ~sa).sum()), "both": int((sa & sb).sum())})
    return out


def a_key(k):
    return f"{k[0]}/{k[1]}"


def main():
    trials = load_trials()
    registry = {c["label"]: {**c, "order": i} for i, c in enumerate(C.load_conditions())}
    infos = {k: condition_info(k, v, registry) for k, v in trials.items()}
    SUMMARY.mkdir(parents=True, exist_ok=True)
    by_depth, by_height, ws_rows, totals = [], [], [], {}
    for k, rows in trials.items():
        info = infos[k]
        totals[a_key(k)] = {**info, **summarize(rows)}
        for d in sorted({r["depth"] for r in rows}):
            by_depth.append({"condition": a_key(k), "name": info["name"], "depth": d,
                             **summarize([r for r in rows if r["depth"] == d])})
        hb = defaultdict(list)
        for r in rows:
            hb[math.floor(r["target_min_height"] / HEIGHT_BIN)].append(r)
        rows_h = []
        for b in sorted(hb):
            rec = {"condition": a_key(k), "name": info["name"], "bin_lo": round(b * HEIGHT_BIN, 2),
                   "bin_mid": round((b + 0.5) * HEIGHT_BIN, 3), **summarize(hb[b])}
            rows_h.append(rec)
        by_height += rows_h
        # operational floor on the unextended goals: the standing-table poses lowered, all reachable at d = 0
        hb0 = defaultdict(list)
        for r in rows:
            if r["ext"] == 0.0:
                hb0[math.floor(r["target_min_height"] / HEIGHT_BIN)].append(r)
        rows_h0 = [{"bin_lo": round(b * HEIGHT_BIN, 2), **summarize(hb0[b])} for b in sorted(hb0)]
        totals[a_key(k)]["operational_floor_m"] = operational_floor(rows_h0, min_n=10)
        totals[a_key(k)]["floor10_m"] = lowest_bin(rows_h0)
        # goals whose lower wrist target lies below / above the standing table's floor
        floor = C.STANDING_PELVIS_HEIGHT + min(json.loads((C.GOALS_DIR / "eval_goals_v1.json").read_text())["standing_min_z"])
        below = [r for r in rows if r["target_min_height"] < floor]
        above = [r for r in rows if r["target_min_height"] >= floor]
        totals[a_key(k)]["below_floor"] = summarize(below) if below else None
        totals[a_key(k)]["above_floor"] = summarize(above) if above else None
        for c in workspace(rows):
            ws_rows.append({"condition": a_key(k), **c})
        # ext 0 only: the standing-table goals lowered, the controlled comparison the brief asks for
        ext0_rows = [r for r in rows if r["ext"] == 0.0]
        if ext0_rows:
            e0 = summarize(ext0_rows)
            totals[a_key(k)]["ext0"] = {q: e0.get(q) for q in ("n", "success", "success_10cm", "fall_rate",
                                                                "hold_err_pos_med", "pos_ok", "rot_ok")}
        for d in sorted({r["depth"] for r in rows}):
            sel = [r for r in rows if r["depth"] == d and r["ext"] == 0.0]
            if sel:
                totals[a_key(k)].setdefault("ext0_by_depth", {})[f"{d:.1f}"] = summarize(sel)["success"]
    # position-only success against the tolerance: the share of trials whose best 1 s window keeps both wrists
    # within tau (orientation not tested), for all goals and for those below / above the standing table's floor
    floor_all = C.STANDING_PELVIS_HEIGHT + min(json.loads((C.GOALS_DIR / "eval_goals_v1.json").read_text())["standing_min_z"])
    tol_rows = []
    taus = np.round(np.arange(0.01, 0.2001, 0.005), 3)
    for k, rows in trials.items():
        if "hold_err_pos" not in rows[0]:
            continue
        for group, sel in (("all", rows), ("below", [r for r in rows if r["target_min_height"] < floor_all]),
                           ("above", [r for r in rows if r["target_min_height"] >= floor_all])):
            if not sel:
                continue
            he = _f(sel, "hold_err_pos")
            for tau in taus:
                tol_rows.append({"condition": a_key(k), "group": group, "tau": float(tau), "n": len(sel),
                                 "success_pos": float(np.mean(he < tau))})
    C.write_csv(SUMMARY / "tolerance.csv", tol_rows)
    # walking: velocity tracking per run (eval_walk.py), the legs' task
    walk_rows = []
    for path in sorted(C.RESULTS_DIR.glob("*/isaac_walk/walk.csv")):
        label = path.parent.parent.name
        if label.startswith("_") or "@" in label:
            continue
        w = list(csv.DictReader(open(path)))
        moving = [r for r in w if r["command"] != "stand"]
        turns = [r for r in w if float(r["wz_cmd"]) != 0.0]
        stand = [r for r in w if r["command"] == "stand"]
        walk_rows.append({"label": label, "lambda_share": float(w[0]["lambda_share"]) if w[0]["lambda_share"] not in ("", "nan") else math.nan,
                          "err_xy_moving": float(np.mean([float(r["err_xy"]) for r in moving])),
                          "err_yaw_turns": float(np.mean([float(r["err_yaw"]) for r in turns])) if turns else math.nan,
                          "stand_drift": float(stand[0]["err_xy"]) if stand else math.nan,
                          "falls": int(sum(int(r["falls"]) for r in w)), "n_env_commands": int(sum(int(r["n"]) + int(r["falls"]) for r in w))})
    C.write_csv(SUMMARY / "walk.csv", walk_rows)
    pairs = paired(trials, infos)
    C.write_csv(SUMMARY / "by_depth.csv", by_depth)
    C.write_csv(SUMMARY / "by_height.csv", by_height)
    C.write_csv(SUMMARY / "workspace.csv", ws_rows)
    C.write_csv(SUMMARY / "paired.csv", pairs)
    goals_meta = json.loads((C.GOALS_DIR / "eval_goals_v1.json").read_text())
    C.write_json(SUMMARY / "summary.json", {"conditions": totals, "goals": goals_meta,
                                            "by_depth": by_depth, "paired": pairs})

    # -- LaTeX numbers: \reach{<condition>}{<quantity>} in the text; an entry not computed yet prints a bold ??
    entries = {
        ("goals", "n"): goals_meta["goals"], ("goals", "bases"): goals_meta["bases"],
        ("goals", "depths"): len(goals_meta["depths"]), ("goals", "depthmax"): f"{max(goals_meta['depths']):.1f}",
        ("goals", "exts"): len(goals_meta["exts"]),
        ("goals", "standingfloor"): f"{100 * (C.STANDING_PELVIS_HEIGHT + min(goals_meta['standing_min_z'])):.0f}",
    }
    for key, t in totals.items():
        entries[(key, "n")] = t["n"]
        entries[(key, "pergoal")] = round(t["n"] / goals_meta["goals"])
        entries[(key, "success")] = f"{100 * t['success']:.0f}"
        if not math.isnan(t.get("success_10cm", math.nan)):
            entries[(key, "successten")] = f"{100 * t['success_10cm']:.0f}"
        e0 = t.get("ext0") or {}
        if e0.get("success_10cm") is not None and not math.isnan(e0["success_10cm"]):
            entries[(key, "successten_e0")] = f"{100 * e0['success_10cm']:.0f}"
        entries[(key, "falls")] = f"{100 * t['fall_rate']:.1f}"
        if not math.isnan(t["operational_floor_m"]):
            entries[(key, "floor")] = f"{100 * t['operational_floor_m']:.0f}"
        if not math.isnan(t["floor10_m"]):
            entries[(key, "floorten")] = f"{100 * t['floor10_m']:.0f}"
        for side in ("below", "above"):
            b = t.get(f"{side}_floor")
            if b:
                entries[(key, f"success_{side}")] = f"{100 * b['success']:.0f}"
                if not math.isnan(b.get("success_10cm", math.nan)):
                    entries[(key, f"successten_{side}")] = f"{100 * b['success_10cm']:.0f}"
                entries[(key, f"n_{side}")] = b["n"]
                if not math.isnan(b["pelvis_drop_last1s_med"]):
                    entries[(key, f"drop_{side}")] = f"{100 * b['pelvis_drop_last1s_med']:.0f}"
        for d, v in (t.get("ext0_by_depth") or {}).items():
            entries[(key, f"success_e0_d{d}")] = f"{100 * v:.0f}"
    for b in by_depth:
        d = f"{b['depth']:.1f}"
        entries[(b["condition"], f"success_d{d}")] = f"{100 * b['success']:.0f}"
        for name, field, scale, fmt in (("drop", "pelvis_drop_last1s_med", 100, ".0f"),
                                        ("height", "pelvis_h_last1s_med", 100, ".0f"),
                                        ("err", "err_last1s_success_med", 100, ".1f"),
                                        ("falls", "fall_rate", 100, ".0f"),
                                        ("com", "com_margin_last1s_med", 100, ".1f"),
                                        ("steps", "steps_med", 1, ".1f"),
                                        ("jitter", "ee_jitter_l_med", 1, ".1f")):
            if not math.isnan(b[field]):
                entries[(b["condition"], f"{name}_d{d}")] = format(scale * b[field], fmt)
    for w in walk_rows:
        entries[(f"{w['label']}/walk", "err_xy")] = f"{100 * w['err_xy_moving']:.0f}"
        entries[(f"{w['label']}/walk", "err_yaw")] = f"{w['err_yaw_turns']:.2f}"
        entries[(f"{w['label']}/walk", "falls")] = w["falls"]
    for p in pairs:
        if p["depth"] == "all":
            entries[(f"{p['a']}-vs-{p['b']}", "diff")] = f"{100 * p['diff']:.0f}"
    lines = ["% Generated by paper/reach/tools/analyze.py from results/*/*/trials.csv. Do not edit by hand.",
             "% Use \\reach{<condition>}{<quantity>}, e.g. \\reach{lambda1/isaac}{success_d0.3}; see the page's"
             " table of quantities.",
             "\\makeatletter",
             "\\providecommand{\\reach}[2]{\\@ifundefined{reach@#1@#2}{\\textbf{??}}{\\@nameuse{reach@#1@#2}}}",
             "\\makeatother"]
    lines += [f"\\expandafter\\def\\csname reach@{c}@{q}\\endcsname{{{v}}}" for (c, q), v in sorted(entries.items())]
    C.TEX_DIR.mkdir(parents=True, exist_ok=True)
    (C.TEX_DIR / "results_macros.tex").write_text("\n".join(lines) + "\n")
    for key, t in sorted(totals.items(), key=lambda kv: kv[1]["order"]):
        print(f"[analyze] {key:38s} {t['name']:34s} N={t['n']:5d} success {100 * t['success']:5.1f}%"
              f" falls {100 * t['fall_rate']:4.1f}% floor {t['operational_floor_m']} / 10 cm {t['floor10_m']}")
    print(f"[analyze] wrote {SUMMARY} and {C.TEX_DIR / 'results_macros.tex'}")


if __name__ == "__main__":
    main()
