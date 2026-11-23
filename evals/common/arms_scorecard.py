"""The arms scorecard, whatever ran it (Isaac or MuJoCo): arm goals held in a row, and the scoring of the recorded
wrists.

Per goal (GOAL_S, training's) and wrist, over the hold (the goal's last HOLD_S, after the reach):
  error      the position error at the goal's end (mean of its last END_S) and its mean over the hold; the rotation
             error at the end (deg in the table, rad in the json); the time until the wrist is first within REACH_POS;
             reached: both wrists held inside a tolerance for REACH_HOLD_S at any time in the goal (5 cm / 20 deg, the
             env's, and 1 mm / 5 deg)
  vibration  the fast shake: the wrist's position less its moving average over SMOOTH_S (motion above ~1 / SMOOTH_S
             Hz), RMS over the hold, mm; and the arm joints' speed, RMS over the hold, rad/s
  wobble     the slow sway: the moving-averaged position with the hold's straight-line drift removed, RMS (mm) and its
             largest peak-to-peak along one axis (mm)
  variance   within a hold, the position error's sd over time (mm); across holds, the sd and p90 of the end error
Goals during which the env reset (a fall) are left out and counted. Reference: the IK alone (zero arms actions) on
the same goals; targets: the success criteria, end error <= 4 mm and <= 5 deg.
"""

from __future__ import annotations

import math

import numpy as np

from .scorecard import Row, num

GOAL_S = 4.0      # s per arm goal, training's
HOLD_S = 2.0      # s: the goal's last part, scored as the hold
END_S = 0.25      # s: the end error's averaging window
SMOOTH_S = 0.1    # s: the moving average that splits vibration (faster) from wobble (slower)
REACH_POS = 0.05  # m: the time-to-reach threshold, the env's reach tolerance
REACH_TOLS = {"5cm_20deg": (0.05, 0.35), "1mm_5deg": (0.001, math.radians(5.0))}
REACH_HOLD_S = 1.0
TARGET_END_MM, TARGET_END_DEG = 4.0, 5.0  # the success criteria (memory success-criteria-legs-arms)


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples along axis 0, the ends over what is there."""
    kernel = np.ones(n)
    total = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, x)
    count = np.convolve(np.ones(len(x)), kernel, mode="same").reshape((-1,) + (1,) * (x.ndim - 1))
    return total / count


def hold_metrics(err: np.ndarray, pos: np.ndarray, qd: np.ndarray, dt: float) -> dict:
    """One wrist's hold: err (T,) m, pos (T, 3) m, qd (T, 7) rad/s, at dt. Returns its scores (m, rad/s)."""
    smooth = moving_average(pos, max(1, int(round(SMOOTH_S / dt))))
    shake = pos - smooth
    t = np.arange(len(pos), dtype=float)
    slope, offset = np.polyfit(t, smooth, 1)
    sway = smooth - (np.outer(t, slope) + offset)
    return {"hold_error": float(err.mean()), "hold_error_std": float(err.std()),
            "vibration_rms": float(np.sqrt((shake ** 2).sum(1).mean())), "joint_speed_rms": float(np.sqrt((qd ** 2).mean())),
            "wobble_rms": float(np.sqrt((sway ** 2).sum(1).mean())), "wobble_p2p": float((sway.max(0) - sway.min(0)).max())}


def compute(err: np.ndarray, rot_t: np.ndarray, pos: np.ndarray, qd: np.ndarray, fell: np.ndarray, dt: float,
            env_reached: np.ndarray | None = None) -> dict:
    """err, rot_t (N, goals, steps, 2): each wrist's position (m) and rotation (rad) error; pos (N, goals, steps, 2,
    3) world wrist positions; qd (N, goals, steps, 14) arm joint speeds (left arm first); fell (N, goals) bool;
    env_reached (N, goals) the env's own reached flag, if it has one."""
    N, G, steps = err.shape[:3]
    hold, end = int(round(HOLD_S / dt)), int(round(END_S / dt))
    half = qd.shape[-1] // 2
    per = {k: np.zeros((N, G, 2)) for k in
           ("end_error", "hold_error", "hold_error_std", "vibration_rms", "joint_speed_rms", "wobble_rms", "wobble_p2p")}
    reach_t = np.full((N, G, 2), np.nan)
    for e in range(N):
        for g in range(G):
            for a in range(2):
                joints = slice(0, half) if a == 0 else slice(half, None)
                for key, value in hold_metrics(err[e, g, -hold:, a], pos[e, g, -hold:, a], qd[e, g, -hold:, joints], dt).items():
                    per[key][e, g, a] = value
                per["end_error"][e, g, a] = err[e, g, -end:, a].mean()
                within = np.nonzero(err[e, g, :, a] < REACH_POS)[0]
                if len(within):
                    reach_t[e, g, a] = (within[0] + 1) * dt
    held = int(round(REACH_HOLD_S / dt))
    reached_at = {}
    for name, (pos_tol, rot_tol) in REACH_TOLS.items():
        inside = ((err < pos_tol) & (rot_t < rot_tol)).all(axis=-1)
        run = np.apply_along_axis(lambda v: np.convolve(v, np.ones(held), mode="valid").max(), 2, inside.astype(float))
        reached_at[name] = run >= held
    keep = ~fell
    if not keep.any():
        raise SystemExit("every goal had a reset: nothing to score")
    rot_end = rot_t[:, :, -end:].mean(2)
    per = {k: v[keep] for k, v in per.items()}
    reach_t, rot_end = reach_t[keep], rot_end[keep]
    return {"num_envs": N, "goals": G, "holds": int(keep.sum() * 2), "goals_with_reset": int(fell.sum()), "policy_dt": dt,
            "reached": float(reached_at["5cm_20deg"][keep].mean()),
            "env_reached": float(env_reached[keep].mean()) if env_reached is not None else None,
            "reached_at": {name: float(v[keep].mean()) for name, v in reached_at.items()},
            "mean": {**{k: float(v.mean()) for k, v in per.items()}, "end_rot_error": float(rot_end.mean())},
            "per_arm": {k: [float(v[..., 0].mean()), float(v[..., 1].mean())] for k, v in per.items()},
            "end_error_std": float(per["end_error"].std()), "end_error_p90": float(np.percentile(per["end_error"], 90)),
            "reach_share": float(np.isfinite(reach_t).mean()),
            "time_to_reach_median": float(np.nanmedian(reach_t)) if np.isfinite(reach_t).any() else float("nan")}


def _tol(res: dict) -> str:
    t = res.get("reach_tol")
    return f"({1000 * t[0]:.0f} mm / {math.degrees(t[1]):.0f} deg, held {REACH_HOLD_S:g} s)" if t else "(held 1 s)"


def _rows_of(res: dict, ref: dict | None, sec: str, prefix: str = "") -> list[Row]:
    m, r = res.get("mean", {}), (ref or {}).get("mean", {})
    rr = (ref or {}).get("reached_at", {})

    def ref_of(v, scale=1.0):
        return None if ref is None else num(v) * scale
    mm, deg = 1000.0, 180.0 / math.pi
    rows = [
        Row(sec, prefix + "end_error_mm", "Position error at the goal's end, mean (mm)", num(m.get("end_error")) * mm,
            ref_of(r.get("end_error"), mm), TARGET_END_MM, "le", 2),
        Row(sec, prefix + "end_rot_error_deg", "Rotation error at the goal's end, mean (deg)",
            num(m.get("end_rot_error")) * deg, ref_of(r.get("end_rot_error"), deg), TARGET_END_DEG, "le", 2),
        Row(sec, prefix + "hold_error_mm", "Position error over the hold, mean (mm)", num(m.get("hold_error")) * mm,
            ref_of(r.get("hold_error"), mm), digits=2),
        Row(sec, prefix + "reached", "Reached 5 cm / 20 deg held 1 s, share",
            num(res.get("reached_at", {}).get("5cm_20deg")), ref_of(rr.get("5cm_20deg"))),
        Row(sec, prefix + "reached_1mm_5deg", "Reached 1 mm / 5 deg held 1 s, share",
            num(res.get("reached_at", {}).get("1mm_5deg")), ref_of(rr.get("1mm_5deg"))),
        Row(sec, prefix + "env_reached", "Reached at the env's own tolerance " + _tol(res) + ", share",
            num(res.get("env_reached")), ref_of((ref or {}).get("env_reached"))),
        Row(sec, prefix + "time_to_reach_s", "Time to within 5 cm, median (s)", num(res.get("time_to_reach_median")),
            ref_of((ref or {}).get("time_to_reach_median")), digits=2),
        Row(sec, prefix + "vibration_mm", "Vibration: wrist shake above ~10 Hz, RMS (mm)",
            num(m.get("vibration_rms")) * mm, ref_of(r.get("vibration_rms"), mm), digits=2),
        Row(sec, prefix + "joint_speed_rms", "Vibration: arm joint speed in the hold, RMS (rad/s)",
            num(m.get("joint_speed_rms")), ref_of(r.get("joint_speed_rms"))),
        Row(sec, prefix + "wobble_mm", "Wobble: slow sway, drift removed, RMS (mm)", num(m.get("wobble_rms")) * mm,
            ref_of(r.get("wobble_rms"), mm), digits=2),
        Row(sec, prefix + "wobble_p2p_mm", "Wobble: largest peak-to-peak (mm)", num(m.get("wobble_p2p")) * mm,
            ref_of(r.get("wobble_p2p"), mm), digits=2),
        Row(sec, prefix + "hold_error_std_mm", "Variance: error sd within a hold (mm)", num(m.get("hold_error_std")) * mm,
            ref_of(r.get("hold_error_std"), mm), digits=2),
        Row(sec, prefix + "end_error_p90_mm", "Variance: end error p90 across holds (mm)",
            num(res.get("end_error_p90")) * mm, ref_of((ref or {}).get("end_error_p90"), mm), digits=2),
        Row(sec, prefix + "goals_with_reset", "Goals with a fall", num(res.get("goals_with_reset")), digits=0),
    ]
    return rows


def rows(res: dict, ref: dict | None = None) -> list[Row]:
    """Rows of compute()'s result (or an older arms_eval json); ref: the IK-only result on the same goals."""
    out = _rows_of(res, ref, "Standing goals" + (" (the legs policy balancing)" if res.get("other_policy") else
                                                  " (pelvis pinned)"))
    if "walk" in res:
        out += _rows_of(res["walk"], None, "Goals while the legs walk", "walk.")
    if res.get("policy_std_mean") is not None:
        out.append(Row("Policy", "policy_std_mean", "Arms policy action std (exploration noise)",
                       num(res.get("policy_std_mean"))))
    return out


def header(res: dict, ref_path: str | None) -> str:
    kind = "zero arms actions (the IK alone)" if res.get("ik_only") else "sampled" if res.get("stochastic") else "mean"
    other = res.get("other_policy")
    lines = [f"# Arms scorecard ({res.get('backend', 'isaac')}): {res.get('run')} @ {res.get('step')}", "",
             f"{res['num_envs']} envs x {res['goals']} arm goals of {GOAL_S:g} s ({res['holds']} wrist holds); hold ="
             f" the last {HOLD_S:g} s; {kind} actions; {'off-table' if res.get('offtable') else 'table-row'} goals;"
             f" {res['goals_with_reset']} goals with a fall left out.",
             f"Legs: {other if other else 'none, the pelvis pinned'}.",
             f"Reference: {'the IK alone on the same goals (' + ref_path + ')' if ref_path else 'none (no IK-only scorecard for this run yet)'}."]
    if res.get("code_commit"):
        lines.append(f"Code {res['code_commit']}, conditions {res.get('conditions_hash')}.")
    return "\n".join(lines)


def footer(res: dict) -> str:
    mm = lambda v: f"{1000 * v:.1f}"  # noqa: E731
    p = res.get("per_arm", {})
    lines = []
    if p:
        lines.append(f"Per wrist (left / right): end error {mm(p['end_error'][0])} / {mm(p['end_error'][1])} mm,"
                     f" vibration {mm(p['vibration_rms'][0])} / {mm(p['vibration_rms'][1])} mm, wobble"
                     f" {mm(p['wobble_rms'][0])} / {mm(p['wobble_rms'][1])} mm.")
    if res.get("steerability"):
        lines += ["", "## Steerability (--steerability)", "", res["steerability"]]
    lines += ["", f"Run overrides: `{' '.join(res.get('overrides', [])) or '(none)'}`"]
    return "\n".join(lines)


PLOT_KEYS = ["end_error_mm", "end_rot_error_deg", "hold_error_mm", "reached_1mm_5deg", "reached", "time_to_reach_s",
             "vibration_mm", "wobble_mm", "hold_error_std_mm", "end_error_p90_mm", "goals_with_reset",
             "policy_std_mean"]
TRAIN_PANELS = [("Metrics/arm_targets/goal_position_error", "Training: goal position error (m)"),
                ("Metrics/arm_targets/goal_orientation_error", "Training: goal rotation error (rad)"),
                ("Reward / Total reward (mean)", "Total reward per episode"),
                ("Episode / Total timesteps (mean)", "Episode length (steps)"),
                ("Policy / Standard deviation (arms)", "Arms policy action std")]
