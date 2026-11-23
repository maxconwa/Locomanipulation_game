"""The legs scorecard, whatever ran it (Isaac or MuJoCo): the scripted velocity segments, and the scoring of the
recorded series against sport mode.

Every env stands SETTLE_S, then runs its own shuffled sequence of segments, each followed by GAP_S of zero command:
  step   a command held HOLD_S: each axis alone (vx +-0.2/0.4/0.6, vy +-0.15/0.25, wz +-0.2/0.35/0.5) and 8
         combinations, written for the baseline's ranges (vx +-0.7, vy +-0.3, wz +-0.5) and scaled per axis to the
         run's own (a run on sport mode's 0.55 / 0.54 / 0.97 gets vx 0.16-0.47, vy 0.27-0.45, wz 0.39-0.97)
  flick  the sport-mode operator's stick: PULSE_S pulses at the axis's full value (the run's range edge), one every
         U(0.3, 0.5) s, for FLICK_S; per axis and sign
  small  a command of norm U(0.05, 0.15) over (vx, vy, wz) in a random direction, held HOLD_S
Each segment type gets the same number of slots over all envs. The backend records per policy step: the pelvis
velocity in the yaw frame and the yaw rate, the 12 leg angles (LowCmd order), the torso's roll and pitch, e-stop
trips, the feet's world heights and contacts (if it can), and the step of the env's first fall. compute() resamples to
100 Hz (commands held, the rest interpolated), cuts each env at its fall, and scores.
"""

from __future__ import annotations

import numpy as np

from . import legs_metrics as lm
from . import sportmode as S
from .scorecard import Row, fmt, num

SETTLE_S, HOLD_S, FLICK_S, GAP_S = 2.0, 4.0, 5.0, 2.0
PULSE_S, PULSE_EVERY = 0.15, (0.3, 0.5)
BASE_FULL = (0.7, 0.3, 0.5)  # the baseline's range edges, which STEPS are written for
KINDS = ("step", "flick", "small")
STEPS = ([(s * v, 0.0, 0.0) for v in (0.2, 0.4, 0.6) for s in (1, -1)]
         + [(0.0, s * v, 0.0) for v in (0.15, 0.25) for s in (1, -1)]
         + [(0.0, 0.0, s * v) for v in (0.2, 0.35, 0.5) for s in (1, -1)]
         + [(0.4, 0.0, 0.3), (0.4, 0.0, -0.3), (0.3, 0.2, 0.0), (0.3, -0.2, 0.0), (-0.3, 0.0, 0.3),
            (-0.3, 0.0, -0.3), (0.3, 0.15, 0.3), (0.3, -0.15, -0.3)])
FLICKS = [(axis, sign) for axis in range(3) for sign in (1.0, -1.0)]
N_SMALL = 12


def program(num_envs: int, per_env: int, dt: float, rng: np.random.Generator, full: tuple[float, float, float]):
    """(commands (N, T, 3), kind index (N, T): -1 settle, -2 after the env's own program, else KINDS' index, the
    segment's gap included). full: the run's range edge per axis; the steps are scaled by full / BASE_FULL."""
    scale = np.asarray(full) / np.asarray(BASE_FULL)
    types = ([("step", tuple(np.asarray(c) * scale)) for c in STEPS] + [("flick", f) for f in FLICKS]
             + [("small", None)] * N_SMALL)
    slots = [types[i % len(types)] for i in range(num_envs * per_env)]
    rng.shuffle(slots)
    seqs = []
    for e in range(num_envs):
        cmd, kind = [np.zeros((int(round(SETTLE_S / dt)), 3))], [np.full(int(round(SETTLE_S / dt)), -1)]
        for name, spec in slots[e * per_env:(e + 1) * per_env]:
            if name == "step" or name == "small":
                if name == "small":
                    d = rng.normal(size=3)
                    spec = d / np.linalg.norm(d) * rng.uniform(0.05, 0.15)
                n = int(round(HOLD_S / dt))
                seg = np.tile(np.asarray(spec, float), (n, 1))
            else:
                axis, sign = spec
                n = int(round(FLICK_S / dt))
                seg, t = np.zeros((n, 3)), 0.0
                while t < FLICK_S:
                    a, b = int(round(t / dt)), int(round((t + PULSE_S) / dt))
                    seg[a:min(b, n), axis] = sign * full[axis]
                    t += rng.uniform(*PULSE_EVERY)
            gap = int(round(GAP_S / dt))
            cmd += [seg, np.zeros((gap, 3))]
            kind += [np.full(n + gap, KINDS.index(name))]
        seqs.append((np.concatenate(cmd), np.concatenate(kind)))
    T = max(len(c) for c, _ in seqs)
    cmds = np.zeros((num_envs, T, 3))
    kinds = np.full((num_envs, T), -2)
    for e, (c, k) in enumerate(seqs):
        cmds[e, :len(c)], kinds[e, :len(k)] = c, k
    return cmds, kinds


def to_100hz(x: np.ndarray, dt: float, hold: bool, n: int, offset: float) -> np.ndarray:
    """(T, ...) samples at times offset + k dt -> n samples at 0.01 s steps: held (the latest sample) or linear."""
    t_src = offset + np.arange(len(x)) * dt
    t = np.arange(n) * S.DT
    if hold:
        k = np.clip(np.searchsorted(t_src, t + 1e-9, side="right") - 1, 0, len(x) - 1)
        return x[k]
    flat = x.reshape(len(x), -1).astype(float)
    out = np.stack([np.interp(t, t_src, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def compute(cmds: np.ndarray, kinds: np.ndarray, rec: dict, fell_at: np.ndarray, dt: float, gait_period: float,
            estop: np.ndarray | None = None) -> dict:
    """Score a recording. cmds (N, T, 3) and kinds (N, T) from program(); rec: (N, T, ...) arrays "vel" (3), "q"
    (12), "rp" (2), and optionally "foot_z" (2) and "contact" (2, bool); fell_at (N,): the step of each env's first
    fall (T: none); estop (N, T) bool or None. dt: the policy step; gait_period: the run's stride (the averaging
    window)."""
    N, T = cmds.shape[:2]
    rp_all = np.unwrap(rec["rp"].astype(float), axis=1)
    has_feet = "foot_z" in rec and "contact" in rec
    series = []
    for e in range(N):
        end = int(fell_at[e])                   # steps 0..end-1 are valid (the fall step's state is the reset's)
        if end < 2 * int(round(1.0 / dt)):
            continue
        n = int(end * dt / S.DT)
        s = dict(cmd=to_100hz(cmds[e, :end], dt, True, n, 0.0),
                 kind=to_100hz(kinds[e, :end], dt, True, n, 0.0),
                 vel=to_100hz(rec["vel"][e, :end], dt, False, n, dt),
                 q=to_100hz(rec["q"][e, :end], dt, False, n, dt),
                 rp=to_100hz(rp_all[e, :end], dt, False, n, dt),
                 estop=to_100hz(estop[e, :end], dt, True, n, dt) if estop is not None else np.zeros(n, bool))
        if has_feet:
            s["foot_z"] = to_100hz(rec["foot_z"][e, :end], dt, False, n, dt)
            s["contact"] = to_100hz(rec["contact"][e, :end].astype(bool), dt, True, n, dt)
        series.append(s)
    masks = [S.stepping(s["q"]) for s in series]
    cmds100, vels100 = [s["cmd"] for s in series], [s["vel"] for s in series]
    stride = int(round(gait_period / S.DT))
    subsets = {kind: [s["kind"] == i for s in series] for i, kind in enumerate(KINDS)}
    res = lm.score(cmds100, vels100, masks, stride, subsets=subsets)
    commanded = [lm.commanded(c) for c in cmds100]
    small = subsets["small"]

    def seconds(ms):
        return float(sum(m.sum() for m in ms) * S.DT)

    feet_masks = [m & c for m, c in zip(masks, commanded)]
    res.update(
        num_envs=N, policy_dt=dt, gait_period=gait_period,
        seconds_valid=seconds([np.ones(len(c), bool) for c in cmds100]), seconds_stepping=seconds(masks),
        seconds_commanded=seconds(commanded),
        commanded_not_stepping_share=seconds([c & ~m for c, m in zip(commanded, masks)]) / max(seconds(commanded), 1e-9),
        small_not_stepping_share=seconds([c & s & ~m for c, s, m in zip(commanded, small, masks)])
        / max(seconds([c & s for c, s in zip(commanded, small)]), 1e-9),
        falls=int((fell_at < T).sum()),
        falls_by_kind={name: int(sum(1 for e in range(N) if fell_at[e] < T and kinds[e, fell_at[e]] == i))
                       for i, name in enumerate(("after_program", "settle") + KINDS, start=-2)},
        falls_per_min=float((fell_at < T).sum() / (np.minimum(fell_at, T).sum() * dt / 60)),
        estop_share=float(np.mean([s["estop"].mean() for s in series])) if series else float("nan"),
        gait=lm.stride_stats([s["q"][:, [3, 9]] for s in series], masks),
        tilt=lm.tilt_stats([s["rp"] for s in series], masks),
        feet=lm.foot_stats([s["q"] for s in series], [s["rp"] for s in series], feet_masks,
                           [s["foot_z"] for s in series] if has_feet else None,
                           [s["contact"] for s in series] if has_feet else None),
    )
    return res


def rows(res: dict) -> list[Row]:
    """The scorecard's rows (scorecard.Row) from compute()'s result, or from an older scorecard json (missing
    metrics are NaN)."""
    lag, ref = res.get("tracking_vs_lagged", {}), S.SPORT_REF
    out = []
    sec = "Tracking: the lag-allowed command, while stepping"
    for key, label, sport, target, sense in S.TARGETS:
        out.append(Row(sec, key, label, num(lag.get(key)), sport, target, sense))
    out += [Row(sec, "lin_err_p95", "Linear velocity error, p95 (m/s)", num(lag.get("lin_err_p95")), 0.221),
            Row(sec, "ang_err_p95", "Angular velocity error, p95 (rad/s)", num(lag.get("ang_err_p95")), 0.122)]
    sec = "Response fit per axis"
    for axis, fit in res.get("response", {}).items():
        out += [Row(sec, f"{axis}_gain", f"{axis} gain", num(fit.get("gain"))),
                Row(sec, f"{axis}_delay", f"{axis} delay (s)", num(fit.get("delay")), digits=2),
                Row(sec, f"{axis}_tau", f"{axis} tau (s)", num(fit.get("tau")), digits=2),
                Row(sec, f"{axis}_bias", f"{axis} bias", num(fit.get("bias")))]
    g, b, f = res.get("gait", {}), res.get("tilt", {}), res.get("feet", {})
    sec = "Gait and feet (while commanded and stepping)"
    out += [Row(sec, "stride_T_median", "Stride period, median (s)", num(g.get("stride_T_median")), ref["stride_T_median"]),
            Row(sec, "stride_T_iqr", "Stride period, IQR (s)", num(g.get("stride_T_iqr")), ref["stride_T_iqr"]),
            Row(sec, "steps_per_s", "Steps per second (knee peaks)", num(f.get("steps_per_s")), ref["steps_per_s"], digits=2),
            Row(sec, "clearance_median_cm", "Swing clearance at the knee peak, median (cm)",
                num(f.get("clearance_median_cm")), ref["clearance_median_cm"], digits=1),
            Row(sec, "clearance_p10_cm", "Swing clearance at the knee peak, p10 (cm)",
                num(f.get("clearance_p10_cm")), ref["clearance_p10_cm"], digits=1),
            Row(sec, "stride_dz_median_cm", "Ankle height difference per stride, median (cm)",
                num(f.get("stride_dz_median_cm")), ref["stride_dz_median_cm"], digits=1),
            Row(sec, "stride_dz_p90_cm", "Ankle height difference per stride, p90 (cm)",
                num(f.get("stride_dz_p90_cm")), ref["stride_dz_p90_cm"], digits=1),
            Row(sec, "knee_swing_median", "Knee swing per stride, median (rad)", num(f.get("knee_swing_median")),
                ref["knee_swing_median"], digits=2),
            Row(sec, "knee_peak_median", "Knee swing peak, median (rad)", num(f.get("knee_peak_median")),
                ref["knee_peak_median"], digits=2),
            Row(sec, "air_share", "Foot in the air, share of foot-time", num(f.get("air_share"))),
            Row(sec, "double_support_share", "Both feet down, share of time", num(f.get("double_support_share"))),
            Row(sec, "air_phases_per_s", "Air phases per second", num(f.get("air_phases_per_s")), digits=2),
            Row(sec, "air_phase_s_median", "Air phase duration, median (s)", num(f.get("air_phase_s_median"))),
            Row(sec, "air_lift_median_cm", "Peak lift per air phase, median (cm)", num(f.get("air_lift_median_cm")), digits=1),
            Row(sec, "air_lift_p90_cm", "Peak lift per air phase, p90 (cm)", num(f.get("air_lift_p90_cm")), digits=1)]
    sec = "Posture and safety"
    out += [Row(sec, "roll_sd_deg", "Torso roll sd while stepping (deg)", num(b.get("roll_sd_deg")), ref["roll_sd_deg"], digits=2),
            Row(sec, "pitch_sd_deg", "Torso pitch sd while stepping (deg)", num(b.get("pitch_sd_deg")), ref["pitch_sd_deg"], digits=2),
            Row(sec, "falls_per_min", "Falls per env-minute", num(res.get("falls_per_min"))),
            Row(sec, "estop_share", "GOLEM e-stop trips, share of steps", num(res.get("estop_share")), digits=4),
            Row(sec, "commanded_not_stepping_share", "Commanded but not stepping, share",
                num(res.get("commanded_not_stepping_share"))),
            Row(sec, "small_not_stepping_share", "Small commands not stepped, share",
                num(res.get("small_not_stepping_share")))]
    sec = "By segment kind"
    for kind, t in res.get("subsets", {}).items():
        for key, label in (("cos_v_mean", "cos v"), ("turn_sign_agree", "turn sense"), ("lin_err_mean", "lin err"),
                           ("ang_err_mean", "ang err"), ("speed_ratio_median", "speed ratio"),
                           ("turn_ratio_median", "turn ratio")):
            out.append(Row(sec, f"{kind}.{key}", f"{kind}: {label}", num(t.get(key))))
    for r in res.get("coupled_rows", []):
        out.append(Row(**r))
    return out


def header(res: dict) -> str:
    full = res.get("full", [float("nan")] * 3)
    other = res.get("other_policy")
    lines = [f"# Legs scorecard ({res.get('backend', 'isaac')}): {res.get('run')} @ {res.get('step')}", "",
             f"{res['num_envs']} envs, {res['seconds_valid']:.0f} s of valid data, {res['seconds_stepping']:.0f} s"
             f" stepping; policy {1 / res['policy_dt']:.0f} Hz, {'sampled' if res.get('stochastic') else 'mean'}"
             f" actions, stride window {res['stride_samples'] * S.DT:.2f} s. Scored against the lag-allowed command"
             f" (no creep correction). Commands at the run's ranges: full stick vx {full[0]:g}, vy {full[1]:g},"
             f" wz {full[2]:g}. Reference: sport mode (the H1-2 Sport Mode Walking Report).",
             f"Arms: {other if other else 'held at 0 rad (no arms policy)'}.",
             f"Falls: {res['falls']} of {res['num_envs']} envs (by the segment they fell in: "
             f"{', '.join(f'{k} {v}' for k, v in res['falls_by_kind'].items())})."]
    if res.get("code_commit"):
        lines.append(f"Code {res['code_commit']}, conditions {res.get('conditions_hash')}.")
    return "\n".join(lines)


def footer(res: dict) -> str:
    lines = []
    w = res.get("wrist")
    if w:
        lines += [f"Wrist tracking ({w['goals']} standing arm goals, goals during a fall left out: {w['fell']}):"
                  f" reached {100 * w['reached']:.1f}%, closest approach {100 * w['best_error']:.1f} cm, final error"
                  f" {100 * w['final_error']:.1f} cm / {w['final_rot_error']:.2f} rad", ""]
    lines.append(f"Run overrides: `{' '.join(res.get('overrides', [])) or '(none)'}`")
    return "\n".join(lines)


def fmt_line(res: dict) -> str:
    lag = res.get("tracking_vs_lagged", {})
    return " ".join(f"{k} {fmt(num(lag.get(k)))}" for k, *_ in S.TARGETS)


# the comparison plots' panels (row keys), direction and turn sense first
PLOT_KEYS = ["cos_v_mean", "turn_sign_agree", "lin_err_mean", "ang_err_mean", "speed_ratio_median",
             "turn_ratio_median", "falls_per_min", "stride_T_median", "steps_per_s", "clearance_median_cm",
             "stride_dz_median_cm", "knee_swing_median", "air_share", "roll_sd_deg", "pitch_sd_deg",
             "commanded_not_stepping_share"]
TRAIN_PANELS = [("Metrics/sport/cos", "Training: direction cos"), ("Metrics/sport/turn_sense", "Training: turn sense"),
                ("Metrics/sport/lin_err", "Training: linear error (m/s)"),
                ("Metrics/sport/ang_err", "Training: angular error (rad/s)"),
                ("Reward / Total reward (mean)", "Total reward per episode"),
                ("Episode / Total timesteps (mean)", "Episode length (steps)"),
                ("Episode_Termination/fell", "Episodes ending in a fall"),
                ("Policy / Standard deviation (legs)", "Legs policy action std")]
