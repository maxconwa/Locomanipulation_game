"""The legs scorecard's metrics: sport mode's tracking numbers (sportmode.py, the report's own code), stride and
torso-tilt statistics, and the feet: how high and how often they lift. Numpy and scipy only; the Isaac and MuJoCo
scorecards and the GOLEM run feed it 100 Hz series, test_sportmode_metrics.py the real sport-mode bags.

Tracking: per axis the best-fit gain, delay and smoothing time constant of the velocity's response to the command;
then, on time spent stepping (sportmode.stepping, the knee-swing detector), the velocity averaged over one stride
against the command delayed and smoothed by that response (the lag-allowed command, without the gain or bias).

Feet (foot_stats): the foot-lift analysis' measures (logs/analysis/foot_lift/feet_metrics.py, 2026-10-26).
  pelvis frame, from the leg joint angles alone (so the real robot's and sport mode's bags give them too): the
    left-minus-right ankle height in the pelvis frame, levelled by the torso's roll and pitch (dz). Clearance = the
    swing ankle's height over the stance ankle around each knee-swing peak; per stride window (one sport-mode stride,
    70 samples) the largest |dz| and the knee's swing range; knee-swing peaks per second
  world frame, when the sim gives foot heights and contacts: the share of commanded foot-time in the air and with
    both feet down, air phases per second, their median duration and the peak lift of each air phase above the
    feet's median standing height
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from . import sportmode as S

DT = S.DT
REPORT_STRIDE = S.REPORT_STRIDE
TARGETS = S.TARGETS

# the H1-2 leg chain in LowCmd order (hip yaw, hip pitch, hip roll, knee, ankle pitch, ankle roll): each joint's
# offset from its parent (left leg; y mirrored for the right) and rotation axis (0 x, 1 y, 2 z)
LEG_CHAIN = [([0, 0.0875, -0.1632], 2), ([0, 0.0755, 0], 1), ([0, 0, 0], 0), ([0, 0, -0.4], 1), ([0, 0, -0.4], 1),
             ([0, 0, -0.02], 0)]


def tracking_safe(ref, vel, mask, stride: int = REPORT_STRIDE) -> dict:
    """sportmode.tracking, with what a subset has samples for when it lacks some (no turn, no walk)."""
    try:
        with np.errstate(all="ignore"):
            return S.tracking(ref, vel, mask, stride)
    except (IndexError, ValueError):
        m = np.concatenate(mask)
        if m.sum() == 0:
            return {"n_samples": 0}
        from scipy.ndimage import uniform_filter1d
        r = np.concatenate(ref)
        v = np.concatenate([uniform_filter1d(x, stride, axis=0) for x in vel])
        e = (v - r)[m]
        return {"lin_err_mean": float(np.linalg.norm(e[:, :2], axis=1).mean()),
                "ang_err_mean": float(np.abs(e[:, 2]).mean())}


def score(cmds, vels, masks, stride_samples: int = REPORT_STRIDE, subsets: dict | None = None,
          creep_removed: bool = False) -> dict:
    """The report's tracking numbers for one controller. cmds, vels: per series (T, 3) arrays (vx, vy, wz; velocity in
    the yaw frame), masks: per series (T,) bool (stepping and valid). subsets: name -> per-series masks, scored against
    the same fitted response. creep_removed: also score with the fitted vx bias removed from the velocity (the
    report's diagnostic for sport mode's forward creep)."""
    from scipy.ndimage import uniform_filter1d
    response = S.fit_resp(cmds, vels, masks, stride_samples)
    ref = S.lagged_ref(cmds, response, stride_samples)
    out = {
        "stride_samples": stride_samples,
        "response": response,
        "tracking_vs_lagged": tracking_safe(ref, vels, masks, stride_samples),
        "tracking_vs_command": tracking_safe(
            [uniform_filter1d(c, stride_samples, axis=0) for c in cmds], vels, masks, stride_samples),
    }
    if creep_removed:
        bias = np.array([response["vx"]["bias"], 0.0, 0.0])
        out["tracking_creep_removed"] = tracking_safe(ref, [v - bias for v in vels], masks, stride_samples)
    for name, sub in (subsets or {}).items():
        out.setdefault("subsets", {})[name] = tracking_safe(ref, vels, [m & s for m, s in zip(masks, sub)],
                                                            stride_samples)
    return out


def stride_stats(knee_series, masks) -> dict:
    """Stride period from each knee's swing peaks while stepping, as the report's gait_stats: peaks of prominence
    0.1 rad at least 0.25 s apart, periods under 2 s. knee_series: per series (T, 2) knee angles (left, right)."""
    periods = []
    for q, w in zip(knee_series, masks):
        t = np.arange(len(q)) * DT
        for j in range(2):
            pk, _ = find_peaks(q[:, j], prominence=0.1, distance=25)
            pk = pk[w[pk]]
            d = np.diff(t[pk])
            periods += d[d < 2].tolist()
    T = np.array(periods)
    if len(T) == 0:
        return {"stride_T_median": float("nan"), "stride_T_iqr": float("nan"), "n_strides": 0}
    return {"stride_T_median": float(np.median(T)), "stride_T_iqr": float(np.subtract(*np.percentile(T, [75, 25]))),
            "n_strides": int(len(T))}


def tilt_stats(rp_series, masks) -> dict:
    """Torso roll and pitch sd while stepping (deg), each series' mean removed, as the report's body_stats.
    rp_series: per series (T, 2) torso roll, pitch (rad)."""
    rp = [x[m] - x[m].mean(0) for x, m in zip(rp_series, masks) if m.sum() >= 100]
    if not rp:
        return {"roll_sd_deg": float("nan"), "pitch_sd_deg": float("nan")}
    rp = np.concatenate(rp)
    return {"roll_sd_deg": float(np.degrees(rp[:, 0].std())), "pitch_sd_deg": float(np.degrees(rp[:, 1].std()))}


def _rotm(axis, q):
    c, s = np.cos(q), np.sin(q)
    M = np.zeros((len(q), 3, 3))
    M[:, axis, axis] = 1
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    M[:, i, i] = c
    M[:, j, j] = c
    M[:, i, j] = -s
    M[:, j, i] = s
    return M


def ankle_in_pelvis(q: np.ndarray, side: int) -> np.ndarray:
    """(T, 3) the ankle roll joint in the pelvis frame from (T, 12) leg angles in LowCmd order; side 0 left, 1 right."""
    Rm = np.tile(np.eye(3), (len(q), 1, 1))
    p = np.zeros((len(q), 3))
    for k, (xyz, ax) in enumerate(LEG_CHAIN):
        xyz = np.array(xyz, dtype=float) * ([1, -1, 1] if side else 1)
        p = p + np.einsum("nij,j->ni", Rm, xyz)
        Rm = Rm @ _rotm(ax, q[:, 6 * side + k])
    return p


def dz_levelled(q: np.ndarray, rp: np.ndarray) -> np.ndarray:
    """(T,) left minus right ankle height in the pelvis frame, levelled by the torso's roll and pitch."""
    r = Rotation.from_euler("xyz", np.c_[rp[:, 0], rp[:, 1], np.zeros(len(q))])
    return r.apply(ankle_in_pelvis(q, 0))[:, 2] - r.apply(ankle_in_pelvis(q, 1))[:, 2]


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    return [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(r)] if len(idx) else []


def foot_stats(q_series, rp_series, masks, foot_z_series=None, contact_series=None, window: int = REPORT_STRIDE) -> dict:
    """The feet over every series. q: per series (T, 12) leg angles (LowCmd order), rp: (T, 2) torso roll, pitch,
    masks: (T,) bool, the samples to score (commanded, stepping, before a fall). foot_z: (T, 2) the soles' world
    heights and contact: (T, 2) bool, both optional (the world-frame measures need them)."""
    clr, kpk, win_dz, win_knee = [], [], [], []
    secs = 0.0
    for q, rp, mask in zip(q_series, rp_series, masks):
        dz = dz_levelled(q, rp)
        for j, sgn in ((3, 1), (9, -1)):
            pk, _ = find_peaks(q[:, j], prominence=0.1, distance=25)
            pk = pk[mask[pk]]
            kpk += q[pk, j].tolist()
            clr += [float((sgn * dz[max(p - 15, 0):p + 15]).max()) for p in pk]
        for run in _runs(mask):
            for a in range(0, len(run) - window + 1, window):
                s = slice(run[a], run[a] + window)
                win_dz.append(float(np.abs(dz[s]).max()))
                win_knee.append(float(max(np.ptp(q[s, 3]), np.ptp(q[s, 9]))))
        secs += mask.sum() * DT

    def pct(x, p, scale=1.0):
        return float(np.percentile(x, p) * scale) if len(x) else float("nan")

    out = {
        "seconds": float(secs),
        "steps_per_s": len(clr) / secs if secs > 0 else float("nan"),
        "clearance_median_cm": pct(clr, 50, 100), "clearance_p10_cm": pct(clr, 10, 100),
        "stride_dz_median_cm": pct(win_dz, 50, 100), "stride_dz_p90_cm": pct(win_dz, 90, 100),
        "knee_swing_median": pct(win_knee, 50), "knee_peak_median": pct(kpk, 50),
    }
    if foot_z_series is not None and contact_series is not None:
        down = [z[c] for z, c in zip(foot_z_series, contact_series) if c.any()]
        ground = float(np.median(np.concatenate(down))) if down else float("nan")
        lifts, durs, air, both, msum = [], [], 0, 0, 0
        for z, con, m in zip(foot_z_series, contact_series, masks):
            msum += int(m.sum())
            air += int((~con & m[:, None]).sum())
            both += int((con.all(1) & m).sum())
            for f in range(2):
                for run in _runs(~con[:, f] & m):
                    if len(run) >= 3:
                        lifts.append(float(z[run, f].max() - ground))
                        durs.append(len(run) * DT)
        out.update({
            "air_share": air / (2 * msum) if msum else float("nan"),
            "double_support_share": both / msum if msum else float("nan"),
            "air_phases_per_s": len(lifts) / (msum * DT) if msum else float("nan"),
            "air_phase_s_median": pct(durs, 50),
            "air_lift_median_cm": pct(lifts, 50, 100), "air_lift_p90_cm": pct(lifts, 90, 100),
        })
    return out


def commanded(cmd: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Samples with a non-zero command, as the report counts them: |vx| or |vy| or |wz| over 0.05."""
    return (np.abs(cmd[:, :2]).max(1) > threshold) | (np.abs(cmd[:, 2]) > threshold)


def verdict(value: float, target: float, sense: str) -> bool:
    return bool(np.isfinite(value) and (value >= target if sense == "ge" else value <= target))
